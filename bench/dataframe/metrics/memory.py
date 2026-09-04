"""Peak memory for a whole workload, not just the process that started it.

`RUSAGE_SELF` sees one process. A distributed run spends its memory in
workers that are children of a nanny, or of a raylet, or of nothing the
driver can name -- so the driver's own high-water mark can be a rounding
error next to what the machine actually held.

Two ways to see the whole thing, in order of preference:

* **cgroup v2.** Every descendant is charged to the same cgroup, so
  `memory.current` is exactly the workload and nothing else. This is what
  `systemd-run --scope` gives the benchmark for free, and it is the only
  method that stays correct on a host shared with anything.
* **The host.** `MemTotal - MemAvailable` from `/proc/meminfo`, which is a
  fair proxy on a dedicated node: `MemAvailable` already discounts
  reclaimable page cache, so reading a 50 GiB dataset does not show up as
  50 GiB of workload. It does count every other process on the box, which
  is why the sampler reports its own starting point for subtraction.

Both are sampled rather than read once, because peak memory is a moment,
not a total, and neither source records a high-water mark that resets.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, final

GIB = 1024**3
#: Fast enough to catch a spike in a phase lasting a second or so, cheap
#: enough that the sampler itself never shows up in the CPU numbers.
INTERVAL_SECONDS = 0.05

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_MEMINFO = Path("/proc/meminfo")


def _cgroup_dir() -> Path | None:
    """This process's cgroup v2 directory, if it has usable memory files."""
    try:
        # cgroup v2 puts a single "0::<path>" line in this file.
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, _, path = line.split(":", 2)
            if hierarchy != "0":
                continue
            directory = _CGROUP_ROOT / path.lstrip("/")
            if (directory / "memory.current").exists():
                return directory
    except OSError:
        return None
    return None


def _meminfo_used() -> float:
    """Bytes the host has in use, counting reclaimable cache as free."""
    total = available = 0
    for line in _MEMINFO.read_text().splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
        elif line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    return float(total - available)


@dataclass
class _CgroupSource:
    """Memory as the kernel charges it to one cgroup and its descendants."""

    directory: Path
    name: str = "cgroup"

    def read(self) -> tuple[float, float]:
        """(workload bytes, total bytes charged) -- anon, and anon + cache."""
        total = float(int((self.directory / "memory.current").read_text()))
        anon = 0.0
        for line in (self.directory / "memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "anon":
                anon = float(int(value))
                break
        return anon, total


@dataclass
class _HostSource:
    """Memory as the host reports it, for a run that owns the machine."""

    name: str = "host"

    def read(self) -> tuple[float, float]:
        used = _meminfo_used()
        return used, used


def _source() -> _CgroupSource | _HostSource:
    directory = _cgroup_dir()
    return _CgroupSource(directory) if directory is not None else _HostSource()


class MemoryRecord(TypedDict):
    """What one measured interval cost in memory, however it was read."""

    mem_source: str
    mem_peak_gib: float
    mem_rise_gib: float
    mem_peak_with_cache_gib: float
    mem_start_gib: float


@final
@dataclass
class Sampler:
    """Watches memory for the length of a `with` block, reporting the peak."""

    source: _CgroupSource | _HostSource = field(default_factory=_source)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _peak_workload: float = 0.0
    _peak_total: float = 0.0
    _start_workload: float = 0.0

    def __enter__(self) -> Sampler:
        """Take the starting reading and begin polling in the background."""
        self._start_workload, total = self.source.read()
        self._peak_workload, self._peak_total = self._start_workload, total
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop polling. The peaks stay readable afterwards."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        while not self._stop.wait(INTERVAL_SECONDS):
            workload, total = self.source.read()
            self._peak_workload = max(self._peak_workload, workload)
            self._peak_total = max(self._peak_total, total)

    def as_record(self) -> MemoryRecord:
        """The peaks, plus what they were measured with and started from.

        `mem_peak_gib` is the number to compare between backends: memory in
        use at the worst moment, across every process. `mem_rise_gib` is
        that minus where the run started, which is what the workload itself
        added; on a dedicated host with a cgroup they are nearly the same,
        and on a busy laptop the rise is the honest figure.
        """
        return MemoryRecord(
            mem_source=self.source.name,
            mem_peak_gib=self._peak_workload / GIB,
            mem_rise_gib=(self._peak_workload - self._start_workload) / GIB,
            mem_peak_with_cache_gib=self._peak_total / GIB,
            mem_start_gib=self._start_workload / GIB,
        )


def sample_baseline(seconds: float = 1.0) -> MemoryRecord:
    """What memory looks like with nothing running, for subtraction."""
    with Sampler() as sampler:
        time.sleep(seconds)
    return sampler.as_record()
