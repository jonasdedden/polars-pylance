"""Machine-wide CPU accounting, because the workers are not our children.

`resource.getrusage` sees this process and reaped children, which is the
wrong set twice over: Ray's workers are not children of the driver at all,
and Dask's are only reaped at shutdown. Both would report a coordinator
sitting idle while eight worker processes saturate the machine.

`/proc/stat` counts every core instead, so the difference across a run is
the CPU time the run cost however many processes it spread over. The price
is that it also counts anything else running: this is a number for an
otherwise idle benchmark host, and `idle_before` is reported so a reader can
see whether the host was in fact idle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

#: Kernel counters are in USER_HZ ticks; 100/s on every Linux worth running.
TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")
#: Fields of `/proc/stat`'s aggregate line, in order.
_IDLE_FIELDS = frozenset({3, 4})  # idle, iowait


def _totals() -> tuple[float, float]:
    """Seconds of (busy, idle) CPU across every core since boot."""
    fields = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    busy = idle = 0.0
    for index, value in enumerate(fields):
        seconds = int(value) / TICKS_PER_SECOND
        if index in _IDLE_FIELDS:
            idle += seconds
        else:
            busy += seconds
    return busy, idle


class CpuUsage(TypedDict):
    """What one measured interval cost the machine's CPUs."""

    cpu_seconds: float
    cpu_utilisation: float
    cores_busy: float
    idle_seconds: float


@dataclass
class Usage:
    """A CPU-time measurement spanning a run, taken from `/proc/stat`."""

    busy_before: float
    idle_before: float

    @classmethod
    def start(cls) -> Usage:
        """Begin measuring. Call `stop` after the run to close the interval."""
        busy, idle = _totals()
        return cls(busy_before=busy, idle_before=idle)

    def stop(self, elapsed: float) -> CpuUsage:
        """The interval's CPU cost, and what share of the machine it took.

        `cpu_seconds` is what the run spent on every core together, so
        dividing by wall time gives the average number of cores busy;
        `cpu_utilisation` scales that to the machine, where 1.0 is every
        core busy for the whole run.
        """
        busy, idle = _totals()
        cores = os.cpu_count() or 1
        cpu_seconds = busy - self.busy_before
        # Whether the *host* was busy with something else before we started
        # is not knowable from one reading, but an interval whose idle time
        # barely moved says the machine was ours for the duration.
        idle_seconds = idle - self.idle_before
        return CpuUsage(
            cpu_seconds=cpu_seconds,
            cpu_utilisation=cpu_seconds / (elapsed * cores) if elapsed else 0.0,
            cores_busy=cpu_seconds / elapsed if elapsed else 0.0,
            idle_seconds=idle_seconds,
        )
