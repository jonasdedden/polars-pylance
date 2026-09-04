"""One measured region: wall time, CPU across the machine, peak memory.

Kept in one place so every backend reports the same fields over the same
interval, and so "what counts as the query" is decided once. Cluster
startup belongs outside it: a benchmark of the query should not be a
benchmark of Ray's boot time.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

from . import cpu, memory
from .cpu import CpuUsage
from .memory import MemoryRecord

if TYPE_CHECKING:
    from collections.abc import Generator


class Measurement(CpuUsage, MemoryRecord):
    """One measured interval: wall time plus the CPU and memory records."""

    seconds: float


@contextlib.contextmanager
def measured() -> Generator[Measurement, None, None]:
    """Measure a block. The dict is filled in on exit, so read it after."""
    # Placeholders, overwritten on exit: the caller only reads the dict after
    # the `with` block, by which point every field below has been replaced.
    # Starting from a complete record (rather than `{}`) keeps this cast-free.
    metrics: Measurement = {
        "seconds": 0.0,
        "cpu_seconds": 0.0,
        "cpu_utilisation": 0.0,
        "cores_busy": 0.0,
        "idle_seconds": 0.0,
        "mem_source": "",
        "mem_peak_gib": 0.0,
        "mem_rise_gib": 0.0,
        "mem_peak_with_cache_gib": 0.0,
        "mem_start_gib": 0.0,
    }
    with memory.Sampler() as ram:
        usage = cpu.Usage.start()
        start = time.perf_counter()
        try:
            yield metrics
        finally:
            elapsed = time.perf_counter() - start
            metrics["seconds"] = elapsed
            cpu_record = usage.stop(elapsed)
            metrics["cpu_seconds"] = cpu_record["cpu_seconds"]
            metrics["cpu_utilisation"] = cpu_record["cpu_utilisation"]
            metrics["cores_busy"] = cpu_record["cores_busy"]
            metrics["idle_seconds"] = cpu_record["idle_seconds"]
    mem_record = ram.as_record()
    metrics["mem_source"] = mem_record["mem_source"]
    metrics["mem_peak_gib"] = mem_record["mem_peak_gib"]
    metrics["mem_rise_gib"] = mem_record["mem_rise_gib"]
    metrics["mem_peak_with_cache_gib"] = mem_record["mem_peak_with_cache_gib"]
    metrics["mem_start_gib"] = mem_record["mem_start_gib"]
