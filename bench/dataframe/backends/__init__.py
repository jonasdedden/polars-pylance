"""The backend registry: what exists, and how to load one.

Every backend module exposes the same `run(src, dst, *, n_shards,
chunk_size, cluster)`, so the driver picks one by name and calls it without
knowing which. Loading is lazy because the three do not share dependencies:
running the Dask backend must not require Ray to be installed.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict, overload

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bench.dataframe.queries import CaseName, ReadCaseName, WriteCaseName

#: In-process first, then the ones that leave the process.
NAMES = ("threads", "daft", "ray-core", "dask", "ray-data", "daft-ray")

#: Tiers at or below this size run the local backends; tiers at or above it
#: run the distributed ones, against the cluster behind `DIST_CLUSTER`. A
#: tier exactly here runs both groups, which is what makes
#: local-vs-distributed comparable on identical data.
LOCAL_MAX_ROWS = 200_000_000  # ~100 GiB at the measured 0.5072 GiB/Mrow

#: Backends that run without a cluster, on 1-100 GiB tiers: the
#: ThreadPoolExecutor baseline, Daft's threaded runner, and Ray Data on a
#: local Ray cluster.
LOCAL = ("threads", "daft", "ray-data")

#: Backends that need a real cluster, on 100-1000 GiB tiers: Daft on Ray,
#: Ray Data on the same cluster, and polars-pylance on Ray Core and Dask.
DISTRIBUTED = ("ray-core", "dask", "ray-data", "daft-ray")


def tier_backends(rows: int) -> list[str]:
    """Which backends run a tier of `rows` rows, in table order."""
    return [
        name
        for name in NAMES
        if (name in LOCAL and rows <= LOCAL_MAX_ROWS)
        or (name in DISTRIBUTED and rows >= LOCAL_MAX_ROWS)
    ]


@dataclass(frozen=True)
class Parallelism:
    """The CPU budget a run gets, split into shards and threads per shard.

    Comparing schedulers is only meaningful when they are given the same
    machine, and left to themselves they do not take the same one. Measured
    on 8 cores with default settings: Ray Core runs 8 tasks at once in 8
    processes and each builds a full 8-thread Polars pool (64 threads), while
    Dask runs the same 8 tasks in 4 processes that share 4 pools (32
    threads). Neither honours `OMP_NUM_THREADS`, which both set to 1, because
    Polars reads `POLARS_MAX_THREADS` instead.

    So the budget is stated rather than inherited: `slots` shards run at
    once, each with `threads` Polars threads, and every backend is
    configured to that. `slots * threads` is then the same number of runnable
    threads on every backend, which is the thing a fair comparison holds
    equal.
    """

    slots: int
    threads: int

    @classmethod
    def from_env(cls) -> Parallelism:
        """Read the budget from `DIST_SLOTS` / `DIST_THREADS`.

        Defaults to one shard per core with one thread each: the shape an
        embarrassingly parallel job wants, and the one where "did it use the
        machine" is easiest to read off the CPU numbers.
        """
        cores = os.cpu_count() or 1
        slots = int(os.environ.get("DIST_SLOTS", str(cores)))
        threads = int(os.environ.get("DIST_THREADS", str(max(1, cores // slots))))
        return cls(slots=slots, threads=threads)

    @property
    def cpus(self) -> int:
        """The whole budget: how many threads may run at once."""
        return self.slots * self.threads

    @property
    def worker_env(self) -> dict[str, str]:
        """What a worker process needs in its environment to honour this.

        Polars sizes its thread pool once, at import, from
        `POLARS_MAX_THREADS`; setting it after the fact does nothing, so it
        has to reach the worker as an environment variable.
        """
        return {"POLARS_MAX_THREADS": str(self.threads)}

    def as_record(self) -> BudgetRecord:
        """The budget, for the measurement record it has to be read with."""
        return BudgetRecord(
            slots=self.slots,
            worker_threads=self.threads,
            cpu_budget=self.cpus,
        )


class BudgetRecord(TypedDict):
    """The CPU budget, as stored on every measurement record."""

    slots: int
    worker_threads: int
    cpu_budget: int


class BaseResult(TypedDict):
    """Fields every backend record carries, whatever the case kind.

    Budget (`slots`, `worker_threads`, `cpu_budget`) and measurement
    (`seconds` and the rest) are merged in by each backend; see
    `Parallelism.as_record` and `metrics.measured`.
    """

    backend: str
    commit_seconds: float
    plan_bytes: int
    metadata_bytes: int
    n_shards: int
    slots: int
    worker_threads: int
    cpu_budget: int
    seconds: float
    cpu_seconds: float
    cpu_utilisation: float
    cores_busy: float
    idle_seconds: float
    mem_source: str
    mem_peak_gib: float
    mem_rise_gib: float
    mem_peak_with_cache_gib: float
    mem_start_gib: float


class WriteResult(BaseResult):
    """A write case: how many rows landed, in how many fragments."""

    result_rows: int
    n_fragments: int


class ReadResult(BaseResult):
    """A read case: the reduction every backend must agree on exactly."""

    result_count: int
    result_sum_id: int


class Backend(Protocol):
    """One backend's entry point: run the pipeline, return one record.

    `case` names the query in `queries.CASES`. `dst` is where a write
    case lands and must be None for a read case, which writes nothing.
    `n_shards` is how the *data* is divided, `parallelism` how the
    *machine* is; they are independent, and more shards than slots is
    the normal case, since it lets a scheduler even out uneven
    fragments.

    `cluster` is the address of an already-running cluster (a Ray
    address, a Dask scheduler); omitted, the backend starts one for the
    run and shuts it down afterwards, which puts startup inside the
    measured wall time. A cluster this process did not start is
    configured by whoever did: `parallelism` then reaches the tasks but
    not the worker processes.

    The overloads make the record kind precise: a read case returns
    `ReadResult`, a write case `WriteResult`. A dynamic `CaseName` (the
    driver reads it from argv) returns the union.
    """

    @overload
    def __call__(
        self,
        src: str,
        dst: str | None,
        *,
        case: ReadCaseName,
        n_shards: int,
        chunk_size: int,
        parallelism: Parallelism | None = None,
        cluster: str | None = None,
    ) -> ReadResult: ...
    @overload
    def __call__(
        self,
        src: str,
        dst: str | None,
        *,
        case: WriteCaseName,
        n_shards: int,
        chunk_size: int,
        parallelism: Parallelism | None = None,
        cluster: str | None = None,
    ) -> WriteResult: ...
    @overload
    def __call__(
        self,
        src: str,
        dst: str | None,
        *,
        case: CaseName,
        n_shards: int,
        chunk_size: int,
        parallelism: Parallelism | None = None,
        cluster: str | None = None,
    ) -> WriteResult | ReadResult: ...


def load(name: str) -> Backend:
    """Import one backend's `run`. Only its own dependencies are needed."""
    if name not in NAMES:
        msg = f"unknown backend {name!r}; expected one of {', '.join(NAMES)}"
        raise SystemExit(msg)
    module = importlib.import_module(f"{__package__}.{name.replace('-', '_')}")
    run: Backend = module.run
    return run


def output_uri(root: str, backend: str, case: CaseName, rows: int) -> str:
    """Where a (backend, case, tier) run writes. One URI, agreed by all."""
    stem = f"out-{backend}-{case.value}-{rows // 1_000_000}m-dist.lance"
    return f"{root.rstrip('/')}/{stem}"


def source_uri(root: str, rows: int) -> str:
    """The ladder tier a run reads, as `bench/gen.py` named it."""
    return f"{root.rstrip('/')}/data/{rows // 1_000_000}m.lance"


def outputs(
    root: str, case: CaseName, rows: int, names: Iterable[str] = NAMES
) -> dict[str, str]:
    """Every backend's output URI for one (case, tier), keyed by backend."""
    return {name: output_uri(root, name, case, rows) for name in names}
