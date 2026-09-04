"""polars-pylance on Ray Core: ship lazy queries, commit fragments once."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, overload

from bench.dataframe import backends, queries

from . import sharded

# Read at Ray's import time, so this has to run before the first `import
# ray` anywhere in the process -- which is why it sits here and not in
# `connect`. Ray's uv integration uploads the driver's whole working
# directory so workers can re-resolve the uv environment: a FUSE-tree walk
# plus staging on every `ray.init`, flaky under load and pure overhead
# here, where task functions ship by value and every dependency is already
# installed in the workers' own venv (polars-pylance reaches it through
# its editable .pth on the shared filesystem).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    import pyarrow as pa
    from lance.fragment import FragmentMetadata

    from bench.dataframe.backends import Parallelism
    from bench.dataframe.queries import CaseName, ReadCaseName, WriteCaseName

    from .sharded import ShardReducer, ShardWriter


def connect(budget: Parallelism, cluster: str | None) -> None:
    """Attach to Ray, starting this process's cluster at most once.

    One process keeps one cluster for its whole life: shutting it down
    between runs would strand anything holding handles into it -- Daft's
    runner keeps its worker actors across queries, so a second run on a
    second cluster dies with "from a different cluster". Process exit
    reaps a self-started cluster; a named cluster was never ours to stop.

    Sizing arguments only apply to a cluster we start: Ray rejects
    `num_cpus` outright when connecting to an existing one, whose CPUs and
    worker environment were settled by whoever started it -- so on a shared
    cluster the budget reaches the tasks (through their `num_cpus`) but the
    Polars pool size has to be in the workers' own environment already.
    """
    import ray

    if ray.is_initialized():
        return
    if cluster is not None:
        ray.init(address=cluster)
    else:
        ray.init(
            num_cpus=budget.cpus,
            # Polars reads this at import, so it has to be in the worker's
            # environment before the worker starts, not set from the task.
            runtime_env={"env_vars": budget.worker_env},
        )


def warm(budget: Parallelism) -> None:
    """Start the worker processes and import their dependencies.

    Ray starts workers on first use, so without this the first shard pays
    for process launch and a Polars import. That is cluster startup, which
    the measurement deliberately excludes.
    """
    import ray

    task = ray.remote(num_cpus=budget.threads)(queries.import_dependencies)
    ray.get([task.remote() for _ in range(budget.slots)])


@overload
def run(
    src: str,
    dst: str | None,
    *,
    case: ReadCaseName,
    n_shards: int,
    chunk_size: int = ...,
    parallelism: Parallelism | None = ...,
    cluster: str | None = ...,
) -> backends.ReadResult: ...
@overload
def run(
    src: str,
    dst: str | None,
    *,
    case: WriteCaseName,
    n_shards: int,
    chunk_size: int = ...,
    parallelism: Parallelism | None = ...,
    cluster: str | None = ...,
) -> backends.WriteResult: ...
@overload
def run(
    src: str,
    dst: str | None,
    *,
    case: CaseName,
    n_shards: int,
    chunk_size: int = ...,
    parallelism: Parallelism | None = ...,
    cluster: str | None = ...,
) -> backends.WriteResult | backends.ReadResult: ...
def run(
    src: str,
    dst: str | None,
    *,
    case: CaseName,
    n_shards: int,
    chunk_size: int = queries.DEFAULT_CHUNK_SIZE,
    parallelism: Parallelism | None = None,
    cluster: str | None = None,
) -> backends.WriteResult | backends.ReadResult:
    """Run one case of the sharded pipeline as one Ray task per shard."""
    import ray

    budget = parallelism or backends.Parallelism.from_env()
    connect(budget, cluster)
    warm(budget)
    spec = queries.CASES[case]
    if spec.kind == "read":
        if dst is not None:
            msg = f"read case {case!r} writes nothing; dst must be None"
            raise SystemExit(msg)

        def map_reduce(
            reduce: ShardReducer, shards: Sequence[pl.LazyFrame]
        ) -> list[dict[str, int]]:
            task = ray.remote(num_cpus=budget.threads)(reduce)
            return ray.get([task.remote(s) for s in shards])

        return sharded.reduce(
            src,
            backend="ray-core",
            case=case,
            n_shards=n_shards,
            map_reduce=map_reduce,
            budget=budget,
        )

    def map_shards(
        write: ShardWriter,
        shards: Sequence[pl.LazyFrame],
        target: str,
        arrow_schema: pa.Schema,
        chunk_size: int,
    ) -> list[list[FragmentMetadata]]:
        # `num_cpus` is what limits concurrency: the budget's CPUs
        # divided by a task's share is exactly its `slots`.
        task = ray.remote(num_cpus=budget.threads)(write)
        return ray.get(
            [task.remote(s, target, arrow_schema, chunk_size) for s in shards]
        )

    assert dst is not None  # narrowed by the read branch above
    return sharded.run(
        src,
        dst,
        backend="ray-core",
        case=case,
        n_shards=n_shards,
        chunk_size=chunk_size,
        map_shards=map_shards,
        budget=budget,
    )


if __name__ == "__main__":
    queries.smoke("ray-core", run)
