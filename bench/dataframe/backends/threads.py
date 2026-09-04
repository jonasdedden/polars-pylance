"""polars-pylance in one process: `write_lance_fragments` on a thread pool.

The library's own sharded path, and the baseline every distributed backend
has to beat. It runs the same shards as `ray-core` and `dask` and commits
the same way, but the shards never leave the process, so there is nothing
to serialize, no worker to start, and no scheduler in the middle. What a
distributed backend adds over this number is what distribution costs; what
it takes off is what distribution buys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from bench.dataframe import backends, queries
from bench.dataframe.metrics import measured

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from bench.dataframe.backends import Parallelism
    from bench.dataframe.queries import CaseName, ReadCaseName, WriteCaseName

    from .sharded import ShardReducer


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
    """Run one case of the sharded pipeline on a thread pool in this process.

    The budget's `slots` become pool threads, and its `threads` are what
    Polars was given at import; the product is the same CPU budget the
    distributed backends get.
    """
    from concurrent.futures import ThreadPoolExecutor

    from polars_pylance import scan_lance_fragments, write_lance_fragments

    from . import sharded

    budget = parallelism or backends.Parallelism.from_env()
    if cluster is not None:
        msg = "the threads backend is in-process; it has no cluster to join"
        raise SystemExit(msg)
    spec = queries.CASES[case]
    if spec.kind == "read":
        if dst is not None:
            msg = f"read case {case!r} writes nothing; dst must be None"
            raise SystemExit(msg)

        def map_reduce(
            reduce: ShardReducer, shards: Sequence[pl.LazyFrame]
        ) -> list[dict[str, int]]:
            with ThreadPoolExecutor(max_workers=budget.slots) as pool:
                return list(pool.map(reduce, shards))

        return sharded.reduce(
            src,
            backend="threads",
            case=case,
            n_shards=n_shards,
            map_reduce=map_reduce,
            budget=budget,
        )

    assert dst is not None  # narrowed by the read branch above
    with measured() as metrics:
        shards = [
            queries.apply_polars(shard, case)
            for shard in scan_lance_fragments(src, n_shards=n_shards)
        ]
        dataset = write_lance_fragments(
            shards,
            dst,
            mode="overwrite",
            max_workers=budget.slots,
            chunk_size=chunk_size,
        )
        rows = dataset.count_rows()

    return backends.WriteResult(
        {
            "backend": "threads",
            # The commit is inside `write_lance_fragments`, so it is not timed
            # separately here; it is the same single commit the others make.
            "commit_seconds": 0.0,
            "result_rows": rows,
            "n_fragments": len(dataset.get_fragments()),
            # Nothing is shipped: the shards are objects in this process.
            "plan_bytes": 0,
            "metadata_bytes": 0,
            "n_shards": len(shards),
            **budget.as_record(),
            **metrics,
        }
    )


if __name__ == "__main__":
    queries.smoke("threads", run)
