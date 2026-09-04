"""The plan-shipping pipelines: written once, mapped by whichever scheduler.

Ray Core and Dask differ in one thing only: how a callable and a list of
arguments become results on workers. Everything around that -- sharding the
scan, counting what crosses the scheduler, and either writing fragment
files plus one commit (write cases) or collecting per-shard reductions
(read cases) -- is the same work either way, so it lives here and each
backend supplies just its own `map_shards` / `map_reduce`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from bench.dataframe import backends, queries
from bench.dataframe.metrics import measured

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import polars as pl
    import pyarrow as pa
    from lance.fragment import FragmentMetadata

    from bench.dataframe.queries import CaseName

    #: `write_shard`'s signature, as a scheduler sees it.
    ShardWriter = Callable[[pl.LazyFrame, str, pa.Schema, int], list[FragmentMetadata]]
    #: `collect_row`'s signature: one shard plan in, one small dict out.
    ShardReducer = Callable[[pl.LazyFrame], dict[str, int]]


class ShardMapper(Protocol):
    """A scheduler's `map`: run `write` on every shard, return every result."""

    def __call__(
        self,
        write: ShardWriter,
        shards: Sequence[pl.LazyFrame],
        target: str,
        arrow_schema: pa.Schema,
        chunk_size: int,
    ) -> list[list[FragmentMetadata]]:
        """Apply `write` to each shard with the three broadcast arguments.

        The arguments after `shards` are the same for every task, so a
        scheduler that distinguishes per-task from broadcast data (Dask's
        `client.map` does) can hand them over as constants.
        """
        ...


class ReduceMapper(Protocol):
    """A scheduler's `map` for reductions: one small dict back per shard."""

    def __call__(
        self,
        reduce: ShardReducer,
        shards: Sequence[pl.LazyFrame],
    ) -> list[dict[str, int]]:
        """Apply `reduce` to each shard plan and return every small dict."""
        ...


def write_shard(
    shard: pl.LazyFrame,
    target: str,
    arrow_schema: pa.Schema,
    chunk_size: int,
) -> list[FragmentMetadata]:
    """Write one shard's fragment files on a worker. Return the metadata.

    Both schedulers ship this same function, by reference: the worker needs
    polars-pylance installed but receives no code. The import is inside for
    the same reason a driver-side import would be wrong -- building the plan
    must not require Lance on the machine that only schedules.
    """
    from polars_pylance import write_lance_shard

    return write_lance_shard(
        shard,
        target,
        mode="overwrite",
        chunk_size=chunk_size,
        arrow_schema=arrow_schema,
    )


def run(
    src: str,
    dst: str,
    *,
    backend: str,
    case: CaseName,
    n_shards: int,
    chunk_size: int,
    map_shards: ShardMapper,
    budget: backends.Parallelism,
) -> backends.WriteResult:
    """Run one write case from `src` to `dst` through `map_shards`.

    The measured region is the query and nothing else: the caller has
    already built and warmed its cluster, so what is timed here is planning,
    the fan-out, and the commit. Reports that time, its commit share, the
    row count and fragment count, and how many bytes crossed the scheduler
    in each direction -- the number that says whether the data stayed off it.
    """
    from polars_pylance import commit_lance_fragments, scan_lance_fragments

    with measured() as metrics:
        shards = [
            queries.apply_polars(s, case)
            for s in scan_lance_fragments(src, n_shards=n_shards)
        ]
        schema = shards[0].collect_schema().to_arrow()
        parts = map_shards(write_shard, shards, dst, schema, chunk_size)
        fragments = [f for part in parts for f in part]

        with measured() as commit:
            dataset = commit_lance_fragments(
                dst, fragments, schema=schema, mode="overwrite"
            )

    return backends.WriteResult(
        {
            "backend": backend,
            "commit_seconds": commit["seconds"],
            "result_rows": dataset.count_rows(),
            "n_fragments": len(dataset.get_fragments()),
            "plan_bytes": queries.plan_bytes(shards),
            "metadata_bytes": queries.metadata_bytes(fragments),
            "n_shards": len(shards),
            **budget.as_record(),
            **metrics,
        }
    )


def collect_row(shard: pl.LazyFrame) -> dict[str, int]:
    """Collect one shard's one-row reduction. Ships by reference, like `write_shard`."""
    row = shard.collect(engine="streaming").to_dicts()[0]
    # A shard with no surviving rows sums to None in some engines; the
    # reduction's zero is 0, not null.
    return {key: value or 0 for key, value in row.items()}


def combine(counts: list[dict[str, int]]) -> dict[str, int]:
    """Sum per-shard reductions. Exact: counts and id sums add in any order."""
    total: dict[str, int] = {}
    for row in counts:
        for key, value in row.items():
            total[key] = total.get(key, 0) + value
    return total


def reduce(
    src: str,
    *,
    backend: str,
    case: CaseName,
    n_shards: int,
    map_reduce: ReduceMapper,
    budget: backends.Parallelism,
) -> backends.ReadResult:
    """Run one read case on `src`: per-shard reductions, summed on the driver.

    No dataset is written and nothing is committed, so only small dicts
    cross the scheduler -- this is the distributed read/compute shape with
    the output IO taken out.
    """
    from polars_pylance import scan_lance_fragments

    with measured() as metrics:
        shards = [
            queries.apply_polars(s, case)
            for s in scan_lance_fragments(src, n_shards=n_shards)
        ]
        totals = combine(map_reduce(collect_row, shards))

    return backends.ReadResult(
        {
            "backend": backend,
            "commit_seconds": 0.0,
            "result_count": totals.get("n", 0),
            "result_sum_id": totals.get("s_id", 0),
            "plan_bytes": queries.plan_bytes(shards),
            "metadata_bytes": 0,
            "n_shards": len(shards),
            **budget.as_record(),
            **metrics,
        }
    )
