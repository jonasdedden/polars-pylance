"""polars-pylance on Dask Distributed: the same pipeline, mapped by Dask."""

# mypy: disallow-untyped-calls=False
# `distributed` ships a `py.typed` marker but leaves its public API
# unannotated, so every `Client`/`LocalCluster` call is untyped. The bodies
# below are still checked; only the call sites opt out.

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, overload

from bench.dataframe import backends, queries

from . import sharded

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import polars as pl
    import pyarrow as pa
    from distributed import Client
    from lance.fragment import FragmentMetadata

    from bench.dataframe.backends import Parallelism
    from bench.dataframe.queries import CaseName, ReadCaseName, WriteCaseName

    from .sharded import ShardReducer, ShardWriter


@contextlib.contextmanager
def _client(budget: Parallelism, cluster: str | None) -> Generator[Client, None, None]:
    """A client on `cluster`, or on a local one built to the budget.

    One process per slot with a single task thread each, rather than fewer
    processes running several tasks apiece: two tasks in one process would
    share that process's Polars pool, and the budget would stop meaning what
    it says.
    """
    from distributed import Client, LocalCluster

    if cluster is not None:
        with Client(cluster) as client:
            yield client
        return

    # `env` reaches the Nanny, which sets it before the worker imports
    # Polars -- the only point at which the pool size can still be chosen.
    # `dashboard_address` is inferred as `str` from its `":8787"` default,
    # but the docs allow `None` to disable the dashboard, which is what a
    # benchmark wants. Typed as `Any` to carry that `None` without a cast.
    no_dashboard: Any = None
    local = LocalCluster(
        n_workers=budget.slots,
        threads_per_worker=1,
        processes=True,
        dashboard_address=no_dashboard,
        env=budget.worker_env,
    )
    with local, Client(local) as client:
        yield client


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
    """Run one case of the sharded pipeline as one Dask task per shard."""
    budget = parallelism or backends.Parallelism.from_env()

    with _client(budget, cluster) as client:
        # Start every worker and import its dependencies before measuring:
        # a worker importing Polars inside the query is startup, not query.
        client.wait_for_workers(budget.slots)
        client.run(queries.import_dependencies)
        spec = queries.CASES[case]
        if spec.kind == "read":
            if dst is not None:
                msg = f"read case {case!r} writes nothing; dst must be None"
                raise SystemExit(msg)

            def map_reduce(
                reduce: ShardReducer, shards: Sequence[pl.LazyFrame]
            ) -> list[dict[str, int]]:
                # `gather` is untyped, so its stub resolves to an async
                # overload union; route through `Any` (no cast) to the
                # declared return.
                gathered: Any = client.gather(client.map(reduce, shards, pure=False))
                out: list[dict[str, int]] = gathered
                return out

            return sharded.reduce(
                src,
                backend="dask",
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
            # `pure=False`: two shards can be byte-identical queries over
            # different fragments, and Dask would otherwise dedupe them into
            # one task by hashing the arguments.
            futures = client.map(
                write,
                shards,
                target=target,
                arrow_schema=arrow_schema,
                chunk_size=chunk_size,
                pure=False,
            )
            gathered: Any = client.gather(futures)
            out: list[list[FragmentMetadata]] = gathered
            return out

        assert dst is not None  # narrowed by the read branch above
        return sharded.run(
            src,
            dst,
            backend="dask",
            case=case,
            n_shards=n_shards,
            chunk_size=chunk_size,
            map_shards=map_shards,
            budget=budget,
        )


if __name__ == "__main__":
    queries.smoke("dask", run)
