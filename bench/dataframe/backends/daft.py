"""Daft on its native runner: one process, multi-threaded, no scheduler.

The control for the distributed backends. Daft plans the same filter,
projection and computed column and executes them across threads in this
process, so nothing is shipped anywhere and there is no cluster to pay for.
If a scheduler cannot beat this, the job did not need one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from bench.dataframe import backends, queries
from bench.dataframe.metrics import measured

if TYPE_CHECKING:
    from bench.dataframe.backends import Parallelism
    from bench.dataframe.queries import CaseName, ReadCaseName, WriteCaseName


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
    """Run one case on Daft's native multi-threaded runner.

    Takes the same CPU budget as the distributed backends, as one pool of
    `slots * threads` threads rather than that many separate workers: Daft
    schedules its own morsels across them, so the split into slots and
    threads has no meaning here and only the product does.
    """
    import lance

    budget = parallelism or backends.Parallelism.from_env()
    if cluster is not None:
        msg = "the native Daft runner is local; use the daft-ray backend for a cluster"
        raise SystemExit(msg)
    spec = queries.CASES[case]

    queries.configure_daft("native", budget.cpus, chunk_size)

    if spec.kind == "read":
        if dst is not None:
            msg = f"read case {case!r} writes nothing; dst must be None"
            raise SystemExit(msg)
        with measured() as metrics:
            totals = queries.daft_totals(queries.daft_query_frame(src, n_shards, case))
        return backends.ReadResult(
            {
                "backend": "daft",
                "commit_seconds": 0.0,
                "result_count": totals["n"],
                "result_sum_id": totals["s_id"],
                "plan_bytes": 0,
                "metadata_bytes": 0,
                "n_shards": n_shards,
                **budget.as_record(),
                **metrics,
            }
        )

    assert dst is not None  # read cases return above
    with measured() as metrics:
        queries.daft_query_frame(src, n_shards, case).write_lance(dst, mode="create")

    written = lance.dataset(dst)
    return backends.WriteResult(
        {
            "backend": "daft",
            # There is no fan-out and no second phase: Daft's write commits the
            # dataset itself, so there is no separate commit to time.
            "commit_seconds": 0.0,
            "result_rows": written.count_rows(),
            "n_fragments": len(written.get_fragments()),
            "plan_bytes": 0,
            "metadata_bytes": 0,
            "n_shards": n_shards,
            **budget.as_record(),
            **metrics,
        }
    )


if __name__ == "__main__":
    queries.smoke("daft", run)
