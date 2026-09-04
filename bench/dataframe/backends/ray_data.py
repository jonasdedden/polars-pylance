"""The same pipeline through Ray Data and lance-ray, for contrast.

Reads with `read_lance` (projection and the SQL filter pushed into the
scanner), transforms Arrow batches, and writes with `write_lance`. No
polars-pylance code runs here: this is the native Ray Data way to do the
same jobs, measured so the plan-shipping backends have something to stand
next to -- or aggregates natively for the read case. It ships no plan
and stages no fragment metadata, so those two counters are zero rather
than small.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, overload

from bench.dataframe import backends, queries
from bench.dataframe.metrics import measured

from . import ray_core

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
    """Run one case as a Ray Data pipeline.

    `n_shards` and `chunk_size` mean here what they mean for the other
    backends: how many blocks the scan is cut into, and how many rows a
    transform sees at once. Ray Data streams read into transform into write
    rather than running them in phases, so its slots are shared between the
    three stages instead of held by one at a time.
    """
    import lance
    from lance_ray import read_lance, write_lance
    from ray.data.aggregate import Count, Sum

    budget = parallelism or backends.Parallelism.from_env()
    task_cpus: dict[str, int] = {"num_cpus": budget.threads}
    spec = queries.CASES[case]
    ray_core.connect(budget, cluster)
    ray_core.warm(budget)
    if spec.kind == "read":
        if dst is not None:
            msg = f"read case {case!r} writes nothing; dst must be None"
            raise SystemExit(msg)
        with measured() as metrics:
            totals = read_lance(
                src,
                columns=list(spec.columns),
                filter=spec.lance_filter or None,
                override_num_blocks=n_shards,
                ray_remote_args=task_cpus,
            ).aggregate(Sum("id"), Count())
        return backends.ReadResult(
            {
                "backend": "ray-data",
                "commit_seconds": 0.0,
                # Empty inputs sum to None; the reduction's zero is 0.
                "result_count": int(totals["count()"] or 0),
                "result_sum_id": int(totals["sum(id)"] or 0),
                "plan_bytes": 0,
                "metadata_bytes": 0,
                "n_shards": n_shards,
                **budget.as_record(),
                **metrics,
            }
        )

    assert dst is not None  # read cases return above
    # Defined here so the `case` closes over the run's argument; still a
    # partial over the top-level `apply_arrow`, which is what stays picklable.
    arrow_fn: Any = partial(queries.apply_arrow, case=case)
    with measured() as metrics:
        ds = read_lance(
            src,
            columns=list(spec.columns),
            filter=spec.lance_filter or None,
            override_num_blocks=n_shards,
            ray_remote_args=task_cpus,
        )
        transformed = ds.map_batches(
            # `batch_format="pyarrow"` guarantees a `pa.Table`; the stub's
            # `DataBatch` union is wider. `Any` carries the partial without a
            # cast while keeping the top-level function picklable.
            arrow_fn,
            batch_format="pyarrow",
            batch_size=chunk_size,
            concurrency=budget.slots,
            num_cpus=budget.threads,
        )
        write_lance(transformed, dst, mode="create", ray_remote_args=task_cpus)
    written = lance.dataset(dst)
    return backends.WriteResult(
        {
            "backend": "ray-data",
            # Ray Data commits inside `write_lance`, so there is no separate
            # commit to time and nothing crosses the scheduler to make it happen.
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
    queries.smoke("ray-data", run)
