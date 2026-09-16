"""End-to-end tests for this package on a Polars cluster in local Kubernetes.

What they are for: `docs/POLARS_CLOUD.md` could say only that a `scan_lance`
plan serializes and survives `prepare_cloud_plan`, and that how a planner
*executes* those nodes was untested without a workspace. These run it. A real
scheduler, two workers, and an object store all three can reach, so a read
executes on the workers and a write is fragments the workers produced and the
client committed.

Skipped unless ``POLARS_K8S_E2E=1``, so a plain ``pytest`` run is unaffected.
The repository-local `.github/actions/polars-onprem-cluster` action deploys the
cluster this expects, including the `pylance` and `polars-pylance` the workers
need to run a scan node at all, and its outputs arrive here as the environment
variables read below; its README covers the rest.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import lance
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_pylance as pll
from polars_pylance.cloud import sink_lance_remote

if TYPE_CHECKING:
    from typing import Literal

    import polars_cloud as pc

pytestmark = pytest.mark.skipif(
    os.environ.get("POLARS_K8S_E2E") != "1",
    reason="set POLARS_K8S_E2E=1 to run against the local k8s Polars cluster",
)

SCHEDULER_URI = os.environ.get("POLARS_SCHEDULER_URI", "http://localhost:5051")
OBSERVATORY_URI = os.environ.get("POLARS_OBSERVATORY_URI", "http://localhost:3001")
BUCKET = os.environ.get("S3_BUCKET", "polars")

# One endpoint for both sides: the workers reach it through the in-cluster
# Service, and the client through the NodePort plus an `/etc/hosts` entry, so
# one dict configures the Lance scan on the workers, the Lance write they
# perform, the staging channel that carries the fragment metadata back, and the
# client's own reads. See the action's `manifests/object-store.yaml`.
STORAGE_OPTIONS: dict[str, str] = {
    "aws_endpoint_url": os.environ.get("S3_ENDPOINT", "http://objectstore:8333"),
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "ci-access-key"),
    # A throwaway credential for a throwaway store, overridden from the
    # environment where the job sets one.
    "aws_secret_access_key": os.environ.get(
        "AWS_SECRET_ACCESS_KEY", "ci-secret-key-not-sensitive"
    ),
    "aws_region": "us-east-1",
    "aws_allow_http": "true",
}

ROWS = 1_000_000
# Eight fragments, so a sharded scan has something to deal out and a worker
# that takes a shard still has more than one file to read.
ROWS_PER_FRAGMENT = ROWS // 8
KEYS = 97
# One key is ~1/97th of the dataset: enough rows to be a real scan, few enough
# that the round trip is quick.
PROBE_KEY = 7


@pytest.fixture(scope="session")
def ctx() -> pc.ClusterContext:
    """The cluster, addressed directly rather than through the control plane."""
    # Imported here rather than at module scope so a plain `pytest` run can
    # collect (and skip) this file without the `cloud` extra installed.
    import polars_cloud as pc

    pc.Config.set_user_name("github-actions")
    return pc.ClusterContext(
        uri=SCHEDULER_URI,
        observatory=pc.ClientOptions(uri=OBSERVATORY_URI),
    )


@pytest.fixture(scope="session")
def run_prefix() -> str:
    """A fresh prefix per run, so a re-run never reads the last one's output."""
    return f"s3://{BUCKET}/e2e/{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def source(run_prefix: str) -> tuple[str, pl.DataFrame]:
    """A multi-fragment Lance dataset in the object store, and its contents.

    Written by `sink_lance` from the client, so a worker can only read it if its
    own Lance install and object-store settings reach the same place -- which is
    half of what these tests are checking.
    """
    uri = f"{run_prefix}/source.lance"
    frame = pl.select(id=pl.int_range(0, ROWS, dtype=pl.Int64)).with_columns(
        key=(pl.col("id") % KEYS).cast(pl.Int32),
        value=(pl.col("id") * 7 % 1_000).cast(pl.Float64),
    )
    pll.sink_lance(
        frame.lazy(),
        uri,
        mode="overwrite",
        storage_options=STORAGE_OPTIONS,
        max_rows_per_file=ROWS_PER_FRAGMENT,
    )
    return uri, frame


def scan(uri: str) -> pl.LazyFrame:
    """A `scan_lance` carrying the credentials the workers will need."""
    return pll.scan_lance(uri, storage_options=STORAGE_OPTIONS)


def _query_result(
    query: pc.QueryResult | pc.DirectQuery | pc.ProxyQuery,
) -> pc.QueryResult:
    """The result itself, whether `execute` blocked for it or handed back a handle."""
    import polars_cloud as pc

    return query if isinstance(query, pc.QueryResult) else query.await_result()


def _submit(
    lf: pl.LazyFrame,
    ctx: pc.ClusterContext,
    mode: Literal["single_node", "distributed"],
) -> pc.ExecuteRemote:
    """`lf` handed to the cluster under one of the two scaling modes."""
    remote = lf.remote(ctx)
    return remote.single_node() if mode == "single_node" else remote.distributed()


def _workers_used(result: pc.QueryResult) -> int:
    """The widest stage of the query, in workers."""
    stages = result.stage_statistics()
    # `stage_statistics()` returns one stage when asked for a number and the
    # whole list when not; this call is the second form.
    assert isinstance(stages, list)
    return max((stage.num_workers_used for stage in stages), default=0)


def test_cluster_smoke(ctx: pc.ClusterContext) -> None:
    """A canary, so a cluster that is simply broken does not read as a Lance bug."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 4, 5]}).with_columns(
        c=pl.col("a").max().over("b")
    )
    result = lf.remote(ctx).single_node().collect()
    assert_frame_equal(result, lf.collect(), check_row_order=False)


@pytest.mark.parametrize("mode", ["single_node", "distributed"])
def test_remote_scan_lance(
    ctx: pc.ClusterContext,
    source: tuple[str, pl.DataFrame],
    mode: Literal["single_node", "distributed"],
) -> None:
    """A `scan_lance` node runs on a worker and returns what it does locally.

    The scan is a Python IO source: the plan carries a URI and the closure that
    opens it, and the worker has to import `polars_pylance` to run it at all.

    Both planners, because they place that node differently and only one of them
    is covered by the sharded test below. The query is one key's worth of rows,
    so the single-node case is a quick first sign of life rather than something
    to wait on, and running it twice costs little.
    """
    uri, expected = source
    lf = scan(uri).filter(pl.col("key") == PROBE_KEY).select("id", "value")

    got = _submit(lf, ctx, mode).collect()

    assert_frame_equal(
        got.sort("id"),
        expected.filter(pl.col("key") == PROBE_KEY).select("id", "value"),
    )


def test_sharded_scan_fans_out(
    ctx: pc.ClusterContext, source: tuple[str, pl.DataFrame]
) -> None:
    """`scan_lance_fragments` under `pl.concat` really is spread across workers.

    The documented way to fan a Lance read out: polars-cloud distributes a union
    of Python scans, so each shard is a node the planner can place on its own
    worker. One worker doing all of it would pass the value check, so the count
    is asserted too.
    """
    uri, expected = source
    shards = pll.scan_lance_fragments(uri, n_shards=4, storage_options=STORAGE_OPTIONS)
    assert len(shards) == 4

    query = (
        pl.concat(shards).group_by("key").agg(n=pl.len(), total=pl.col("value").sum())
    )
    result = _query_result(query.remote(ctx).distributed().execute())

    workers = _workers_used(result)
    assert workers >= 2, f"expected a multi-worker plan, got {workers} worker(s)"

    want = (
        expected.lazy()
        .group_by("key")
        .agg(n=pl.len(), total=pl.col("value").sum())
        .collect()
    )
    assert_frame_equal(result.lazy().collect().sort("key"), want.sort("key"))


def test_sink_lance_remote(
    ctx: pc.ClusterContext, source: tuple[str, pl.DataFrame], run_prefix: str
) -> None:
    """The workers write Lance data files; one client-side commit publishes them.

    This is the whole arrangement in `_remote.py` doing its job for real: a
    cloudpickled fragment writer runs on the workers, stages its fragment
    metadata next to the dataset because a callback has no return path, and the
    client turns the staged metadata into one version.
    """
    uri, expected = source
    out = f"{run_prefix}/filtered.lance"
    lf = scan(uri).filter(pl.col("key") < 25).select("id", "key", "value")

    dataset = sink_lance_remote(
        lf.remote(ctx).distributed(),
        out,
        mode="create",
        storage_options=STORAGE_OPTIONS,
        chunk_size=100_000,
    )

    assert isinstance(dataset, lance.LanceDataset)
    # Several fragments, because several batches were written independently.
    # One would mean the write collapsed to a single callback invocation.
    assert len(dataset.get_fragments()) > 1

    want = expected.filter(pl.col("key") < 25).select("id", "key", "value")
    got = scan(out).collect(engine="streaming")
    assert_frame_equal(got.sort("id"), want)
