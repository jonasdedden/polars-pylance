# Polars Cloud

> **Not installable today.** polars-cloud 0.10 pins `polars==1.43.2`, below this
> package's `polars>=1.44.1` floor, so there is no `cloud` extra and the two
> cannot be resolved together. Everything here is written and kept working
> against the 0.10 API; it becomes usable when polars-cloud ships a release
> tracking 1.44.

Reads are designed to ship: a scan serializes to a few kB and carries a URI,
never an open dataset handle. Workers need `pylance` and `polars-pylance`
installed, via `ComputeContext(requirements=...)`.
`polars_pylance.cloud.requirements_txt()` renders the pinned lines, since Polars
Cloud rejects a context whose polars version differs from the client's.

## Writing from a remote query

polars-cloud 0.10 added `sink_batches()`, which hands each result batch to a
Python callable. That callable is **cloudpickled into the serialized query
plan**, so it runs on the workers. Lance no longer has to be a sink format
Polars Cloud knows about, and the write is genuinely distributed rather than
streamed back through the client.

```python
from polars_pylance.cloud import requirements_txt, sink_lance_remote

ctx = pc.ComputeContext(cpus=8, memory=32, requirements=requirements_txt().encode())
lf = pll.scan_lance("s3://bucket/in.lance").filter(pl.col("score") > 0.9)

sink_lance_remote(
    lf.remote(ctx).distributed(),
    "s3://bucket/out.lance",
    mode="overwrite",
    chunk_size=100_000,  # the remote counterpart of max_rows_per_file
)
```

The shape is Lance's write/commit split. Each worker writes **data files only**
and commits nothing, so no worker can publish a partial dataset; it then stages
the resulting fragment metadata as JSON next to the dataset, because a callback
shipped to a worker has no return path. When the query finishes, the client
lists the staging prefix and makes every fragment one version with a single
commit.

polars-cloud documents that the callback "might be called multiple times from
different workers", and appending a fragment is not idempotent. So each staging
object is named after a deterministic digest of its batch, and a replayed batch
overwrites its own metadata instead of adding a second copy. `tests/test_remote.py`
delivers every batch twice and asserts the row count is unchanged. Pass
`fragment_key=` if your query has a natural key, such as a partition column;
the default digest would collapse two *distinct* batches that are byte-identical.
Replays do leave their earlier data files unreferenced. Reclaim them with
`dataset.cleanup_old_versions(..., delete_unverified=True)`.

Drive the query yourself with `stage_lance_sink()` when you want to pick a
planner or inspect the query handle:

```python
staged = pll.cloud.stage_lance_sink("s3://bucket/out.lance", lf, mode="overwrite")
query = lf.remote(ctx).distributed(planner="miso").sink_batches(staged.callback)
query.await_result()
staged.commit()
```

## Reading

The scan survives `prepare_cloud_plan`, on its own and under `pl.concat()` of
`scan_lance_fragments()` shards. Since 0.9 distributes unions of Python scans,
the sharded form is the sanctioned way to fan a read across workers rather than
a workaround, and 0.10's
`pl.collect_all(lazyframes, lazy=True).remote(ctx).distributed().execute()`
submits N shards as one distributed query instead of N remote ones. Note that
`collect_all(lazy=True)` requires every LazyFrame to end in a sink.

Still untested without a workspace: how the planner *executes* those nodes.
Serializing is necessary, not sufficient.
