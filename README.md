# polars-lance

Lazy, streaming [Lance](https://lance.org) ↔ [Polars](https://pola.rs) integration.

`scan_lance()` returns a real `LazyFrame`: the Polars optimizer pushes column
projections, filters and row limits down into Lance, and batches are pulled only
as the streaming engine consumes them. `sink_lance()` writes a query into Lance
batch by batch. Neither direction holds the dataset — or a whole fragment — in
memory.

```python
import polars as pl
import polars_lance as pll

lf = pll.scan_lance("s3://bucket/embeddings.lance")

pll.sink_lance(
    lf.filter(pl.col("score") > 0.9).select("id", "text", "vector"),
    "filtered.lance",
    mode="overwrite",
)
```

## Why this exists

Polars has no native Lance reader or writer
([pola-rs/polars#14452](https://github.com/pola-rs/polars/issues/14452) has been
open since 2024). Lance datasets do implement the PyArrow dataset protocol, so
`pl.scan_pyarrow_dataset` works, but it cannot push down row limits, cannot pin a
dataset version, cannot reach vector or full-text search, and leaves Lance's
read-ahead defaults — `io_buffer_size` alone defaults to 2 GiB — untouched.

## Reading

```python
lf = pll.scan_lance(
    "data.lance",
    version=7,  # or a tag; omit to follow the latest
    options=pll.LanceScanOptions(),  # readahead / buffer tuning
    nearest={"column": "vector", "q": query, "k": 10},  # ANN search
    with_row_id=True,
)
```

What gets pushed into Lance: the column projection, the filter (as a PyArrow
expression, so Lance's scalar indices and page statistics apply), and — when no
filter follows the scan — the row limit. `.head()` stops the scan early rather
than reading to the end.

Two implementations sit behind `impl=`:

| `impl` | Polars hook | Notes |
| --- | --- | --- |
| `"provider"` (default) | `PyLazyFrame.new_from_dataset_object` | The hook behind `scan_delta`/`scan_iceberg`. Best pushdown, ~1 kB serialized plans. Private, unstable. |
| `"io_plugin"` | `polars.io.plugins.register_io_source` | Public API. The predicate arrives as a Polars expression, so it is applied per batch and additionally translated to Lance SQL where possible. |

`scan_lance_fragments()` returns one `LazyFrame` per fragment (or per shard) when
you want to fan a read out over threads, processes or workers yourself.

## Writing

```python
pll.sink_lance(lf, "out.lance", mode="append", max_rows_per_file=1_000_000)
pll.sink_lance(updates, "out.lance", mode="merge", on="id")  # upsert
plan = pll.sink_lance(lf, "out.lance", lazy=True)  # write on collect
```

For distributed writes, each shard writes its own fragments and a single commit
makes them one version:

```python
shards = pll.scan_lance_fragments("in.lance")
pll.write_lance_fragments([s.filter(pl.col("ok")) for s in shards], "out.lance")
```

`commit_lance_fragments()` is that second half on its own — publish fragments
someone else wrote. It is what the [Polars Cloud](#polars-cloud) path commits
with once the workers are done.

## Memory behaviour

From `uv run bench/mem.py all` on a 527 MB Lance dataset (1 M rows, incompressible
`fixed_size_binary(512)` payload column, 5 fragments). Peak anonymous RSS, one
case per process, against an ~87 MB interpreter-and-libraries baseline:

| | peak anon RSS |
| --- | --- |
| `scan_lance().select(...)` — payload column never read | **87 MB** |
| Fragment-parallel write (`write_lance_fragments`) | 162 MB |
| Pulling every batch through Polars and discarding | 284 MB |
| Streaming aggregation over the payload column | 398 MB |
| The same, with `LanceScanOptions.throughput()` | 530 MB |
| A plain Lance `to_batches` loop, no Polars involved | 606 MB |
| `sink_lance` (scan → transform → write) | 646 MB |
| The same streaming query on the **in-memory** engine | 845 MB |
| `lance.dataset().to_table()` → `pl.from_arrow` | 1 035 MB |

Three things to take from that. Projection pushdown is the biggest lever by far —
not reading a column costs nothing. Collect with `engine="streaming"`; the
in-memory engine materialises the result and gives up the advantage. And the
default `LanceScanOptions` beat a naive Lance loop, because Lance's own
`io_buffer_size` default is 2 GiB.

Tripling the source to 1.58 GB moves peak RSS by 1.19× — nowhere near the 3× a
buffering pipeline would show. `uv run bench/mem.py guard` asserts this.

## Upstream quirks worked around

Two things that bite anyone building this integration by hand, both found by
running the pipeline rather than reading the docs:

- **`collect_batches()` as an Arrow C stream deadlocks** (polars 1.43.0+, a
  regression from 1.42.x). Handing `__arrow_c_stream__` to
  `pa.RecordBatchReader.from_stream` is the obvious zero-copy route into Lance,
  but it hangs whenever the plan contains a `PythonDataset` scan node — what
  `scan_lance` produces by default. Nothing about Lance is involved: plain
  `read_all()` hangs identically, and so does `pl.scan_delta`. `sink_lance` pulls
  through a Python generator instead.
- **Polars hands IO plugins a predicate it cannot evaluate** (polars 1.39.0+, a
  regression from 1.38.1). A `sort(...).head(n)` makes the engine inject an opaque
  `dynamic_pred: <uuid>` node into the pushed-down predicate — with or without a
  `filter` in the query. Every route into the expression engine panics with
  `internal error: entered unreachable code`. It is a top-k pruning hint, so
  `_predicate.prune_unevaluable` strips it and lets the top-k operator apply the
  real limit downstream.

Reports and standalone reproducers for both are in [`upstream/`](upstream/).

## Polars Cloud

Reads are designed to ship: a scan serializes to a few kB and carries a URI,
never an open dataset handle. Workers need `pylance` and `polars-lance`
installed, via `ComputeContext(requirements=...)` —
`polars_lance.cloud.requirements_txt()` renders the pinned lines, since Polars
Cloud rejects a context whose polars version differs from the client's.

### Writing from a remote query

polars-cloud 0.10 added `sink_batches()`, which hands each result batch to a
Python callable. That callable is **cloudpickled into the serialized query
plan**, so it runs on the workers — Lance no longer has to be a sink format
Polars Cloud knows about, and the write is genuinely distributed rather than
streamed back through the client.

```python
from polars_lance.cloud import requirements_txt, sink_lance_remote

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
different workers", and appending a fragment is not idempotent — so each staging
object is named after a deterministic digest of its batch, and a replayed batch
overwrites its own metadata instead of adding a second copy. `tests/test_remote.py`
delivers every batch twice and asserts the row count is unchanged. Pass
`fragment_key=` if your query has a natural key, such as a partition column;
the default digest would collapse two *distinct* batches that are byte-identical.
Replays do leave their earlier data files unreferenced — reclaim them with
`dataset.cleanup_old_versions(..., delete_unverified=True)`.

Drive the query yourself with `stage_lance_sink()` when you want to pick a
planner or inspect the query handle:

```python
staged = pll.cloud.stage_lance_sink("s3://bucket/out.lance", lf, mode="overwrite")
query = lf.remote(ctx).distributed(planner="miso").sink_batches(staged.callback)
query.await_result()
staged.commit()
```

The Parquet-staging route remains as the conservative fallback: sink to Parquet
and convert with `polars_lance.cloud.convert_parquet_to_lance()`. 0.10's
`DirectQuery.delete_result()` makes cleaning up the intermediate a single call,
in direct mode with anonymous storage configured for `allow_delete`.

### Reading

Both scan implementations survive `prepare_cloud_plan` — `provider` and
`io_plugin`, on their own and under `pl.concat()` of `scan_lance_fragments()`
shards. Since 0.9 distributes unions of Python scans, the sharded form is the
sanctioned way to fan a read across workers rather than a workaround, and 0.10's
`pl.collect_all(lazyframes, lazy=True).remote(ctx).distributed().execute()`
submits N shards as one distributed query instead of N remote ones. Note that
`collect_all(lazy=True)` requires every LazyFrame to end in a sink.

Still untested without a workspace: how the planner *executes* those nodes.
Serializing is necessary, not sufficient.

### The polars pin

polars-cloud 0.10 requires `polars==1.43.2`, up from 1.42.1 in 0.9. That crosses
into the range where `collect_batches()` deadlocks as an Arrow C stream, and the
reproducer still hangs on 1.43.2 for the default `provider` scan — so the
workaround below stays. The rest of the suite passes unchanged on 1.43.2.

The bump is not incidental to `sink_lance_remote()`: on 1.42.1 the plan is
rejected before it leaves the client with *logical plan ineligible for execution
on Polars Cloud: contains callback sink*. The remote write needs 0.10 for the
polars it pins as much as for the API it adds. Everything else in the package
still works on the declared `polars>=1.42.1` floor — the test that ships a
callback sink skips there.

## Development

```sh
uv run --with-editable . --with pytest --with numpy python -m pytest
uv run bench/mem.py --help
```

## Licence

MIT
