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

The scan goes through `PyLazyFrame.new_from_dataset_object`, the hook behind
`scan_delta`/`scan_iceberg`: Polars resolves it at IR-resolution time and hands
Lance a ready-made PyArrow predicate plus a pushed-down limit, in ~1 kB of
serialized plan. The hook is private and carries no stability guarantee — a
deliberate trade, since it is measurably the better path (see
[Why the private hook](#why-the-private-hook)). If a future Polars changes it,
pin the previous polars-lance rather than expecting a fallback.

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

## Why the private hook

`scan_lance` used to ship a second implementation on the public
`polars.io.plugins.register_io_source`, as a fallback in case
`new_from_dataset_object` ever went away. It was removed in favour of the one
path, because the public route is not merely equivalent-but-safer — it is slower
and no lighter. Best of 4 runs per case, one process each, on the 527 MB / 1 M-row
dataset from `bench/`, peak anonymous RSS:

| query | provider | io_plugin | |
| --- | --- | --- | --- |
| `head(5)` | 0.005 s / 177 MB | 0.033 s / 245 MB | **5.5× slower, +68 MB** |
| filter + sort + `head(7)` | 0.098 s / 554 MB | 0.165 s / 659 MB | **1.7× slower, +105 MB** |
| full scan + aggregate | 0.151 s / 407 MB | 0.234 s / 465 MB | **1.5× slower, +58 MB** |
| projection-only aggregate | 0.016 s / 182 MB | 0.017 s / 156 MB | par |
| filter pushdown, count | 0.039 s / 182 MB | 0.040 s / 186 MB | par |

The gap is pushdown, not overhead. The provider receives the filter already
translated to a PyArrow expression and the row limit as a number, and passes both
straight to Lance's scanner. The IO plugin receives a Polars expression it has to
re-apply per batch, and gets no limit it can hand to Lance — so `head(5)` still
pulls a whole Lance batch. Where neither matters, the two are level.

Keeping the fallback also meant keeping a hand-written Polars-expression → Lance
SQL translator (~180 lines) whose only job was recovering some of that pushdown.
Deleting the path deleted the translator with it.

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

## Polars Cloud

> **Not installable today.** polars-cloud 0.10 pins `polars==1.43.2`, below this
> package's `polars>=1.44.0` floor, so there is no `cloud` extra and the two
> cannot be resolved together. Everything in this section is written and kept
> working against the 0.10 API; it becomes usable when polars-cloud ships a
> release tracking 1.44. See [The polars pin](#the-polars-pin).

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

The scan survives `prepare_cloud_plan`, on its own and under `pl.concat()` of
`scan_lance_fragments()` shards. Since 0.9 distributes unions of Python scans, the sharded form is the
sanctioned way to fan a read across workers rather than a workaround, and 0.10's
`pl.collect_all(lazyframes, lazy=True).remote(ctx).distributed().execute()`
submits N shards as one distributed query instead of N remote ones. Note that
`collect_all(lazy=True)` requires every LazyFrame to end in a sink.

Still untested without a workspace: how the planner *executes* those nodes.
Serializing is necessary, not sufficient.

### The polars pin

polars-cloud 0.10 requires `polars==1.43.2`; this package requires
`polars>=1.44.0`. Those cannot both hold, which is why the `cloud` extra was
dropped rather than left declared — an extra pinning below the floor makes even
`uv lock` unresolvable, not just `pip install polars-lance[cloud]`.

1.44.0 is the floor because 1.43.2 is the last release carrying two bugs this
package used to work around, both verified as fixed in 1.44.0:

| on polars 1.43.2 | on polars 1.44.0 |
| --- | --- |
| `collect_batches()` as an Arrow C stream hangs on the default `provider` scan | streams normally |
| `sort().head()` pushes an unevaluable `dynamic_pred` node into an IO plugin's predicate, panicking with `internal error: entered unreachable code` | no such node is passed |

So installing polars-cloud alongside polars-lance is not a workaround: it
downgrades polars into that range and reintroduces both. `sink_lance_remote()`
also genuinely needs 0.10 for the API — on 1.42.1 the plan is rejected before it
leaves the client with *logical plan ineligible for execution on Polars Cloud:
contains callback sink*. The wait is for a polars-cloud release that tracks
1.44; `tests/test_remote.py` skips until then.

## Development

```sh
uv run --with-editable . --with pytest --with numpy --with cloudpickle python -m pytest
uv run bench/mem.py --help
```

`cloudpickle` is optional: without it the one test that serializes a callback
sink into a cloud plan skips, and the other 117 run.

## Licence

MIT
