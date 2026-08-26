# polars-pylance

Lazy, streaming [Lance](https://lance.org) ↔ [Polars](https://pola.rs) integration.

`scan_lance()` returns a real `LazyFrame`: the Polars optimizer pushes column
projections, filters and row limits down into Lance, and batches are pulled only
as the streaming engine consumes them. `sink_lance()` writes a query into Lance
batch by batch. Neither direction holds the dataset -- or a whole fragment -- in
memory.

```python
import polars as pl
import polars_pylance as pll

lf = pll.scan_lance("s3://bucket/embeddings.lance")

pll.sink_lance(
    lf.filter(pl.col("score") > 0.9).select("id", "text", "vector"),
    "filtered.lance",
    mode="overwrite",
)
```

## The name

`polars` + `pylance`, composed in Python. There is no compiled extension here and
no `lance` crate linked in: the package is pure Python over the two libraries it
joins, which is why it installs as a single wheel, tracks new `pylance` releases
without a rebuild, and reaches Lance features as fast as `pylance` exposes them.

## Comparison with `polars-lance`

Note that PyPI's `polars-lance` is an unrelated package by a different author
(extensive [benchmarks and feature insights](https://github.com/jonasdedden/polars-pylance/blob/main/COMPARISON.md) for comparison).
The two can coexist in one environment but follow different implementation designs.

## Why this exists

Polars has no native Lance reader or writer
([pola-rs/polars#14452](https://github.com/pola-rs/polars/issues/14452) has been
open since 2024). Lance datasets do implement the PyArrow dataset protocol, so
`pl.scan_pyarrow_dataset` works, but it cannot push down row limits, cannot pin a
dataset version, cannot reach vector or full-text search, and leaves Lance's
read-ahead defaults -- `io_buffer_size` alone defaults to 2 GiB -- untouched.

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
expression, so Lance's scalar indices and page statistics apply), and -- when no
filter follows the scan -- the row limit. `.head()` stops the scan early rather
than reading to the end.

The scan goes through `PyLazyFrame.new_from_dataset_object`, the hook behind
`scan_delta`/`scan_iceberg`: Polars resolves it at IR-resolution time and hands
Lance a ready-made PyArrow predicate plus a pushed-down limit, in ~2 kB of
serialized plan.

### When the filter is the expensive part

Polars lowers only part of its expression language to PyArrow: comparisons,
boolean structure and null checks. `is_in`, string matching, arithmetic,
temporal parts, list and struct access are handed to the engine instead, so
Lance reads rows the filter would have skipped.

```python
lf = pll.scan_lance("data.lance", impl="io_plugin")
lf.filter(pl.col("text").str.contains("needle")).collect(engine="streaming")
```

`impl="io_plugin"` scans through Polars' IO-plugin hook, which hands over the
whole predicate; polars-pylance lowers it into a Lance SQL filter instead. On a
4M-row dataset that is 4.6x faster for the query above, 74x with an NGRAM index
on the column -- a scalar index cannot help a predicate that never reaches
Lance. It is level or slightly slower on everything else, which is why it is not
the default. [`docs/PREDICATE_PUSHDOWN.md`](https://github.com/jonasdedden/polars-pylance/blob/main/docs/PREDICATE_PUSHDOWN.md)
has the coverage table and the measurements.

Three Polars branches ([#28994](https://github.com/pola-rs/polars/pull/28994),
[#28995](https://github.com/pola-rs/polars/pull/28995),
[#28996](https://github.com/pola-rs/polars/pull/28996)) close the gap from the
other side, and the default path picks them up with no change here: `is_in`,
arithmetic and `eq_missing` start reaching Lance through Polars' own lowering,
and the whole predicate reaches the provider hook for this package to lower
itself. Measured at 2x to 16x, and up to 74x with scalar indices, in
[`docs/PATCHED_POLARS_PUSHDOWN.md`](https://github.com/jonasdedden/polars-pylance/blob/main/docs/PATCHED_POLARS_PUSHDOWN.md).

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

`commit_lance_fragments()` is that second half on its own -- publish fragments
someone else wrote. It is what the [Polars Cloud](#polars-cloud) path commits
with once the workers are done.

## Memory behaviour

Peak RSS for `polars-pylance` across the size ladder in
[`bench/`](https://github.com/jonasdedden/polars-pylance/blob/main/bench/README.md), on datasets from 1 GiB to 190.7 GiB -- a **190×**
increase in data:

| | 1 GiB source | 190.7 GiB source |
| --- | --- | --- |
| Projection-only scan -- payload column never read | 0.20 GiB | 0.28 GiB |
| Sharded fragment scan (`scan_lance_fragments`) | 0.20 GiB | 0.28 GiB |
| Full scan + aggregate over the payload column | 0.46 GiB | 0.59 GiB |
| 50% filter + payload aggregate | 0.93 GiB | 2.24 GiB |
| `sink_lance` (scan → transform → write) | 1.11 GiB | 3.87 GiB |

Three things to take from that. **Reads are flat**: a full scan of 190 GiB costs
0.13 GiB more than a full scan of 1 GiB, because nothing accumulates. Projection
pushdown is the biggest lever -- not reading the 512-byte column is the difference
between 0.28 and 0.59 GiB, and it never stops paying. And `sink_lance` grows
slowly rather than with the result, which is what lets it write a 190 GiB source
in under 4 GiB of RAM.

## Development

```sh
uv run pytest                    # 202 tests
uv run pytest -m "not cloud"     # 176 -- what CI runs
uv run mypy                      # strict, over src, tests and bench
uv run basedpyright
uvx ruff check . && uvx ruff format --check .
# benchmarking:
uv run --group bench bench/plot.py bench/results-m8id4xl.jsonl --out bench/plots
```

## Polars Cloud

> **Not installable today.** polars-cloud 0.10 pins `polars==1.43.2`, below this
> package's `polars>=1.44.0` floor, so there is no `cloud` extra and the two
> cannot be resolved together. Everything in this section is written and kept
> working against the 0.10 API; it becomes usable when polars-cloud ships a
> release tracking 1.44.

Reads are designed to ship: a scan serializes to a few kB and carries a URI,
never an open dataset handle. Workers need `pylance` and `polars-pylance`
installed, via `ComputeContext(requirements=...)` --
`polars_pylance.cloud.requirements_txt()` renders the pinned lines, since Polars
Cloud rejects a context whose polars version differs from the client's.

### Writing from a remote query

polars-cloud 0.10 added `sink_batches()`, which hands each result batch to a
Python callable. That callable is **cloudpickled into the serialized query
plan**, so it runs on the workers -- Lance no longer has to be a sink format
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
different workers", and appending a fragment is not idempotent -- so each staging
object is named after a deterministic digest of its batch, and a replayed batch
overwrites its own metadata instead of adding a second copy. `tests/test_remote.py`
delivers every batch twice and asserts the row count is unchanged. Pass
`fragment_key=` if your query has a natural key, such as a partition column;
the default digest would collapse two *distinct* batches that are byte-identical.
Replays do leave their earlier data files unreferenced -- reclaim them with
`dataset.cleanup_old_versions(..., delete_unverified=True)`.

Drive the query yourself with `stage_lance_sink()` when you want to pick a
planner or inspect the query handle:

```python
staged = pll.cloud.stage_lance_sink("s3://bucket/out.lance", lf, mode="overwrite")
query = lf.remote(ctx).distributed(planner="miso").sink_batches(staged.callback)
query.await_result()
staged.commit()
```

### Reading

The scan survives `prepare_cloud_plan`, on its own and under `pl.concat()` of
`scan_lance_fragments()` shards. Since 0.9 distributes unions of Python scans, the sharded form is the
sanctioned way to fan a read across workers rather than a workaround, and 0.10's
`pl.collect_all(lazyframes, lazy=True).remote(ctx).distributed().execute()`
submits N shards as one distributed query instead of N remote ones. Note that
`collect_all(lazy=True)` requires every LazyFrame to end in a sink.

Still untested without a workspace: how the planner *executes* those nodes.
Serializing is necessary, not sufficient.
