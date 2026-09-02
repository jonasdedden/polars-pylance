# polars-pylance

[![PyPI](https://img.shields.io/pypi/v/polars-pylance.svg)](https://pypi.org/project/polars-pylance/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-pylance.svg)](https://pypi.org/project/polars-pylance/)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://jonasdedden.github.io/polars-pylance/)
[![CI](https://github.com/jonasdedden/polars-pylance/actions/workflows/ci.yml/badge.svg)](https://github.com/jonasdedden/polars-pylance/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/polars-pylance.svg)](https://pypi.org/project/polars-pylance/)

Lazy, streaming [Lance](https://lance.org) <-> [Polars](https://pola.rs) integration.

`scan_lance()` returns a real `LazyFrame`: the Polars optimizer pushes column
projections, filters and row limits down into Lance, and batches are pulled only
as the streaming engine consumes them. `sink_lance()` writes a query into Lance
batch by batch. Neither direction holds the dataset or a whole fragment in memory.

## Installation

```sh
pip install polars-pylance
```

## Quick start

The following example creates a local Lance dataset, then lazily scans, filters
and collects it:

```python
import polars as pl
import polars_pylance as pll

source = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "score": [0.72, 0.95, 0.98],
        "text": ["draft", "guide", "reference"],
    }
).lazy()
pll.sink_lance(source, "docs.lance", mode="overwrite")

result = (
    pll.scan_lance("docs.lance")
    .filter(pl.col("score") > 0.9)
    .select("id", "text")
    .collect(engine="streaming")
)
```

Passing a local path or an object-store URI such as
`s3://bucket/embeddings.lance` works the same way. Supply credentials and other
object-store settings with `storage_options=`.

## The name

`polars` + `pylance`, composed in Python. There is no compiled extension here and
no `lance` crate linked in: the package is pure Python over the two libraries it
joins, which is why it installs as a single wheel, tracks new `pylance` releases
without a rebuild, and reaches Lance features as fast as `pylance` exposes them.

## Comparison with `polars-lance`

Note that PyPI's `polars-lance` is an unrelated package by a different author
(extensive [benchmarks and feature insights](https://jonasdedden.github.io/polars-pylance/dev/COMPARISON/) for comparison).
The two can coexist in one environment but follow different implementation designs.

## Why this exists

Polars has no native Lance reader or writer
([pola-rs/polars#14452](https://github.com/pola-rs/polars/issues/14452) has been
open since 2024). Lance datasets do implement the PyArrow dataset protocol, so
`pl.scan_pyarrow_dataset` works, but it cannot push down row limits, cannot pin a
dataset version, cannot reach vector or full-text search, and leaves Lance's
read-ahead defaults untouched where `io_buffer_size` alone defaults to 2 GiB.

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

What gets pushed into Lance: the column projection, the row limit, and the
filter, translated into Lance's own SQL filter language so that scalar indices,
page statistics and late materialisation all apply. `.head()` stops the scan
early rather than reading to the end. `scan_lance` walks the Polars expression
itself and emits Lance SQL:

```python
>>> pll.to_lance_filter(pl.col("cat").str.starts_with("b") & pl.col("id").is_in([1, 2]))
LanceFilter(sql="(starts_with(`cat`, 'b') AND (`id` IN (1, 2)))", exact=True)
```

A predicate that only partly translates is pushed as far as it goes and finished
in Polars (`exact=False` says so), so the answer never depends on how much of it
Lance understood. `predicate_pushdown=False` turns the whole thing off. What this
is worth, measured with and without scalar indices, is in
[predicate pushdown](https://jonasdedden.github.io/polars-pylance/dev/PUSHDOWN/).

### Vector search

`nearest=` searches a vector column, returning a LazyFrame ordered by
`_distance`. `prefilter=` restricts what that search may return:

```python
lf = pll.scan_lance(
    "docs.lance",
    nearest={"column": "embedding", "q": embedding, "k": 10, "metric": "cosine"},
    prefilter="category = 'docs'",  # or a Polars expression
)
```

`prefilter=` runs before the search, so it returns `k` rows chosen from the ones
it admits. A downstream `.filter()` runs after, over rows already ranked, and so
may return fewer than `k`; it is never promoted to a prefilter. A prefilter that
does not translate exactly raises rather than quietly becoming a postfilter,
since nothing downstream can repair a candidate set the search has already used.
The [vector search guide](https://jonasdedden.github.io/polars-pylance/dev/VECTOR_SEARCH/) has the details.

### Full-text search

`full_text_query=` exposes Lance's index-backed full-text search. It returns a
LazyFrame with a `_score` column, so normal Polars operations can refine the
ranked result:

```python
matches = (
    pll.scan_lance("docs.lance", full_text_query="streaming")
    .select("id", "text", "_score")
    .collect(engine="streaming")
)
```

The dataset must have a Lance inverted index on the searched text column. Pass
a string to use Lance's default search behavior, or select the column explicitly
with `{"query": "streaming", "columns": ["text"]}`.

## Writing

`sink_lance()` runs a query and hands Lance each batch as it comes, so the
result is written incrementally and never materialised in full. It also accepts
an eager `DataFrame` for convenience, though that input is necessarily already
in memory. Use a `LazyFrame` when the data starts in a file or another query:
the query and write then run at the same time, and peak memory is one batch
rather than one dataset.

```python
pll.sink_lance(lf, "out.lance", mode="append", max_rows_per_file=1_000_000)
pll.sink_lance(updates, "out.lance", mode="merge", on="id")  # upsert
plan = pll.sink_lance(lf, "out.lance", lazy=True)  # write on collect
```

## Sharded end-to-end pipelines

`scan_lance_fragments()` returns one lazy query per fragment (or shard), and
`write_lance_fragments()` executes those queries and writes their output
fragments concurrently. A single commit then publishes all fragments as one
dataset version:

```python
shards = pll.scan_lance_fragments("data.lance", n_shards=4)
pll.write_lance_fragments(
    [shard.filter(pl.col("score") > 0.9) for shard in shards],
    "filtered.lance",
    mode="overwrite",
    max_workers=4,
)
```

Each shard stays lazy and bounded in memory; the full result is never collected
before writing. A lazy `pl.concat(shards)` is also streaming, but Polars
currently pulls registered Python IO sources one after another, so concatenation
alone does not execute these shard scans concurrently.

For distributed execution, workers can write fragment files separately;
`commit_lance_fragments()` publishes their metadata as one dataset version
afterward. It is what the [Polars Cloud](#polars-cloud) path commits with once
the workers are done.

## Memory behaviour

Peak RSS for `polars-pylance` across the size ladder in
[`bench/`](https://jonasdedden.github.io/polars-pylance/dev/BENCHMARKS/), on datasets from 1 GiB to 49.2 GiB, a **49x**
increase in data:

**Reads**

| | 1 GiB source | 49.2 GiB source |
| --- | --- | --- |
| Projection-only scan, payload column never read | 0.20 GiB | 0.25 GiB |
| Sharded fragment scan (`scan_lance_fragments`) | 0.20 GiB | 0.27 GiB |
| Substring filter + payload aggregate | 0.22 GiB | 0.32 GiB |
| 50% filter + payload aggregate | 0.79 GiB | 1.24 GiB |
| Full scan + aggregate over the payload column | 0.46 GiB | 0.58 GiB |

Reads are flat: a full scan of 49 GiB costs 0.12 GiB more than a full scan of
1 GiB, because nothing accumulates. Projection pushdown is the biggest lever,
and a pushed-down filter is the next one: not reading the 512-byte column for
rows that will not survive is the difference between 0.32 and 1.24 GiB.

**Writes**

| | 1 GiB source | 49.2 GiB source |
| --- | --- | --- |
| `sink_lance` (scan -> transform -> write) | 0.99 GiB | 1.60 GiB |
| `write_lance_fragments` (parallel, 16 shards) | 0.90 GiB | 7.40 GiB |

`sink_lance` grows slowly rather than with the result, which is what lets it
write a 49 GiB source in under 2 GiB of RAM where an eager writer needs 51 GiB.
The fragment-parallel path trades memory for wall time: 16 shards write at once,
so its peak tracks the shard count.

## Development

```sh
uv run pytest
uv run pytest -m "not cloud"     # what CI runs
uv run mypy                      # strict, over src, tests and bench
uv run basedpyright
uv run --only-group lint ruff check .   # the version in uv.lock, as CI uses
uv run --only-group lint ruff format --check .
uv run --group docs properdocs build --strict -f mkdocs.yml
# benchmarking:
uv run --group bench bench/plot.py bench/results-m8id4xl.jsonl --out bench/plots
```

## Polars Cloud

Reads serialize into a cloud query plan, and since polars-cloud 0.10 the write
runs on the workers too: `sink_batches()` cloudpickles a Lance fragment writer
into the plan, and a single client-side commit publishes what they wrote. It is
not installable today, because polars-cloud pins a polars below this package's
floor.

The [Polars Cloud guide](https://jonasdedden.github.io/polars-pylance/dev/POLARS_CLOUD/) has the whole story.
