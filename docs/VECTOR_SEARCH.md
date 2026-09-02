# Vector search

`scan_lance(nearest={...})` searches a vector column. The result carries a
`_distance` column and is ordered by it.

The examples below use two-dimensional vectors so the distances can be checked
by eye. Real embeddings have hundreds of dimensions; nothing else differs.

Create the small dataset used throughout the guide:

```python
import polars as pl
import polars_pylance as pll

data = pl.DataFrame(
    {
        "id": range(7),
        "cat": ["blog", "blog", "blog", "faq", "docs", "docs", "docs"],
        "embedding": pl.Series(
            [
                [0.1, 0.0],
                [0.0, 0.2],
                [0.3, 0.0],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.0, 1.5],
                [2.0, 0.0],
            ],
            dtype=pl.Array(pl.Float32, 2),
        ),
    }
)
pll.sink_lance(data, "tiny.lance", mode="overwrite")
```

```python
pll.scan_lance("tiny.lance").collect(engine="streaming")
```

```
shape: (7, 3)
┌─────┬──────┬───────────────┐
│ id  ┆ cat  ┆ embedding     │
│ --- ┆ ---  ┆ ---           │
│ i64 ┆ str  ┆ array[f32, 2] │
╞═════╪══════╪═══════════════╡
│ 0   ┆ blog ┆ [0.1, 0.0]    │
│ 1   ┆ blog ┆ [0.0, 0.2]    │
│ 2   ┆ blog ┆ [0.3, 0.0]    │
│ 3   ┆ faq  ┆ [0.5, 0.5]    │
│ 4   ┆ docs ┆ [1.0, 0.0]    │
│ 5   ┆ docs ┆ [0.0, 1.5]    │
│ 6   ┆ docs ┆ [2.0, 0.0]    │
└─────┴──────┴───────────────┘
```

Searching for the three nearest to the origin returns the three points closest
to it, which happen to be the `blog` rows:

```python
search = {"column": "embedding", "q": [0.0, 0.0], "k": 3}
pll.scan_lance("tiny.lance", nearest=search).collect(engine="streaming")
```

```
shape: (3, 4)
┌─────┬──────┬───────────────┬───────────┐
│ id  ┆ cat  ┆ embedding     ┆ _distance │
│ --- ┆ ---  ┆ ---           ┆ ---       │
│ i64 ┆ str  ┆ array[f32, 2] ┆ f32       │
╞═════╪══════╪═══════════════╪═══════════╡
│ 0   ┆ blog ┆ [0.1, 0.0]    ┆ 0.01      │
│ 1   ┆ blog ┆ [0.0, 0.2]    ┆ 0.04      │
│ 2   ┆ blog ┆ [0.3, 0.0]    ┆ 0.09      │
└─────┴──────┴───────────────┴───────────┘
```

Note `_distance` for the default `l2` metric is the **squared** euclidean
distance: the point at `[0.3, 0.0]` is 0.3 away and scores 0.09. Ranking is
unaffected, but a threshold has to be squared to match.

The `nearest` dict is handed to Lance as-is, so every key
`LanceDataset.scanner(nearest=...)` takes works here and new ones arrive without
a release of this package. `metric` is `"l2"`, `"cosine"` or `"dot"`; omitted,
Lance uses whichever the index was built with. `use_index=False` forces an exact
flat search, which is the ground truth an indexed search approximates.
`nprobes`, `refine_factor`, `ef`, `distance_range` and the rest pass through the
same way.

## Two ways to restrict a search

They answer different questions, so they stay separate:

| | when it runs | how many rows |
| --- | --- | --- |
| `prefilter=` | before the search | `k`, chosen from the rows it admits |
| `.filter()` on the result | after the search | `k` or fewer, out of what was ranked |

A prefilter searches only the `docs` rows, so it returns three of them, ranked:

```python
pll.scan_lance("tiny.lance", nearest=search, prefilter="cat = 'docs'").collect(
    engine="streaming"
)
```

```
shape: (3, 4)
┌─────┬──────┬───────────────┬───────────┐
│ id  ┆ cat  ┆ embedding     ┆ _distance │
│ --- ┆ ---  ┆ ---           ┆ ---       │
│ i64 ┆ str  ┆ array[f32, 2] ┆ f32       │
╞═════╪══════╪═══════════════╪═══════════╡
│ 4   ┆ docs ┆ [1.0, 0.0]    ┆ 1.0       │
│ 5   ┆ docs ┆ [0.0, 1.5]    ┆ 2.25      │
│ 6   ┆ docs ┆ [2.0, 0.0]    ┆ 4.0       │
└─────┴──────┴───────────────┴───────────┘
```

A downstream `.filter()` ranks first and filters after. The three nearest to the
origin are all `blog`, so filtering them for `docs` leaves nothing:

```python
(
    pll.scan_lance("tiny.lance", nearest=search)
    .filter(pl.col("cat") == "docs")
    .collect(engine="streaming")
)
```

```
shape: (0, 4)
┌─────┬─────┬───────────────┬───────────┐
│ id  ┆ cat ┆ embedding     ┆ _distance │
│ --- ┆ --- ┆ ---           ┆ ---       │
│ i64 ┆ str ┆ array[f32, 2] ┆ f32       │
╞═════╪═════╪═══════════════╪═══════════╡
└─────┴─────┴───────────────┴───────────┘
```

The empty result is the distinction, not a bug: the `docs` rows exist and the
prefilter found them, but none of them are among the three nearest overall. A
downstream `.filter()` is still pushed into Lance where it translates, but only
into the postfilter position; it is never promoted to a prefilter.

Lance has one filter slot (`scanner(filter=..., prefilter=bool)`), so the two
cannot both be pushed. A prefilter takes the slot, and the query's own filter is
evaluated in Polars instead.

## Why a prefilter can fail

A pushed-down predicate may lower *loosely*: an untranslatable conjunct is
dropped, Lance reads a superset, and Polars re-applies the original per batch
(see [predicate pushdown](PUSHDOWN.md)).

A prefilter decides which rows get ranked at all, and there is no second pass
that can repair a wrong candidate set. So it raises rather than quietly becoming
a postfilter:

```python
pll.scan_lance(uri, nearest=search, prefilter=pl.col("id").hash() % 2 == 0)
```

```
ValueError: prefilter does not translate to a Lance filter:
[([(col("id").hash()) % (dyn int: 2)]) == (dyn int: 0)]. It is not applied as a
postfilter, because filtering the ranked result is a different question;
rewrite it, or pass the Lance SQL as a string.
```

The refusal lands at the call site, before anything is read. Pass Lance SQL as a
string to say exactly what should be pushed, bypassing the translation entirely.

## Indices

A prefilter is where a scalar index earns its place: it runs before the vector
search, so Lance can use `cat`'s BTREE to pick the candidate set instead of
scanning it. The plan for a prefiltered search over an indexed column shows both
nodes:

```python
dataset.scanner(nearest=..., filter="cat = 'docs'", prefilter=True).explain_plan(True)
# ANNIvfPartition ... ScalarIndexQuery ...
```

Ranking order survives the scan, and `.head(n)` truncates the ranked result
rather than re-running the search with a smaller `k`.
