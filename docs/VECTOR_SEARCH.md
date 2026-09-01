# Vector search

`scan_lance(nearest={...})` searches a vector column. The result carries a
`_distance` column and is ordered by it.

```python
search = {"column": "embedding", "q": query, "k": 3}
pll.scan_lance("docs.lance", nearest=search).select("id", "cat", "_distance").collect()
```

```
shape: (3, 3)
┌─────┬──────┬───────────┐
│ id  ┆ cat  ┆ _distance │
│ --- ┆ ---  ┆ ---       │
│ i64 ┆ str  ┆ f32       │
╞═════╪══════╪═══════════╡
│ 0   ┆ blog ┆ 0.0       │
│ 461 ┆ blog ┆ 0.165233  │
│ 314 ┆ faq  ┆ 0.263081  │
└─────┴──────┴───────────┘
```

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

A prefilter picks the `k` nearest **among the rows it admits**:

```python
pll.scan_lance(uri, nearest=search, prefilter="cat = 'docs'")
```

```
┌─────┬──────┬───────────┐
│ id  ┆ cat  ┆ _distance │
╞═════╪══════╪═══════════╡
│ 261 ┆ docs ┆ 0.26594   │
│ 329 ┆ docs ┆ 0.301591  │
│ 346 ┆ docs ┆ 0.461135  │
└─────┴──────┴───────────┘
```

A downstream `.filter()` ranks first and filters after, so it keeps whichever of
the 3 nearest happen to be `docs`. On the same data and query as above, that is
none of them:

```python
pll.scan_lance(uri, nearest=search).filter(pl.col("cat") == "docs")
```

```
shape: (0, 3)
┌─────┬─────┬───────────┐
│ id  ┆ cat ┆ _distance │
╞═════╪═════╪═══════════╡
└─────┴─────┴───────────┘
```

The empty result is the distinction, not a bug. A downstream `.filter()` is
still pushed into Lance where it translates, but only into the postfilter
position; it is never promoted to a prefilter.

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
