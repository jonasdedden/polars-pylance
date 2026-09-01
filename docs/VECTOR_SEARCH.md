# Vector search

`scan_lance(nearest={...})` searches a vector column. The result carries a `_distance` column and is ordered by it.

```python
lf = pll.scan_lance(
    "docs.lance",
    nearest={"column": "embedding", "q": embedding, "k": 10, "metric": "cosine"},
    prefilter="category = 'docs'",
)
```

The `nearest` dict is handed to Lance as-is, so every key `LanceDataset.scanner(nearest=...)` takes works here and new ones arrive without a release of this package. `metric` is `"l2"`, `"cosine"` or `"dot"`; omitted, Lance uses whichever the index was built with. `use_index=False` forces an exact flat search, which is the ground truth an indexed search approximates. `nprobes`, `refine_factor`, `ef`, `distance_range` and the rest pass through the same way.

## Two ways to restrict a search

They answer different questions, so they stay separate:

| | when it runs | how many rows |
| --- | --- | --- |
| `prefilter=` | before the search | `k`, chosen from the rows it admits |
| `.filter()` on the result | after the search | `k` or fewer, out of what was ranked |

```python
search = {"column": "v", "q": q, "k": 10}

# k rows, all of them 'docs'
pll.scan_lance(uri, nearest=search, prefilter="cat = 'docs'")

# 10 nearest overall, then whichever of those are 'docs' -- possibly none
pll.scan_lance(uri, nearest=search).filter(pl.col("cat") == "docs")
```

A downstream `.filter()` is still pushed into Lance where it translates, but only into the postfilter position. It is never promoted to a prefilter.

Lance has one filter slot (`scanner(filter=..., prefilter=bool)`), so the two cannot both be pushed. A prefilter takes the slot, and the query's own filter is evaluated in Polars instead. That costs some pushdown and keeps the positions honest.

Without a search there is nothing to rank, and a prefilter is simply a filter the caller pushed by hand.

## Why a prefilter can fail

A pushed-down predicate is allowed to lower *loosely*. An untranslatable conjunct of an `AND` is dropped, Lance reads a superset, and Polars re-applies the original per batch, so the answer does not depend on how much of it Lance understood; `docs/PUSHDOWN.md` covers this.

A prefilter decides which rows get ranked at all. There is no second pass that can repair a wrong candidate set: the search has already happened. So the relaxation that is safe for a postfilter is not safe here, and a prefilter that does not translate exactly raises instead:

```python
>>> pll.scan_lance(uri, nearest=search, prefilter=pl.col("id").hash() % 2 == 0)
ValueError: prefilter does not translate to a Lance filter: ...
```

The refusal lands at the call site, before anything is read. Lance rejecting the filter at plan time raises for the same reason, where an ordinary pushed-down predicate would warn and re-scan without it.

Pass Lance SQL as a string to say exactly what should be pushed, bypassing the translation entirely.

## Indices

A prefilter is where a scalar index earns its place: it runs before the vector search, so Lance can use `cat`'s BTREE to pick the candidate set instead of scanning it. The plan for a prefiltered search over an indexed column shows both nodes:

```python
dataset.scanner(nearest=..., filter="cat = 'docs'", prefilter=True).explain_plan(True)
# ANNIvfPartition ... ScalarIndexQuery ...
```

Ranking order survives the scan, and `.head(n)` truncates the ranked result rather than re-running the search with a smaller `k`.
