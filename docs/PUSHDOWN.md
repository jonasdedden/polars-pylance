# Predicate pushdown

`scan_lance` is an IO plugin, so Polars hands it the whole filter as a
`polars.Expr`. `polars_pylance._predicate` walks that expression and emits the
equivalent Lance SQL filter string, which is what
`LanceDataset.scanner(filter=...)` takes.

## Coverage

55 predicate shapes run through a real scan, recording whether Lance's scanner
was handed a filter at all:

| | shapes reaching Lance |
| --- | --- |
| Polars' PyArrow lowering (`scan_pyarrow_dataset`, `scan_delta`) | **11 / 55** |
| this translation | **52 / 55** |

PyArrow can express comparisons, `AND`/`OR`/`NOT`, null checks, `is_between`,
`all_horizontal` and datetime literals. It drops the rest silently.

Reaching Lance only here: `is_in` (any length), `xor`, `eq_missing`, every
string function below, all arithmetic, `abs`, `**`, `min_horizontal` /
`max_horizontal`, `fill_null`, `is_nan` / `is_not_nan` / `is_infinite` /
`is_finite`, `cast`, `dt.year` / `month` / `hour` / `weekday` / `date` /
`truncate`, `list.contains` / `len` / `get`, `struct.field`, `concat_str`.

## Relaxation

A lowering may be a superset. An untranslatable conjunct of an `AND` is
dropped, so `a > 5 & a.str.slice(0, 2) == "xy"` still pushes `a > 5`. That is
only sound in positive position, so a dropped branch of an `OR` or anything
under a `NOT` declines instead. Every lowering carries an `exact` flag; when it
is false, `scan_lance` re-applies the original predicate to each batch.

## Deliberate declines

Each of these has a Lance spelling that means something slightly different.

| construct | why |
| --- | --- |
| `round` | Polars breaks ties to even, Lance away from zero |
| `sqrt`, `ln`, `log10`, `cbrt` | outside their domain Polars gives NaN, Lance NULL |
| `a ** b`, fractional or negative exponent | same |
| `str.strip_chars()` with no argument | Polars strips Unicode whitespace, `btrim` strips spaces |
| `str.replace(..., literal=True)` | SQL's `replace` is literal but replaces every occurrence |
| `str.contains_any(ascii_case_insensitive=True)` | Polars folds ASCII only |
| `str.to_titlecase` | `initcap` disagrees on words starting with a digit |
| `str.slice` | Lance has no `substr` |
| `is_in(nulls_equal=True)` | `IN` propagates NULL |
| `list.get` without `null_on_oob` | Polars raises where `array_element` returns null |
| `dt.epoch` | `date_part('epoch', ...)` keeps the fraction |
| `dt.truncate` of a multiple ("2d") | `date_trunc` takes a unit, not a window |
| `//` | Polars floors, SQL truncates |
| narrowing integer cast | Polars raises on overflow, Lance wraps |
| non-strict cast, unless widening | Polars yields null, Lance fails the scan |
| `Time` and `Duration` literals | Lance has no matching type |
| `when/then` | Lance rejects `CASE` |

`is_nan` is a special case: it has to be `isnan(x)`, not `x != x`. Lance
compares by SQL's total ordering, under which NaN equals itself, so the second
spelling drops every NaN row.

## What it buys

4M rows, a 256-byte payload column, 1.1 GiB on disk, `polars` 1.44.0 and
`pylance` 9.0.0, best of five, one process per measurement, against the
dataset-provider scan this replaces.

A selective filter in front of a wide column, `filter(...).select("id",
"payload")`. Wall time and peak RSS:

| predicate | provider | SQL filter | |
| --- | --- | --- | --- |
| `id.is_in(100 values)` | 0.556 s / 1951 MiB | 0.045 s / 328 MiB | **12.3x** |
| `id.is_in(200 values)` | 0.496 s / 2274 MiB | 0.045 s / 324 MiB | **11.0x** |
| `val * 2 > 1.9995` | 0.384 s / 578 MiB | 0.031 s / 327 MiB | **12.3x** |
| `(id > 3.99M) xor (val > 0.999)` | 0.400 s / 563 MiB | 0.032 s / 333 MiB | **12.4x** |
| `text.str.contains(...)`, 1 in 4k | 0.556 s / 2361 MiB | 0.091 s / 360 MiB | **6.1x** |
| `ts.dt.hour() < 1` | 0.441 s / 1163 MiB | 0.095 s / 436 MiB | **4.7x** |
| `val > 0.999` (both push it) | 0.028 s | 0.027 s | par |

With scalar indices on the filtered columns (BTREE, NGRAM on `text`):
`is_in(100)` **16.6x**, `text.str.contains` **19.4x**, `val * 2 > 1.9995`
**14.2x**. An index cannot help a predicate that never reaches Lance, so the
gap widens rather than closes.

## What it costs

With a narrow projection there is no wide column to skip, and Lance evaluating
the predicate row by row is slower than Polars evaluating it on the decoded
column:

| predicate, `select(pl.len())` | provider | SQL filter | |
| --- | --- | --- | --- |
| `text.str.contains(...) & id > 3.99M` | 0.016 s | 0.048 s | 3.0x slower |
| `ts.dt.hour() < 1` | 0.032 s | 0.088 s | 2.8x slower |
| `id.is_in(200 values)` | 0.032 s | 0.069 s | 2.1x slower |
| `cat.str.starts_with("bet")`, indexed | 0.125 s | 0.341 s | 2.7x slower |
| full scan, no filter | 0.011 s | 0.016 s | +5 ms |

The last row is fixed cost: the plugin resolves its schema through a second
dataset open. It does not grow with the data.

The rest is Lance's SQL evaluation being the expensive part when nothing is
saved by it, plus, on the indexed run, a BTREE lookup for a predicate that
keeps a quarter of the table producing a row-id list bigger than the scan it
replaces. Both are data-dependent.
`LanceScanOptions(use_scalar_index=False)` turns off the second, and
`scan_lance(..., predicate_pushdown=False)` turns off the translation.
