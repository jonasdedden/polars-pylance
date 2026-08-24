# Comparison with `jorritsandbrink/polars-lance`

[`jorritsandbrink/polars-lance`](https://github.com/jorritsandbrink/polars-lance)
(PyPI `polars-lance` 0.5.0) solves the same problem with a different architecture:
a compiled Rust extension linking the `lance` crate, versus this package's pure
Python built on `pylance`.

Everything below was measured on 2026-07-31 against their published wheel 0.5.0
and this working copy, on the same machine and the same datasets. Harnesses:
`compare.py`, `compare_nulls.py` (in the session scratchpad); their repo at commit
`f95cf2d`.

**Naming:** `polars-lance` on PyPI is theirs. This package is published as
`polars-pylance` -- named for what it is, `polars` plus `pylance` composed in
Python, against their `lance` crate compiled in. Its import name is
`polars_pylance`, so both can be installed side by side.

---

## At a glance

| | theirs (0.5.0) | this package |
| --- | --- | --- |
| Implementation | Rust extension (`pyo3`, `maturin`), links `lance` 0.38.2 | Pure Python on `pylance` 9 |
| Polars hook | `register_io_source` | dataset-provider hook |
| Runtime deps | `polars>=1.0.0` only — Lance is statically linked | `polars>=1.44.0`, `pylance>=9`, `pyarrow` |
| Read | lazy, streaming | lazy, streaming |
| Write | eager `DataFrame` only | streaming from a `LazyFrame` |
| Predicate pushdown into Lance | **no** (acknowledged `TODO`; filters in Rust polars) | yes |
| Published | PyPI, 15 wheels, 5 releases | not yet |
| CI | Linux/macOS/Windows × py3.10–3.14 | none yet |
| Cloud-storage tests | yes (MinIO via testcontainers, Azure) | no |
| Test functions | 16 | 93 (parametrized over both scan impls) |
| Lines of code | 1 112 Rust + 142 Python | 1 233 Python |
| License | MIT | MIT |
| Repo activity | 43 commits, 2026-05-09 → 2026-05-21 | new |

## Architecture

Theirs calls `register_io_source` with a Rust `LanceScanner` behind it, so
per-batch work never touches the interpreter. Lance is compiled in: no `pylance`
at runtime, verified by installing their wheel with only polars present — both
scan and write work. The cost is a 60–68 MB platform wheel per Python version
(15 wheels for 0.5.0) and coupling to polars' internal Rust API.

This package resolves scans through the same hook `pl.scan_delta` and
`pl.scan_iceberg` use, which yields a richer pushdown surface. It needs `pylance`
(a 76 MB wheel) but is itself pure Python: no build step, no per-platform wheels,
and it inherits every Lance feature `pylance` exposes.

## Read features

| | theirs | this package |
| --- | --- | --- |
| Projection pushdown | yes | yes |
| Row-limit pushdown | yes | yes (when no filter follows the scan) |
| Predicate pushdown into Lance | no | yes (PyArrow expression, or SQL on the fallback path) |
| Early stop on `.head()` | yes | yes |
| `storage_options` (S3/Azure/GCS) | yes | yes |
| Version / tag pinning, time travel | no | yes |
| Vector search (`nearest`) | no | yes |
| Full-text search | no | yes |
| `_rowid` / `_rowaddr` | no | yes |
| Per-fragment scans, sharding | no | yes (`scan_lance_fragments`) |
| Reader tuning (`io_buffer_size`, readahead) | no | yes (`LanceScanOptions`) |
| Scan-plan serialization (Polars Cloud) | untested | yes, 2–6 kB, round-trip tested |
| Distributed write to Lance (Polars Cloud) | no | yes (`cloud.sink_lance_remote`) |

Their `scan_lance` takes exactly two arguments (`source`, `storage_options`).
The Lance-specific capabilities — vector search, versioning, fragments — are the
main functional gap, and are the reason this package exposes them as scan
arguments: they cannot be expressed as polars expressions, so an IO plugin has to
surface them explicitly.

Their missing predicate pushdown is a deliberate, documented `TODO` in
`src/scan.rs`:

```rust
// TODO: Translate the Polars `Expr` into a `LanceFilter` and push the predicate
// down into the Lance scanner.
```

In practice it costs little on a small local dataset — Lance still gets the
projection, and filtering in Rust polars is fast — but it forfeits page skipping
and scalar-index use, which is what makes a selective filter cheap on a large or
remote dataset.

## Write features

| | theirs | this package |
| --- | --- | --- |
| Input | `pl.DataFrame` (eager) | `pl.LazyFrame` (streamed) |
| Larger-than-memory writes | no — caller must chunk manually | yes |
| Modes | `error`, `append`, `overwrite` | `create`, `append`, `overwrite`, `merge` (upsert) |
| File-layout control | `max_rows_per_file`, `max_bytes_per_file` | all `lance.write_dataset` kwargs |
| Deferred / composable sink | no | yes (`lazy=True`) |
| Parallel fragment write + single commit | no | yes (`write_lance_fragments`) |

This is the sharpest difference. `write_lance(df, ...)` requires a materialized
`DataFrame`, so writing the result of a large query means holding it in RAM:

```python
polars_lance.write_lance(lf.collect(), "out.lance")  # theirs: full result in memory
polars_pylance.sink_lance(lf, "out.lance")  # here: streamed batch by batch
```

Measured on the 527 MB source, writing a 500 101-row filtered projection:
**1 242 MB peak RSS theirs vs 774 MB here** — and theirs grows with the result
size while this does not.

## Correctness (measured)

Ten predicates over a 12-row dataset containing nulls in both filtered columns,
checked against polars' own semantics on an eagerly loaded frame
(`compare_nulls.py`, polars 1.43.1):

| predicate | theirs | mine |
| --- | --- | --- |
| `col == "a"` | ok | ok |
| `col != "a"` | ok | ok |
| `~(col == "a")` | ok | ok |
| `val > 0.5` | ok | ok |
| `col.is_null()` | **ComputeError** | ok |
| `val.is_not_null()` | **ComputeError** | ok |
| `and`, `or`, `not_or` | ok | ok |
| nested `and`/`or`/`is_null` | **ComputeError** | ok |

All 10 of this package's results match polars exactly, which is the important
result for a package that *does* push predicates down: pushing them into Lance
does not change which rows come back, including the null cases where SQL and
polars semantics could have diverged.

Theirs fails on `is_null`/`is_not_null` with:

```
BindingsError: Error when deserializing 'Expr'. This may be due to mismatched
polars versions. OtherString("invalid value: integer `4`, expected variant
index 0 <= i < 3")
```

Their Rust side (polars 0.53, `pyo3-polars` 0.26) cannot deserialize the
`Boolean` function variants a newer Python polars emits. I checked polars 1.31.0,
1.32.3, 1.34.0, 1.38.1, 1.42.1 and 1.43.1 — `is_null` fails on **all** of them,
so this is a genuine gap rather than a fixable version mismatch. Their filter test
covers only `pl.col("int32") > 0`, which is why it went unnoticed.

Two more behavioural cases:

| query | theirs | this package |
| --- | --- | --- |
| `filter(...).head(7)` | ok, 7 rows | ok, 7 rows |
| `filter(...).sort(...).head(7)` | **ComputeError** | ok |
| `sort(...).head(3)` | **ComputeError** | ok |

The top-k failures are the upstream `dynamic_pred` bug: a `sort().head()` made
polars push an opaque, unevaluable node into an IO plugin's predicate, and theirs
hits it because it also evaluates the pushed predicate with polars. The bug ran
from polars 1.39.0 through 1.43.2 and is fixed in 1.44.0 — verified with a
standalone `register_io_source` reproducer, where 1.43.2 panics four ways and
1.44.0 passes cleanly. The rows above were measured against polars 1.43.1 and so
predate that fix; theirs was not re-measured on 1.44.0.

## Performance (best of 3, 527 MB source, polars 1.43.1)

| case | theirs | this package (default) | this package (`throughput()`) |
| --- | --- | --- | --- |
| full scan + aggregate over payload column | 0.22 s | 0.33 s | 0.23 s |
| projection-only aggregate | 0.01 s | 0.02 s | not measured |
| selective filter + aggregate | 0.01 s | 0.02 s | not measured |
| `select(pl.len())` | 0.01 s | 0.01 s | not measured |

On the only case big enough to measure, the 50 % gap closes entirely once this
package's memory-conservative defaults are swapped for
`LanceScanOptions.throughput()` (0.23 s vs 0.22 s). So the difference is the
readahead defaults, not Rust versus Python per-batch overhead. Their throughput
comes with Lance's 2 GiB `io_buffer_size` default and no way to turn it down.

## Memory (peak RSS, same 527 MB source)

| case | theirs | this package |
| --- | --- | --- |
| streaming aggregation over payload column | 937 MB | 623 MB |
| write a 500 k-row filtered projection | 1 242 MB | 774 MB |
