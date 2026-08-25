## Why the private hook is the default

`scan_lance` can scan through either of two Polars hooks: the private
dataset-provider hook (`impl="provider"`, the default) or the public
`register_io_source` (`impl="io_plugin"`). The provider hook is the default
because Polars resolves the scan while building the IR and hands over the
projection, the row limit and a filter it has already lowered to a PyArrow
expression -- which is exactly what Lance's scanner accepts -- and because a
resolved scan is re-used across collects and serializes to a smaller plan.

Best of 4 runs per case, one process each, on the 527 MB / 1 M-row dataset from
`bench/`, peak anonymous RSS. This is the measurement that decided the default;
it was taken against an IO-plugin implementation that **did not pass the row
limit to Lance** and re-applied the predicate in Python:

| query | provider | io_plugin (as it was then) | |
| --- | --- | --- | --- |
| `head(5)` | 0.005 s / 177 MB | 0.033 s / 245 MB | **5.5× slower, +68 MB** |
| filter + sort + `head(7)` | 0.098 s / 554 MB | 0.165 s / 659 MB | **1.7× slower, +105 MB** |
| full scan + aggregate | 0.151 s / 407 MB | 0.234 s / 465 MB | **1.5× slower, +58 MB** |
| projection-only aggregate | 0.016 s / 182 MB | 0.017 s / 156 MB | par |
| filter pushdown, count | 0.039 s / 182 MB | 0.040 s / 186 MB | par |

Two of those three gaps turned out to be the implementation rather than the
hook, and both are gone in the current one:

- Polars *does* hand an IO plugin `n_rows`, under the same rule the provider
  path gets it (only when no filter sits above the scan). Passing it through to
  Lance's scanner closes the `head()` gap entirely.
- The full-scan gap was the engine's 100k-row batch-size hint, which the old
  implementation always took. `LanceScanOptions.batch_size` now wins over the
  hint, and the two paths are level on a full scan.

What remains is a per-scan overhead of roughly 0.07 s per 4M rows on a filtered
narrow scan, a plan that serializes to 5.0 kB rather than 1.8 kB, and the loss
of resolution caching across collects. That is enough to keep the provider hook
as the default, and small enough that it is worth paying whenever the filter is
one Polars cannot lower -- which is most of the expression language.

The hand-written Polars-expression → Lance SQL translator that this document
once described as dead weight is back, considerably wider, and is now the reason
the IO-plugin path exists at all. See
[`PREDICATE_PUSHDOWN.md`](PREDICATE_PUSHDOWN.md) for the coverage table and the
measurements behind the recommendation.

Both hooks are unstable API: one is private and carries no stability guarantee,
the other is marked unstable upstream. If a future Polars changes either
interface, pin the previous polars-pylance rather than expecting a fallback.
