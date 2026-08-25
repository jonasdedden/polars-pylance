## Why the private hook

`scan_lance` used to ship a second implementation on the public
`polars.io.plugins.register_io_source`, as a fallback in case
`new_from_dataset_object` ever went away. It was removed in favour of the one
path, because the public route is not merely equivalent-but-safer -- it is slower
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
re-apply per batch, and gets no limit it can hand to Lance -- so `head(5)` still
pulls a whole Lance batch. Where neither matters, the two are level.

Keeping the fallback also meant keeping a hand-written Polars-expression → Lance
SQL translator (~180 lines) whose only job was recovering some of that pushdown.
Deleting the path deleted the translator with it.
