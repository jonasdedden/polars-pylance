"""Distributed benchmark: the sharded pipeline on six backends.

Layout, by area:

- `queries` -- the five cases, each one Narwhals query run on every engine.
- `backends` -- platform code: how each engine runs a case (`threads`,
  `ray-core`, `dask`, `ray-data`, `daft`, `daft-ray`), plus the shared
  plan-shipping pipeline in `sharded`.
- `metrics` -- measurement code: wall time, CPU and peak memory over the
  query only, never cluster startup.
- `driver` -- the case matrix and its per-process measurements.
- `analyse` -- renders `dist-results.jsonl` as comparison tables.
"""
