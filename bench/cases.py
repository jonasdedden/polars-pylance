"""One (impl, size, case) measurement per process. JSON to stdout.

Separate processes because peak RSS is a per-process high-water mark: one
materialising run would poison every later reading in the same interpreter.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any, cast

import polars as pl

IMPL, ROWS, CASE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ROOT = Path(os.environ.get("BENCH_ROOT", "/mnt/nvme")) / "data"
URI = str(ROOT / f"{ROWS // 1_000_000}m.lance")
OUT = str(ROOT.parent / f"out-{IMPL}-{ROWS // 1_000_000}m-{CASE}.lance")


def scan(**kw: Any) -> pl.LazyFrame:
    if IMPL == "polars-lance":
        # the comparison package; deliberately not a dependency of this project
        from polars_lance import scan_lance as their_scan

        return cast("pl.LazyFrame", their_scan(URI))
    from polars_pylance import scan_lance

    return scan_lance(URI, **kw)


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


# import cost is not what we are measuring
if IMPL == "polars-lance":
    import polars_lance  # noqa: F401
else:
    import polars_pylance  # noqa: F401

READS = {
    # whole 512-byte column has to move
    "r_full": lambda lf: lf.select(pl.col("payload").is_not_null().sum()),
    # projection pushdown: only `val` should ever be read
    "r_proj": lambda lf: lf.select(pl.col("val").sum()),
    # very selective numeric predicate
    "r_filter_lo": lambda lf: lf.filter(pl.col("val") > 0.999).select(pl.len()),
    # half the rows, and it drags the payload along
    "r_filter_hi": lambda lf: lf.filter(pl.col("val") > 0.5).select(
        pl.col("payload").is_not_null().sum()
    ),
    # string equality predicate
    "r_cat": lambda lf: lf.filter(pl.col("cat") == "a").select(pl.len()),
    # limit pushdown: should stop almost immediately
    "r_head": lambda lf: lf.head(10),
    # top-k: needs the sort, cannot stop early
    "r_topk": lambda lf: lf.sort("val", descending=True).head(10),
}

t0 = time.perf_counter()
extra: dict[str, Any] = {}

if CASE in READS:
    out = READS[CASE](scan()).collect(engine="streaming")
    result = out.height if CASE in ("r_head", "r_topk") else out.item()

elif CASE == "w_sink":
    shutil.rmtree(OUT, ignore_errors=True)
    lf = scan().filter(pl.col("val") > 0.5).select("id", "cat", "val", "payload")
    if IMPL == "polars-lance":
        from polars_lance import write_lance

        # no streaming writer on their side: the result must be materialised
        write_lance(lf.collect(engine="streaming"), OUT, mode="overwrite")
    else:
        from polars_pylance import sink_lance

        sink_lance(lf, OUT, mode="overwrite")
    import lance

    result = lance.dataset(OUT).count_rows()

elif CASE == "w_frag":  # polars-pylance only
    from polars_pylance import scan_lance_fragments, write_lance_fragments

    shutil.rmtree(OUT, ignore_errors=True)
    shards = [
        s.filter(pl.col("val") > 0.5).select("id", "cat", "val", "payload")
        for s in scan_lance_fragments(URI, n_shards=16)
    ]
    ds = write_lance_fragments(shards, OUT, mode="overwrite", max_workers=16)
    result = ds.count_rows()
    extra["shards"] = len(shards)

elif CASE == "l_version":  # polars-pylance only: time travel
    import lance

    from polars_pylance import scan_lance

    v = lance.dataset(URI).version
    result = (
        scan_lance(URI, version=v).select(pl.len()).collect(engine="streaming").item()
    )
    extra["version"] = v

elif CASE == "l_rowid":  # polars-pylance only: Lance-generated column
    from polars_pylance import scan_lance

    result = (
        scan_lance(URI, with_row_id=True)
        .select(pl.col("_rowid").max())
        .collect(engine="streaming")
        .item()
    )

elif CASE == "l_fragments":  # polars-pylance only: sharded per-fragment scan
    from polars_pylance import scan_lance_fragments

    parts = scan_lance_fragments(URI, n_shards=16)
    result = sum(
        p.select(pl.col("val").sum()).collect(engine="streaming").item() for p in parts
    )
    extra["shards"] = len(parts)

else:
    raise SystemExit(f"unknown case {CASE}")

elapsed = time.perf_counter() - t0
shutil.rmtree(OUT, ignore_errors=True)
print(
    json.dumps(
        {
            "impl": IMPL,
            "rows": ROWS,
            "case": CASE,
            "seconds": elapsed,
            "peak_gib": peak_gib(),
            "result": result,
            **extra,
        }
    )
)
