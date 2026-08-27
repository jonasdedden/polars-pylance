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


# polars-lance links its own polars crate and re-decodes the predicate Polars
# hands it. An expression built by a newer polars either fails to decode or
# decodes as a different variant, and the error escapes its scan node instead of
# being reported as "predicate not applied". These are the signatures.
DECODE_ERRORS = (
    "Error when deserializing",
    "operation not supported for dtype",
    "is not allowed in this context",
)


def collect(query: pl.LazyFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Collect `query`, retrying without predicate pushdown if the scan refuses.

    The retry is what the reader would have done anyway: neither implementation
    under test pushes this predicate into Lance, so filtering in Polars is the
    same work, measured the same way. Without it the case would be a bare
    failure and there would be nothing to compare against.
    """
    global t0
    t0 = time.perf_counter()
    try:
        return query.collect(engine="streaming"), {}
    except Exception as exc:
        if not any(m in str(exc) for m in DECODE_ERRORS):
            raise
    # The first attempt raised while building the scanner, before reading, so it
    # costs a little wall time and no memory. Only the retry is timed.
    t0 = time.perf_counter()
    frame = query.collect(
        engine="streaming",
        optimizations=pl.QueryOptFlags(predicate_pushdown=False),
    )
    return frame, {"pushdown": "declined-by-scan"}


# import cost is not what we are measuring
if IMPL == "polars-lance":
    import polars_lance  # noqa: F401
else:
    import polars_pylance  # noqa: F401

# 200 ids spread across the whole dataset, so the predicate is selective without
# being local to one fragment.
IS_IN_IDS = list(range(0, ROWS, max(1, ROWS // 200)))

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
    # set membership in front of the payload. PyArrow has no `is_in`, so this is
    # the first shape polars-pylance pushes and nothing else can.
    "r_is_in": lambda lf: lf.filter(pl.col("id").is_in(IS_IN_IDS)).select(
        pl.col("payload").is_not_null().sum()
    ),
    # substring search, one row in 10_000, again in front of the payload
    "r_str": lambda lf: lf.filter(
        pl.col("text").str.contains("-rare", literal=True)
    ).select(pl.col("payload").is_not_null().sum()),
    # a computed predicate: the comparison is against an expression, not a column
    "r_arith": lambda lf: lf.filter((pl.col("val") * 2) > 1.999).select(
        pl.col("payload").is_not_null().sum()
    ),
    # a temporal part, which no PyArrow expression can carry either
    "r_temporal": lambda lf: lf.filter(pl.col("ts").dt.hour() < 1).select(
        pl.col("payload").is_not_null().sum()
    ),
    # limit pushdown: should stop almost immediately
    "r_head": lambda lf: lf.head(10),
    # top-k: needs the sort, cannot stop early
    "r_topk": lambda lf: lf.sort("val", descending=True).head(10),
}

t0 = time.perf_counter()
extra: dict[str, Any] = {}

if CASE in READS:
    out, note = collect(READS[CASE](scan()))
    extra.update(note)
    result = out.height if CASE in ("r_head", "r_topk") else out.item()

elif CASE == "r_cat_noindex":  # polars-pylance only: the same query, index off
    from polars_pylance import LanceScanOptions

    lf = scan(options=LanceScanOptions(use_scalar_index=False))
    result = (
        lf.filter(pl.col("cat") == "a")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )

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
