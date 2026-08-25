"""Render results.jsonl as per-case scaling tables."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

GIB_PER_MROW = 0.4915  # measured: 190.9 GiB / 388M rows

PHASES = {
    "scaling": "generous cap: how peak RSS and runtime grow with dataset size",
    "fixed-budget": "memory pinned, data grown past it: does the job finish?",
}


def _load(path: str) -> list[dict[str, Any]]:
    """Read results, skipping any trailing partial line.

    The file is appended to while the matrix runs, so fetching it mid-run can
    catch a half-written record.
    """
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"(skipped one malformed line: {line[:60]}...)", file=sys.stderr)
    return out


rows = _load(sys.argv[1])
by: defaultdict[tuple[Any, str, int], dict[str, Any]] = defaultdict(dict)
for r in rows:
    by[(r.get("phase"), r["case"], r["rows"])][r["impl"]] = r

CASE_LABEL = {
    "r_full": "full scan + aggregate over payload",
    "r_proj": "projection-only aggregate (val)",
    "r_filter_lo": "highly selective filter (val > 0.999)",
    "r_filter_hi": "50% filter + payload aggregate",
    "r_cat": "string predicate (cat == 'a')",
    "r_head": "head(10) - limit pushdown",
    "r_topk": "sort + head(10) - top-k",
    "w_sink": "write filtered projection",
    "w_frag": "fragment-parallel write (polars-pylance only)",
    "l_version": "pinned version / time travel (polars-pylance only)",
    "l_rowid": "_rowid column (polars-pylance only)",
    "l_fragments": "sharded fragment scan (polars-pylance only)",
}


def cell(r: dict[str, Any] | None) -> str:
    if r is None:
        return "n/a"
    if r.get("status") != "ok":
        label = {"OOM-killed": "**OOM**", "panic-unsendable": "*panic*"}
        return label.get(r["status"], str(r["status"]))
    return f"{r['seconds']:.1f}s / {r['peak_gib']:.2f}G"


for phase, blurb in PHASES.items():
    cases = sorted(
        {c for (p, c, _) in by if p == phase}, key=lambda c: list(CASE_LABEL).index(c)
    )
    if not cases:
        continue
    print(f"\n{'=' * 78}\n{phase.upper()}  —  {blurb}\n{'=' * 78}")
    for case in cases:
        sizes = sorted({n for (p, c, n) in by if p == phase and c == case})
        print(f"\n{CASE_LABEL.get(case, case)}  [{case}]")
        print(
            f"  {'source':>9} | {'polars-lance':>18} | {'polars-pylance':>18} | ratio"
        )
        for n in sizes:
            d = by[(phase, case, n)]
            t, o = d.get("polars-lance"), d.get("polars-pylance")
            ratio = ""
            if t and o and t.get("status") == "ok" and o.get("status") == "ok":
                ratio = (
                    f"{o['seconds'] / t['seconds']:.2f}x time, "
                    f"{o['peak_gib'] / t['peak_gib']:.2f}x mem"
                )
            gib = n * GIB_PER_MROW / 1e6
            print(f"  {gib:8.1f}G | {cell(t):>18} | {cell(o):>18} | {ratio}")
