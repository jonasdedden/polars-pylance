"""Render dist-results.jsonl as one comparison table per case."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import backends, queries

# Measured on the ladder `bench/gen.py` writes: 105 GiB / 207M rows. Shared
# with `bench/analyse.py`, which prints its tiers the same way.
GIB_PER_ROW = 0.5072 / 1e6


def load(path: str) -> list[dict[str, Any]]:
    """Read results, skipping a trailing partial line from a live run."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"(skipped one malformed line: {line[:60]}...)", file=sys.stderr)
    return out


def row(name: str, rec: dict[str, Any] | None, fastest: float) -> str:
    """One backend's line for one tier.

    Cores as well as seconds, because "how much of the machine did it use"
    is most of what separates these backends, and a fast run at two cores
    means something different from a fast run at eight.
    """
    if rec is None:
        return f"  {name:<10} not run"
    if rec.get("status") != "ok":
        return f"  {name:<10} {rec['status']}  {rec.get('detail', '')[:60]}"
    verdict = rec.get("verify", "unverified")
    flag = "" if verdict == "ok" else f"  [!] {verdict}"
    count = rec.get("result_rows", rec.get("result_count", "?"))
    return (
        f"  {name:<10} {rec['seconds']:7.2f}s  {rec['seconds'] / fastest:5.2f}x  "
        f"{rec['cores_busy']:4.1f} cores  {rec['cpu_seconds']:7.1f} cpu-s  "
        f"{rec['mem_rise_gib']:5.2f}G peak ({rec['peak_gib']:.2f}G driver)  "
        f"rows={count}{flag}"
    )


def main() -> None:
    """Print one section per case, one block per tier, one line per backend."""
    by_case: defaultdict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for rec in load(sys.argv[1]):
        if rec["impl"] == "baseline":
            _print_baseline(rec)
            continue
        by_case[(rec["case"], rec["rows"], rec.get("setting", ""))][rec["impl"]] = rec

    for (case, rows, setting), results in sorted(by_case.items()):
        ok = [results[n] for n in backends.NAMES if _succeeded(results.get(n))]
        fastest = min((r["seconds"] for r in ok), default=1.0)
        case_name = queries.CaseName(case)
        title = f"\n=== [{case}] {rows / 1e6:.0f}M rows / {rows * GIB_PER_ROW:.1f} GiB"
        if setting:
            title += f" / {setting}"
        print(title + f" -- {queries.CASES[case_name].blurb} ===")
        for name in backends.NAMES:
            print(row(name, results.get(name), fastest))
        if ok:
            print(f"  -- {summary(ok)}")
        if case_name is queries.CaseName.W_COMMIT and ok:
            print(f"  -- {_commit_trend(ok)}")


def _print_baseline(rec: dict[str, Any]) -> None:
    print(
        f"host while idle: {rec['cores_busy']:.2f} cores busy, "
        f"{rec['mem_peak_gib']:.2f} GiB in use, memory via "
        f"{rec['mem_source']}. `peak` below is the rise above the "
        "run's own starting point."
    )


def _succeeded(rec: dict[str, Any] | None) -> bool:
    return rec is not None and rec.get("status") == "ok"


def summary(ok: list[dict[str, Any]]) -> str:
    """The line under a tier: what all backends agree on, and who won."""
    counts = {rec.get("result_rows", rec.get("result_count")) for rec in ok}
    rows = counts.pop() if len(counts) == 1 else f"DISAGREE {sorted(counts, key=repr)}"
    fastest = min(ok, key=lambda rec: rec["seconds"])
    # Only the plan-shipping backends have scheduler traffic to report; Ray
    # Data moves its blocks through Ray's object store instead.
    shipped = max(rec.get("plan_bytes", 0) for rec in ok)
    returned = max(rec.get("metadata_bytes", 0) for rec in ok)
    budgets = {rec.get("cpu_budget") for rec in ok}
    # A speed ranking across unequal budgets is not a ranking of the
    # schedulers, so say so rather than print it as though it were.
    budget = (
        f"{budgets.pop()} CPUs each"
        if len(budgets) == 1
        else f"UNEQUAL BUDGETS {sorted(b for b in budgets if b is not None)}"
    )
    return (
        f"rows={rows}, {budget}, fastest={fastest['impl']} "
        f"({fastest['seconds']:.1f}s, commit {fastest['commit_seconds']:.2f}s), "
        f"scheduler traffic {shipped / 1024:.0f} KiB out / "
        f"{returned / 1024:.1f} KiB back"
    )


def _commit_trend(ok: list[dict[str, Any]]) -> str:
    """Commit time against fragment count: the commit-stress question."""
    parts = sorted(
        f"{r['impl']} {r.get('n_fragments', '?')} frags -> "
        f"commit {r['commit_seconds']:.3f}s of {r['seconds']:.2f}s"
        for r in ok
    )
    return "commit share: " + "; ".join(parts)


if __name__ == "__main__":
    main()
