"""Drive the case matrix on the instance. One process per measurement, each in
its own cgroup scope so an over-budget run is killed cleanly instead of swapping.

Two passes, answering different questions:

"scaling"      - generous cap, so nothing is constrained: how do peak RSS and
                 runtime grow with dataset size?
"fixed-budget" - memory pinned and the data grown past it: does the job finish
                 at all on the RAM you actually have?
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("BENCH_ROOT", "/mnt/nvme"))
PY = os.environ.get("BENCH_PYTHON", str(ROOT / "venv/bin/python"))
CASES = str(Path(__file__).with_name("cases.py"))
RESULTS = ROOT / "results.jsonl"

READ_CASES = [
    "r_full",
    "r_proj",
    "r_filter_lo",
    "r_filter_hi",
    "r_cat",
    "r_head",
    "r_topk",
]
PYLANCE_ONLY = ["w_frag", "l_version", "l_rowid", "l_fragments"]


def run(
    impl: str, rows: int, case: str, cap_gib: int, attempts: int = 3
) -> dict[str, Any]:
    """Run one case, retrying past polars-lance's unsendable-thread panic."""
    last: dict[str, Any] = {}
    for attempt in range(attempts):
        p = subprocess.run(
            [
                "systemd-run",
                "--scope",
                "-q",
                "-p",
                f"MemoryMax={cap_gib}G",
                "-p",
                "MemorySwapMax=0",
                "--",
                PY,
                CASES,
                impl,
                str(rows),
                case,
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        for line in p.stdout.splitlines():
            if line.startswith("{"):
                return {
                    **json.loads(line),
                    "status": "ok",
                    "cap_gib": cap_gib,
                    "attempts": attempt + 1,
                }
        err = p.stderr
        if "unsendable" in err:
            last = {"status": "panic-unsendable"}
            continue  # retry: it is probabilistic
        if p.returncode in (137, -9) or "Killed" in err or "MemoryError" in err:
            return {
                "impl": impl,
                "rows": rows,
                "case": case,
                "cap_gib": cap_gib,
                "status": "OOM-killed",
                "attempts": attempt + 1,
            }
        tail = (
            err.strip().splitlines()[-1][:200] if err.strip() else f"rc={p.returncode}"
        )
        last = {"status": "failed", "detail": tail}
    return {
        "impl": impl,
        "rows": rows,
        "case": case,
        "cap_gib": cap_gib,
        "attempts": attempts,
        **last,
    }


def emit(rec: dict[str, Any]) -> None:
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    size = f"{rec['rows'] // 1_000_000}M" if "rows" in rec else "?"
    if rec.get("status") == "ok":
        print(
            f"  {rec['impl']:7} {size:>5} {rec['case']:<12} "
            f"{rec['seconds']:8.1f}s  {rec['peak_gib']:6.2f} GiB"
            f"{'  (retried)' if rec.get('attempts', 1) > 1 else ''}",
            flush=True,
        )
    else:
        print(
            f"  {rec['impl']:7} {size:>5} {rec['case']:<12} "
            f"{rec['status'].upper()}  {rec.get('detail', '')[:70]}",
            flush=True,
        )


if __name__ == "__main__":
    ladder = [int(x) for x in sys.argv[1].split(",")]
    big_cap, small_cap = int(sys.argv[2]), int(sys.argv[3])

    print(
        f"=== scaling pass: {big_cap} GiB cap (headroom, not a limit) ===",
        flush=True,
    )
    for rows in ladder:
        print(f"-- {rows // 1_000_000}M rows --", flush=True)
        cases = list(READ_CASES)
        if rows > 100_000_000:
            # a sort has to materialise; at this scale it exceeds RAM on both
            # sides, so it measures the machine rather than either package.
            cases.remove("r_topk")
        for case in [*cases, "w_sink"]:
            for impl in ("polars-lance", "polars-pylance"):
                emit({**run(impl, rows, case, big_cap), "phase": "scaling"})
        for case in PYLANCE_ONLY:
            emit({**run("polars-pylance", rows, case, big_cap), "phase": "scaling"})

    print(f"\n=== fixed-budget pass: {small_cap} GiB, swap off ===", flush=True)
    for rows in ladder:
        for case in ("r_full", "w_sink"):
            for impl in ("polars-lance", "polars-pylance"):
                emit({**run(impl, rows, case, small_cap), "phase": "fixed-budget"})
    print("BENCHMARK COMPLETE", flush=True)
