"""Drive the distributed case matrix: every backend on every ladder tier.

One process per measurement, like the single-node `run_matrix.py`: Ray and
Dask both start background threads and child processes, so peak RSS is only
meaningful when each measurement gets its own interpreter. Every output is
read back before the next tier runs, because a benchmark that is not also a
correctness check is a benchmark of the wrong thing.

Usage:
    BENCH_ROOT=/mnt/fast-nvme uv run --group dataframe \
        python -m bench.dataframe.driver.matrix 2000000,4000000 55
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bench.dataframe import backends, queries
from bench.dataframe.metrics import cpu, memory
from bench.dataframe.queries import CaseName

ROOT = os.environ.get("BENCH_ROOT", "/mnt/nvme")
PY = os.environ.get("BENCH_PYTHON", sys.executable)
RESULTS = Path(ROOT) / "dataframe-results.jsonl"
# Memory caps need systemd with cgroup v2; `BENCH_CAPS=0` runs each
# measurement directly, like the single-node driver.
CAPS = os.environ.get("BENCH_CAPS", "1") != "0"
# `cases` imports the package by name, so it runs from the repository root.
REPO = Path(__file__).parent.parent.parent.parent


def run(backend: str, case: CaseName, rows: int, cap_gib: int) -> list[dict[str, Any]]:
    """Run one backend on one case and tier. Never raises: failures are records.

    A list, because the commit-stress case sweeps several shard counts in
    one warmed process and prints one record each.
    """
    scope = (
        [
            "systemd-run",
            "--scope",
            "-q",
            "-p",
            f"MemoryMax={cap_gib}G",
            "-p",
            "MemorySwapMax=0",
            "--",
        ]
        if CAPS
        else []
    )
    proc = subprocess.run(
        [
            *scope,
            PY,
            "-m",
            "bench.dataframe.driver.cases",
            backend,
            case.value,
            str(rows),
        ],
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
        cwd=REPO,
    )
    records = [
        {
            **json.loads(line),
            "status": "ok",
            "cap_gib": cap_gib if CAPS else None,
        }
        for line in proc.stdout.splitlines()
        if line.startswith("{")
    ]
    if records:
        return records
    err = proc.stderr
    if proc.returncode in (137, -9) or "Killed" in err or "MemoryError" in err:
        failure: dict[str, Any] = {"status": "OOM-killed"}
    else:
        tail = (
            err.strip().splitlines()[-1][:200]
            if err.strip()
            else f"rc={proc.returncode}"
        )
        failure = {"status": "failed", "detail": tail}
    return [
        {
            "impl": backend,
            "rows": rows,
            "case": case.value,
            "cap_gib": cap_gib if CAPS else None,
            **failure,
        }
    ]


def verify(case: CaseName, rows: int, records: list[dict[str, Any]]) -> dict[str, str]:
    """Check one case's outputs on one tier, per backend. Returns verdicts.

    Read cases compare the reductions the workers returned -- exact
    integers, no tolerance involved. Write cases re-scan each output with a
    streaming checksum (count and id sum exact, value sum within tolerance)
    instead of collecting whole datasets, which is what makes verification
    affordable at the top of the ladder.
    """
    done = {r["impl"]: r for r in records if r.get("status") == "ok"}
    if queries.CASES[case].kind == "read":
        counts = {
            name: (r["result_count"], r["result_sum_id"]) for name, r in done.items()
        }
        if len(set(counts.values())) != 1:
            return dict.fromkeys(done, f"reduction disagrees: {counts}")
        return dict.fromkeys(done, "ok")

    try:
        sums = {
            name: _checksum(backends.output_uri(ROOT, name, case, rows))
            for name in done
        }
    except Exception as exc:
        return dict.fromkeys(done, f"unreadable: {exc}")
    verdicts: dict[str, str] = {}
    reference = next(iter(sums.values()))
    for name, total in sums.items():
        problems = [
            key
            for key in ("n", "s_id", "s_val", "columns")
            if not _matches(key, total[key], reference[key])
        ]
        verdicts[name] = "ok" if not problems else f"differs in {problems}"
    return verdicts


def _checksum(uri: str) -> dict[str, Any]:
    """One streaming pass over a dataset: count, id sum, value sum, columns."""
    import polars_pylance as pll

    frame = pll.scan_lance(uri).select("id", "val").collect(engine="streaming")
    total: dict[str, Any] = dict(queries.checksum(frame))
    total["columns"] = ",".join(pll.scan_lance(uri).collect_schema().names())
    return total


def _matches(key: str, got: Any, want: Any) -> bool:
    """Whether one checksum field agrees. Floats get tolerance, rest exact."""
    if key == "s_val":
        return queries.agree(got, want)
    return bool(got == want)


def emit(rec: dict[str, Any]) -> None:
    """Append one record to the results file and summarise it on stdout."""
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if rec["impl"] == "baseline":
        return
    size = f"{rec['rows'] // 1_000_000}M"
    if rec.get("status") == "ok":
        count = rec.get("result_rows", rec.get("result_count", "?"))
        print(
            f"  {rec['impl']:9} {rec['case']:9} {size:>5} {rec['seconds']:8.2f}s "
            f"{rec['cores_busy']:5.1f} cores  {rec['mem_rise_gib']:5.2f} GiB"
            f"  rows={count}",
            flush=True,
        )
    else:
        print(
            f"  {rec['impl']:9} {rec.get('case', '?'):9} {size:>5} "
            f"{rec['status'].upper()}  {rec.get('detail', '')[:70]}",
            flush=True,
        )


def baseline(seconds: float = 3.0) -> dict[str, Any]:
    """What the host is doing when the benchmark is not.

    `cpu_seconds` is machine-wide, so a host that is busy with something
    else inflates every measurement. Sampling the idle machine once puts a
    number on that instead of assuming it away: on a dedicated benchmark box
    `cores_busy` here is ~0, and anything else is the correction a reader
    has to apply.
    """
    usage = cpu.Usage.start()
    start = time.perf_counter()
    with memory.Sampler() as ram:
        time.sleep(seconds)
    return {
        "impl": "baseline",
        "case": "idle",
        "rows": 0,
        "status": "ok",
        **usage.stop(time.perf_counter() - start),
        **ram.as_record(),
    }


def _cases() -> list[CaseName]:
    raw = os.environ.get("DIST_CASES")
    if raw is None:
        return list(queries.CASES)
    names = [part.strip() for part in raw.split(",")]
    by_value = {c.value: c for c in CaseName}
    unknown = [name for name in names if name not in by_value]
    if unknown:
        msg = f"unknown cases {unknown}; expected {sorted(by_value)}"
        raise SystemExit(msg)
    return [by_value[name] for name in names]


def main() -> None:
    """Run every backend over every case and tier, verifying each combination."""
    ladder = [int(x) for x in sys.argv[1].split(",")]
    cap_gib = int(sys.argv[2]) if len(sys.argv) > 2 else 55
    budget = backends.Parallelism.from_env()
    cases = _cases()
    distributed_tiers = [rows for rows in ladder if rows >= backends.LOCAL_MAX_ROWS]
    if distributed_tiers and not (os.environ.get("DIST_CLUSTER") or None):
        tiers = ", ".join(f"{rows // 1_000_000}M" for rows in distributed_tiers)
        msg = (
            f"tiers {tiers} need the distributed backends "
            f"({', '.join(backends.DISTRIBUTED)}), which run against the "
            "cluster behind DIST_CLUSTER -- set it to the Ray address or "
            "Dask scheduler both the driver and the workers reach"
        )
        raise SystemExit(msg)

    print(
        f"=== distributed pass: {cap_gib} GiB cap, "
        f"{budget.slots} slots x {budget.threads} threads "
        f"= {budget.cpus} CPUs per backend; "
        f"cases {', '.join(c.value for c in cases)} ===",
        flush=True,
    )
    idle = baseline()
    emit(idle)
    print(
        f"  host baseline: {idle['cores_busy']:.2f} cores busy, "
        f"{idle['mem_peak_gib']:.2f} GiB in use while idle "
        f"(memory read from the {idle['mem_source']})",
        flush=True,
    )
    for rows in ladder:
        names = backends.tier_backends(rows)
        print(f"-- {rows // 1_000_000}M rows runs {', '.join(names)} --", flush=True)
        for case in cases:
            blurb = queries.CASES[case].blurb
            print(
                f"-- {rows // 1_000_000}M rows [{case.value}] {blurb} --",
                flush=True,
            )
            records = [rec for name in names for rec in run(name, case, rows, cap_gib)]
            # Verified once per (case, tier) rather than once per backend:
            # the check is partly a cross-backend comparison within the
            # tier's group, so it needs them all finished first.
            verdicts = verify(case, rows, records)
            bad = {
                name: verdict for name, verdict in verdicts.items() if verdict != "ok"
            }
            for rec in records:
                if rec.get("status") == "ok":
                    rec["verify"] = verdicts.get(rec["impl"], "nothing to verify")
                emit(rec)
            for name, verdict in bad.items():
                print(f"  VERIFY {name}: {verdict}", flush=True)
    print("BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
