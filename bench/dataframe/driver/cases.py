"""One (backend, size) measurement per process. JSON to stdout.

Separate processes for the same reason as the single-node `cases.py`: peak
RSS is a per-process high-water mark, and Ray and Dask both start background
threads and child processes that would poison later readings in one
interpreter. Each backend owns its cluster lifecycle; this file only reads
the environment, calls the one it was named, and weighs the answer.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
from typing import Any

from bench.dataframe import backends, queries
from bench.dataframe.queries import CaseName


def peak_gib() -> float:
    """This process's high-water resident memory, in GiB.

    The coordinator's alone: `RUSAGE_SELF` excludes the worker processes,
    which is where a distributed run spends its memory. `cpu_seconds` in the
    same record is machine-wide and does cover them.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _settings(case: CaseName, shards: int) -> list[tuple[str | None, int]]:
    """(setting label, shard count) pairs one `case` measures. Usually one.

    `w_commit` sweeps the shard count at fixed output: each shard writes at
    least one fragment whatever the file-size knob says, so the shard count
    is what moves the fragment count, and the trend of commit time against
    it is the measurement. The cluster is warmed once and shared, which is
    exactly right now that startup is excluded from the timing.
    """
    if case is not CaseName.W_COMMIT:
        return [(None, shards)]
    return [
        (f"shards={count}", count)
        for count in _int_list("DIST_COMMIT_SHARDS", queries.COMMIT_SHARDS)
    ]


def _int_list(env: str, default: tuple[int, ...]) -> list[int]:
    raw = os.environ.get(env)
    if raw is None:
        return list(default)
    return [int(part) for part in raw.split(",")]


def main() -> None:
    """Measure one backend on one case and tier, named by argv."""
    name, raw_case, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        case = CaseName(raw_case)
    except ValueError:
        msg = f"unknown case {raw_case!r}; expected {[c.value for c in CaseName]}"
        raise SystemExit(msg) from None
    root = os.environ.get("BENCH_ROOT", "/mnt/nvme")
    src = backends.source_uri(root, rows)
    run = backends.load(name)
    spec = queries.CASES[case]
    chunk = int(os.environ.get("DIST_CHUNK", str(queries.DEFAULT_CHUNK_SIZE)))
    budget = backends.Parallelism.from_env()
    cluster = os.environ.get("DIST_CLUSTER") or None
    shards = int(os.environ.get("DIST_SHARDS", "16"))

    # No timer here: every backend times its own query, having first built
    # and warmed whatever cluster it needs. Measuring from out here would
    # measure Ray's boot time as though it were the query.
    for setting, n_shards in _settings(case, shards):
        dst = (
            None if spec.kind == "read" else backends.output_uri(root, name, case, rows)
        )
        if dst is not None:
            shutil.rmtree(dst, ignore_errors=True)
        measurement = run(
            src,
            dst,
            case=case,
            n_shards=n_shards,
            chunk_size=chunk,
            parallelism=budget,
            cluster=cluster,
        )
        record: dict[str, Any] = {
            "impl": name,
            # The tier, which keys the matrix. The output's own count is
            # `result_rows` (writes) or `result_count` (reads), and the
            # driver checks one against the other.
            "rows": rows,
            "case": case.value,
            "peak_gib": peak_gib(),
            # Carries `seconds` and the CPU/memory counters, for the query
            # alone: the backend measured them around it.
            **measurement,
        }
        if setting is not None:
            record["setting"] = setting
        print(json.dumps(record))


# Guarded so Dask's spawn-based workers can re-import this module without
# re-running the measurement in every child process.
if __name__ == "__main__":
    main()
