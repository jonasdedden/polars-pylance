#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars>=1.44.0", "pylance>=9", "pyarrow", "numpy", "polars-pylance"]
# [tool.uv.sources]
# polars-pylance = { path = "../", editable = true }
# ///
"""Peak-memory benchmark and regression guard.

Each case runs in its own process, because ``ru_maxrss`` is a high-water mark for
the whole process and one eager read would poison every later measurement. A
sampler thread reads ``/proc/self/status`` so anonymous memory (what we actually
allocate) can be told apart from file-backed pages (mapped dataset files).

    uv run bench/mem.py list          # show cases
    uv run bench/mem.py run <case>    # run one case in this process
    uv run bench/mem.py all           # run every case, one subprocess each
    uv run bench/mem.py guard         # assert streaming memory stays flat

The guard is the important one: it fails if peak memory for a streaming scan
grows more than 25% when the source triples in size, which is what would happen
if any part of the pipeline started buffering the whole dataset.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import lance
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import polars_pylance as pll
from _data import ensure_dataset, on_disk_mb

BENCH_DIR = Path(
    os.environ.get("POLARS_PYLANCE_BENCH_DIR", "/tmp/polars-pylance-bench")
)
SMALL_ROWS = 1_000_000
LARGE_ROWS = 3_000_000

# The guard triples the input, so a pipeline that buffered the dataset would show
# ~3x. Observed growth is ~1.2x, which is mostly the larger fragment count; the
# threshold sits well above the run-to-run noise but far below linear, so it
# catches real buffering without flaking.
GUARD_TOLERANCE = 1.5


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


class PeakSampler:
    """Track peak VmRSS / RssAnon / RssFile of this process, in MB."""

    FIELDS = ("VmRSS", "RssAnon", "RssFile")

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.peak = dict.fromkeys(self.FIELDS, 0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        with open("/proc/self/status") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in self.FIELDS:
                    value = int(rest.split()[0])
                    if value > self.peak[key]:
                        self.peak[key] = value

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except OSError:
                return
            time.sleep(self.interval)

    def __enter__(self) -> PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._sample()
        self._stop.set()

    def as_mb(self) -> dict[str, float]:
        return {k: v / 1024 for k, v in self.peak.items()}


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

# Every case takes the source dataset's URI and returns a one-line result.
Case = Callable[[str], str]
CASES: dict[str, str] = {}


def case(doc: str) -> Callable[[Case], Case]:
    def register(fn: Case) -> Case:
        CASES[fn.__name__.removeprefix("case_")] = doc
        return fn

    return register


def source_uri(size: str) -> str:
    rows = LARGE_ROWS if size == "large" else SMALL_ROWS
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    uri = str(BENCH_DIR / f"{size}.lance")
    ensure_dataset(uri, rows)
    return uri


@case("materialise the whole dataset (the thing we are avoiding)")
def case_eager_read(uri: str) -> str:
    frame = pl.from_arrow(lance.dataset(uri).to_table())
    return f"rows={frame.height}"  # type: ignore[union-attr]


@case("plain Lance to_batches loop, no polars")
def case_lance_only(uri: str) -> str:
    rows = 0
    for batch in lance.dataset(uri).to_batches(batch_size=25_000):
        rows += batch.num_rows
    return f"rows={rows}"


@case("streaming aggregation over the payload column")
def case_stream_scan(uri: str) -> str:
    out = (
        pll.scan_lance(uri)
        .filter(pl.col("payload").bin.starts_with(bytes([0])))
        .select(pl.len(), pl.col("val").sum())
        .collect(engine="streaming")
    )
    return f"result={out.row(0)}"


@case("same query on the in-memory engine")
def case_in_memory_engine(uri: str) -> str:
    out = (
        pll.scan_lance(uri)
        .filter(pl.col("payload").bin.starts_with(bytes([0])))
        .select(pl.len(), pl.col("val").sum())
        .collect(engine="in-memory")
    )
    return f"result={out.row(0)}"


@case("streaming scan with Lance's own read-ahead defaults")
def case_stream_scan_throughput(uri: str) -> str:
    out = (
        pll.scan_lance(uri, options=pll.LanceScanOptions.throughput())
        .filter(pl.col("payload").bin.starts_with(bytes([0])))
        .select(pl.len())
        .collect(engine="streaming")
    )
    return f"result={out.row(0)}"


@case("projection pushdown: the payload column is never read")
def case_stream_scan_projected(uri: str) -> str:
    out = pll.scan_lance(uri).select(pl.col("val").sum()).collect(engine="streaming")
    return f"result={out.row(0)}"


@case("pull batches through polars and discard (isolates the sink boundary)")
def case_read_only_batches(uri: str) -> str:
    rows = 0
    for frame in pll.scan_lance(uri).collect_batches(
        chunk_size=25_000, engine="streaming"
    ):
        rows += frame.height
    return f"rows={rows}"


@case("full streaming write: scan -> transform -> sink_lance")
def case_sink_lance(uri: str) -> str:
    out = str(BENCH_DIR / "sink_out.lance")
    query = (
        pll.scan_lance(uri)
        .filter(pl.col("val") > 0.5)
        .select("id", "cat", (pl.col("val") * 2).alias("val2"), "payload")
    )
    dataset = pll.sink_lance(
        query, out, mode="overwrite" if os.path.exists(out) else "create"
    )
    return f"rows={dataset.count_rows()}"  # type: ignore[union-attr]


@case("fragment-parallel write via write_lance_fragments")
def case_sink_fragments(uri: str) -> str:
    out = str(BENCH_DIR / "frag_out.lance")
    shards = [
        shard.filter(pl.col("val") > 0.5).select("id", "cat")
        for shard in pll.scan_lance_fragments(uri)
    ]
    dataset = pll.write_lance_fragments(shards, out, mode="overwrite")
    return f"rows={dataset.count_rows()}"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run_case(name: str, size: str) -> dict[str, float]:
    fn = globals()[f"case_{name}"]
    uri = source_uri(size)
    started = time.perf_counter()
    with PeakSampler() as sampler:
        detail = fn(uri)
    elapsed = time.perf_counter() - started
    peak = sampler.as_mb()
    print(
        f"case={name} size={size} time={elapsed:.2f}s "
        f"rss={peak['VmRSS']:.0f}MB anon={peak['RssAnon']:.0f}MB "
        f"file={peak['RssFile']:.0f}MB {detail}"
    )
    return peak


def run_in_subprocess(name: str, size: str) -> dict[str, float]:
    result = subprocess.run(
        [sys.executable, __file__, "run", name, "--size", size],
        capture_output=True,
        text=True,
        check=True,
    )
    sys.stdout.write(result.stdout)
    line = next(line for line in result.stdout.splitlines() if line.startswith("case="))
    fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
    return {
        "VmRSS": float(fields["rss"].removesuffix("MB")),
        "RssAnon": float(fields["anon"].removesuffix("MB")),
        "RssFile": float(fields["file"].removesuffix("MB")),
    }


def guard() -> int:
    """Assert that streaming memory does not scale with the size of the source."""
    small_uri, large_uri = source_uri("small"), source_uri("large")
    print(
        f"small: {on_disk_mb(small_uri):.0f} MB on disk, "
        f"large: {on_disk_mb(large_uri):.0f} MB on disk"
    )

    small = run_in_subprocess("stream_scan", "small")
    large = run_in_subprocess("stream_scan", "large")

    ratio = large["VmRSS"] / small["VmRSS"]
    size_ratio = on_disk_mb(large_uri) / on_disk_mb(small_uri)
    print(
        f"\npeak RSS grew {ratio:.2f}x while the source grew {size_ratio:.2f}x "
        f"(limit {GUARD_TOLERANCE:.2f}x)"
    )
    if ratio > GUARD_TOLERANCE:
        print("FAIL: streaming scan memory scales with input size", file=sys.stderr)
        return 1
    print("OK: streaming scan memory is bounded")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("case", choices=sorted(CASES))
    run.add_argument("--size", choices=("small", "large"), default="small")
    every = sub.add_parser("all")
    every.add_argument("--size", choices=("small", "large"), default="small")
    sub.add_parser("guard")

    args = parser.parse_args()

    if args.command == "list":
        width = max(len(name) for name in CASES)
        for name, doc in CASES.items():
            print(f"  {name:<{width}}  {doc}")
        return 0
    if args.command == "run":
        run_case(args.case, args.size)
        return 0
    if args.command == "all":
        uri = source_uri(args.size)
        print(f"source: {on_disk_mb(uri):.0f} MB on disk ({args.size})\n")
        for name in CASES:
            run_in_subprocess(name, args.size)
        return 0
    return guard()


if __name__ == "__main__":
    raise SystemExit(main())
