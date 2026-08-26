"""How much does lowering the *whole* predicate into Lance actually buy?

Compares the two scan paths of :func:`polars_pylance.scan_lance` on predicates
that differ only in whether Polars can lower them itself:

``provider``   the private dataset hook. Polars lowers the predicate to a
               PyArrow expression, or gives up and hands over nothing.
``io_plugin``  the public IO-plugin hook. The whole predicate arrives as an
               expression and ``polars_pylance._predicate`` lowers it to Lance
               SQL.
``engine``     ``io_plugin`` with ``predicate_pushdown=False``: nothing is
               pushed, so the difference to ``io_plugin`` is the translator's
               contribution and the difference to ``provider`` is the hook's.

Every measurement runs in its own process -- peak RSS is a per-process
high-water mark, so one materialising run would poison every later reading.

    uv run bench/pushdown.py gen [rows]        # write the dataset
    uv run bench/pushdown.py index             # add scalar indices to it
    uv run bench/pushdown.py run [--json out]  # the matrix, as markdown

``BENCH_ROOT`` (default ``/tmp``) decides where the dataset lives; it wants a
real filesystem, not a tmpfs.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa

ROWS = int(os.environ.get("BENCH_ROWS", 4_000_000))
PAYLOAD = 256
ROOT = Path(os.environ.get("BENCH_ROOT", "/tmp")) / "pushdown-bench"
# `BENCH_URI` points the matrix at an existing dataset -- e.g. an indexed copy
# alongside the plain one, so both can be measured without an index/unindex
# round trip between runs.
URI = os.environ.get("BENCH_URI") or str(ROOT / f"{ROWS // 1_000_000}m.lance")
WORDS = np.array(["alpha", "beta", "gamma", "delta"])

# A membership test big enough to be realistic and small enough to spell out.
# Polars renders an `is_in` haystack into the PyArrow predicate only up to
# `LIST_ITEM_LIMIT = 100` elements, so the two sizes fall on opposite sides of
# that cap and separate what each lowering can reach.
IN_LIST = [i * 9_973 % ROWS for i in range(200)]
IN_LIST_SMALL = IN_LIST[:100]
# Matches ~1 row in 10_000: the shape where skipping pages is everything.
NEEDLE = "row-0001234"


def _batches(rows: int, chunk: int = 100_000, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    schema = _schema()
    for start in range(0, rows, chunk):
        n = min(chunk, rows - start)
        ids = np.arange(start, start + n, dtype=np.int64)
        base = dt.datetime(2024, 1, 1)
        yield pa.record_batch(
            [
                pa.array(ids),
                pa.array(WORDS[ids % 4]),
                pa.array([f"row-{i:010d}-{WORDS[i % 4]}" for i in ids]),
                pa.array(rng.random(n)),
                pa.array([base + dt.timedelta(seconds=int(i)) for i in ids]),
                pa.FixedSizeBinaryArray.from_buffers(
                    pa.binary(PAYLOAD), n, [None, pa.py_buffer(rng.bytes(n * PAYLOAD))]
                ),
            ],
            schema=schema,
        )


def _schema() -> pa.Schema:
    fields: list[pa.Field[Any]] = [
        pa.field("id", pa.int64()),
        pa.field("cat", pa.string()),
        pa.field("text", pa.string()),
        pa.field("val", pa.float64()),
        pa.field("ts", pa.timestamp("us")),
        pa.field("payload", pa.binary(PAYLOAD)),
    ]
    return pa.schema(fields)


def generate() -> None:
    shutil.rmtree(URI, ignore_errors=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reader = pa.RecordBatchReader.from_batches(_schema(), _batches(ROWS))
    lance.write_dataset(
        reader,
        URI,
        schema=_schema(),
        max_rows_per_file=ROWS // 8,
        max_rows_per_group=25_000,
    )
    size = sum(
        os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(URI) for f in fs
    )
    print(
        f"wrote {ROWS:,} rows to {URI} "
        f"({size / 1024**3:.2f} GiB) in {time.perf_counter() - start:.1f}s"
    )


def index() -> None:
    dataset = lance.dataset(URI)
    for column, kind in (("id", "BTREE"), ("cat", "BITMAP"), ("text", "NGRAM")):
        start = time.perf_counter()
        dataset.create_scalar_index(column, kind)
        print(f"{kind} index on {column}: {time.perf_counter() - start:.1f}s")


def unindex() -> None:
    """Drop the scalar indices, to measure the same matrix without them."""
    dataset = lance.dataset(URI)
    for existing in dataset.list_indices():
        dataset.drop_index(existing["name"])
        print(f"dropped {existing['name']}")


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------

Case = Callable[[pl.LazyFrame], pl.LazyFrame]

CASES: dict[str, tuple[str, Case]] = {
    "full": (
        "full scan, no filter",
        lambda lf: lf.select(pl.col("payload").is_not_null().sum()),
    ),
    "proj": ("projection only", lambda lf: lf.select(pl.col("val").sum())),
    "head": ("head(10)", lambda lf: lf.head(10)),
    "topk": ("top-k sort", lambda lf: lf.sort("val", descending=True).head(10)),
    # both paths push this one: the difference is the hook, not the lowering
    "numeric": (
        "val > 0.999 (both push)",
        lambda lf: lf.filter(pl.col("val") > 0.999).select(pl.len()),
    ),
    "numeric_payload": (
        "val > 0.999, reads payload (both push)",
        lambda lf: lf.filter(pl.col("val") > 0.999).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    # only the io_plugin path can push these
    "prefix": (
        "cat.str.starts_with, 25% match",
        lambda lf: lf.filter(pl.col("cat").str.starts_with("be")).select(pl.len()),
    ),
    "prefix_payload": (
        "cat.str.starts_with, reads payload",
        lambda lf: lf.filter(pl.col("cat").str.starts_with("be")).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    "contains": (
        "text.str.contains, 1 in 10k",
        lambda lf: lf.filter(pl.col("text").str.contains(NEEDLE, literal=True)).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    "is_in": (
        "id.is_in(200 values)",
        lambda lf: lf.filter(pl.col("id").is_in(IN_LIST)).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    "is_in_small": (
        "id.is_in(100 values, under Polars' cap)",
        lambda lf: lf.filter(pl.col("id").is_in(IN_LIST_SMALL)).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    # Lowered to PyArrow only by pola-rs/polars#28996 (arithmetic) and #28994
    # (`eq_missing`, which before it emitted `==v` -- a SyntaxError for every
    # consumer of the hook). Both are pure provider-path cases: the visitor has
    # always handled them, so they separate the two upstream changes.
    "arith": (
        "val * 2 > 1.9995, reads payload",
        lambda lf: lf.filter((pl.col("val") * 2) > 1.9995).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    "eq_missing": (
        "cat.eq_missing('beta'), reads payload",
        lambda lf: lf.filter(pl.col("cat").eq_missing("beta")).select(
            pl.col("payload").is_not_null().sum()
        ),
    ),
    "temporal": (
        "ts.dt.year() == 2024 (matches all)",
        lambda lf: lf.filter(pl.col("ts").dt.year() == 2024).select(pl.len()),
    ),
    "mixed": (
        "untranslatable AND numeric",
        lambda lf: lf.filter(
            pl.col("text").str.contains(NEEDLE, literal=True) & (pl.col("val") > 0.5)
        ).select(pl.col("payload").is_not_null().sum()),
    ),
}

# Which scan paths to put in the matrix. `bench/_variants.py` documents them all;
# the default three are the ones a user can actually choose today.
IMPLS = tuple(os.environ.get("BENCH_IMPLS", "provider,io_plugin,engine").split(","))
REPEATS = int(os.environ.get("BENCH_REPEATS", 3))


def build(impl: str) -> pl.LazyFrame:
    from _variants import build as build_variant

    return build_variant(impl, URI)


def measure(impl: str, case: str) -> dict[str, Any]:
    """Run one case, reporting time, peak RSS and the rows Lance handed over."""
    from polars_pylance import _scan

    rows = [0]
    original = _scan.LanceScanSpec.iter_frames
    filters: list[str] = []

    def spy(self: Any, dataset: Any, **kwargs: Any) -> Any:
        if kwargs.get("filter") is not None:
            filters.append(str(kwargs["filter"]))
        for frame in original(self, dataset, **kwargs):
            rows[0] += frame.height
            yield frame

    _scan.LanceScanSpec.iter_frames = spy  # type: ignore[method-assign]

    _, plan = CASES[case]
    lazy = build(impl)
    best = float("inf")
    for _ in range(REPEATS):
        rows[0] = 0
        start = time.perf_counter()
        result = plan(lazy).collect(engine="streaming")
        best = min(best, time.perf_counter() - start)
    return {
        # `build` names which Polars this ran against, so results files from
        # different builds can be plotted side by side.
        "build": os.environ.get("BENCH_BUILD", "unknown"),
        "polars": pl.__version__,
        "impl": impl,
        "case": case,
        "seconds": best,
        "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "rows_from_lance": rows[0],
        "pushed": filters[0] if filters else None,
        "result": str(result.row(0)) if result.height else "",
    }


def run(out: Path | None) -> None:
    records: list[dict[str, Any]] = []
    for case in CASES:
        for impl in IMPLS:
            proc = subprocess.run(
                [sys.executable, __file__, "case", impl, case],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(proc.stderr[-2000:], file=sys.stderr)
                records.append({"impl": impl, "case": case, "error": "failed"})
                continue
            records.append(json.loads(proc.stdout.splitlines()[-1]))
    _report(records)
    _consistency(records)
    if out is not None:
        out.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _report(records: list[dict[str, Any]]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case"], {})[record["impl"]] = record

    build_label = os.environ.get("BENCH_BUILD", "unknown")
    print(
        f"\n{ROWS:,} rows, {PAYLOAD}-byte payload, {URI}\n"
        f"polars {pl.__version__} ({build_label})\n"
    )
    print("| case | | " + " | ".join(IMPLS) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in IMPLS) + " |")
    for case, (label, _) in CASES.items():
        got = by_case.get(case, {})
        cells = []
        for impl in IMPLS:
            record = got.get(impl, {})
            if "seconds" not in record:
                cells.append("failed")
                continue
            cells.append(
                f"{record['seconds']:.3f} s / {record['peak_mib']:.0f} MiB / "
                f"{record['rows_from_lance']:,} rows"
            )
        print(f"| `{case}` | {label} | " + " | ".join(cells) + " |")
    print("\nFilter reaching Lance:\n")
    for case in CASES:
        for impl in IMPLS:
            pushed = by_case.get(case, {}).get(impl, {}).get("pushed")
            if pushed:
                print(f"- `{case}` / {impl}: `{pushed}`")


def _consistency(records: list[dict[str, Any]]) -> None:
    for case in CASES:
        results = {
            record["impl"]: record.get("result")
            for record in records
            if record["case"] == case
        }
        if len(set(results.values())) > 1:
            print(f"!! {case}: implementations disagree: {results}", file=sys.stderr)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "gen":
        if len(sys.argv) > 2:
            ROWS = int(sys.argv[2])
            URI = str(ROOT / f"{ROWS // 1_000_000}m.lance")
        generate()
    elif command == "index":
        index()
    elif command == "unindex":
        unindex()
    elif command == "case":
        print(json.dumps(measure(sys.argv[2], sys.argv[3])))
    else:
        target = None
        if "--json" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--json") + 1])
        run(target)
