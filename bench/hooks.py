"""What the two hooks cost per *query*, as opposed to per row.

``bench/pushdown.py`` measures scans big enough that the filter dominates. This
one measures the other end: a dataset small enough that everything is hook
overhead, so the fixed cost of each mechanism is visible.

Three things separate the dataset-provider hook from ``register_io_source``
regardless of any predicate:

``resolve``   how often Polars asks the source to describe itself. The provider
              hook resolves the scan while building the IR and caches the result
              against a version key, so a second collect of the same LazyFrame
              can skip it; an IO plugin is called afresh every collect.
``plan``      how large the query serializes, which is what Polars Cloud ships
              to a worker. The provider sends a dataset object, the IO plugin a
              cloudpickled closure.
``calls``     how many times Lance is opened and the schema recomputed.

    uv run bench/hooks.py            # markdown, ~20 s
    uv run bench/hooks.py --json out.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
from _variants import build as build_variant

ROWS = 10_000
REPEATS = int(os.environ.get("BENCH_REPEATS", 200))
IMPLS = tuple(os.environ.get("BENCH_IMPLS", "provider,io_plugin").split(","))


def dataset(directory: Path) -> str:
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "id": pa.array(np.arange(ROWS, dtype=np.int64)),
            "val": pa.array(rng.random(ROWS)),
        }
    )
    uri = str(directory / "hooks.lance")
    lance.write_dataset(table, uri)
    return uri


def _counters() -> dict[str, int]:
    """Count the calls each hook makes into Lance, per collect."""
    from polars_pylance import _scan

    counts = {"open": 0, "schema": 0, "resolve": 0, "reresolve": 0}

    open_ = _scan.LanceScanSpec.open
    schema = _scan.LanceScanSpec.arrow_schema
    resolve = _scan.LanceDatasetProvider.to_dataset_scan

    def counting_open(self: Any) -> Any:
        counts["open"] += 1
        return open_(self)

    def counting_schema(self: Any, dataset: Any = None) -> Any:
        counts["schema"] += 1
        return schema(self, dataset)

    def counting_resolve(self: Any, **kwargs: Any) -> Any:
        counts["resolve"] += 1
        out = resolve(self, **kwargs)
        # `None` means "the version key you hold is still current", i.e. Polars
        # reused its cached expansion. Anything else is a full re-resolution.
        counts["reresolve"] += out is not None
        return out

    _scan.LanceScanSpec.open = counting_open  # type: ignore[method-assign]
    _scan.LanceScanSpec.arrow_schema = counting_schema  # type: ignore[method-assign]
    _scan.LanceDatasetProvider.to_dataset_scan = counting_resolve  # type: ignore[method-assign]
    return counts


# ---------------------------------------------------------------------------
# does this Polars believe a source that says it applied the predicate?
# ---------------------------------------------------------------------------

PROBE_SCHEMA = pa.schema([pa.field("a", pa.int64())])
PROBE_FRAME = pl.DataFrame({"a": list(range(10))})
PROBE_PREDICATE = pl.col("a") > 4  # keeps 5 of the 10 rows


def _honours_provider_flag() -> bool:
    """Whether a provider-resolved scan's ``predicate_applied`` is believed.

    The scan below hands back every row while claiming the predicate is applied.
    If Polars believes it, ten rows survive; if it filters above the scan
    anyway, five do -- and then the flag is decoration, and any filter pushed
    into Lance is an IO hint rather than the source of truth.
    """
    from polars._plr import PyLazyFrame
    from polars._utils.wrap import wrap_ldf

    class Provider:
        def schema(self) -> pa.Schema:
            return PROBE_SCHEMA

        def to_dataset_scan(self, **_: Any) -> tuple[pl.LazyFrame, str]:
            def impl(*_a: Any, **_k: Any) -> tuple[Any, bool]:
                return iter([PROBE_FRAME]), True

            lf = pl.LazyFrame._scan_python_function(
                PROBE_SCHEMA, impl, pyarrow=True, is_pure=True
            )
            return lf, "v1"

    lazy = wrap_ldf(PyLazyFrame.new_from_dataset_object(Provider()))
    kept = lazy.filter(PROBE_PREDICATE).select(pl.len()).collect(engine="streaming")
    return kept.item() == PROBE_FRAME.height


def _honours_io_plugin_flag() -> bool:
    """The same question for an IO plugin, which answers it the other way.

    `register_io_source` hardcodes the flag to True, so a source that pushes a
    *relaxed* filter has to re-apply the predicate itself, per batch, in Python.
    Reporting False instead -- which needs `_scan_python_function` directly --
    hands that second evaluation to the streaming engine. This checks the engine
    actually takes it.
    """

    def wrap(*_a: Any, **_k: Any) -> tuple[Any, bool]:
        return iter([PROBE_FRAME]), False

    lazy = pl.LazyFrame._scan_python_function(
        {"a": pl.Int64}, wrap, pyarrow=False, is_pure=True
    )
    kept = lazy.filter(PROBE_PREDICATE).select(pl.len()).collect(engine="streaming")
    return kept.item() == PROBE_FRAME.filter(PROBE_PREDICATE).height


def measure(impl: str, uri: str) -> dict[str, Any]:
    counts = _counters()

    # One LazyFrame collected many times: the shape where the provider hook's
    # resolved-scan cache can pay off, and the one a served query looks like.
    lazy = build_variant(impl, uri).filter(pl.col("val") > 0.5).select(pl.len())
    lazy.collect(engine="streaming")  # warm: first collect resolves and opens
    warm = dict(counts)

    start = time.perf_counter()
    for _ in range(REPEATS):
        lazy.collect(engine="streaming")
    per_query = (time.perf_counter() - start) / REPEATS

    # A fresh LazyFrame per collect: no cache can help, so this is the floor a
    # server handing out one plan per request actually pays.
    start = time.perf_counter()
    for _ in range(REPEATS):
        build_variant(impl, uri).filter(pl.col("val") > 0.5).select(pl.len()).collect(
            engine="streaming"
        )
    per_fresh_query = (time.perf_counter() - start) / REPEATS

    plan = build_variant(impl, uri).filter(pl.col("val") > 0.5).serialize()
    return {
        "build": os.environ.get("BENCH_BUILD", "unknown"),
        "polars": pl.__version__,
        "impl": impl,
        "ms_per_collect": per_query * 1000,
        "ms_per_fresh_collect": per_fresh_query * 1000,
        "plan_bytes": len(plan),
        "opens_first_collect": warm["open"],
        "schemas_first_collect": warm["schema"],
        "resolves_first_collect": warm["resolve"],
        "opens_per_repeat": (counts["open"] - warm["open"]) / REPEATS,
        "schemas_per_repeat": (counts["schema"] - warm["schema"]) / REPEATS,
        "resolves_per_repeat": (counts["resolve"] - warm["resolve"]) / REPEATS,
        "reresolves_per_repeat": (counts["reresolve"] - warm["reresolve"]) / REPEATS,
    }


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None
    directory = Path(
        tempfile.mkdtemp(prefix="hooks-", dir=os.environ.get("BENCH_ROOT"))
    )
    records = []
    try:
        uri = dataset(directory)
        for impl in IMPLS:
            # Each impl in its own process: the variants monkey-patch the
            # package, and `_counters` would otherwise stack.
            proc = __import__("subprocess").run(
                [sys.executable, __file__, "one", impl, uri],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(proc.stderr[-2000:], file=sys.stderr)
                continue
            records.append(json.loads(proc.stdout.splitlines()[-1]))
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    print(f"\n{ROWS:,} rows, {REPEATS} collects, polars {pl.__version__}\n")
    print(
        "predicate_applied believed on the provider path: "
        f"**{_honours_provider_flag()}**; the engine re-applies the predicate "
        f"for an IO plugin that reports it unapplied: "
        f"**{_honours_io_plugin_flag()}**\n"
    )
    columns = (
        "path",
        "ms/collect (kept plan)",
        "ms/collect (fresh plan)",
        "plan",
        "resolves/collect",
        "re-resolves/collect",
        "Lance opens/collect",
    )
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for record in records:
        print(
            f"| `{record['impl']}` | {record['ms_per_collect']:.2f} | "
            f"{record['ms_per_fresh_collect']:.2f} | {record['plan_bytes']:,} B | "
            f"{record['resolves_per_repeat']:.2f} | "
            f"{record['reresolves_per_repeat']:.2f} | "
            f"{record['opens_per_repeat']:.2f} |"
        )
    if out is not None:
        out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        print(json.dumps(measure(sys.argv[2], sys.argv[3])))
    else:
        main()
