"""Which predicates survive each lowering, side by side.

Not a timing benchmark: it answers the prior question of *what reaches Lance at
all*. For every predicate it records what Polars' own PyArrow lowering produced
(what the ``provider`` path would push) next to what
``polars_pylance._predicate`` produces (what the ``io_plugin`` path pushes), and
checks both against the dataset so a wrong answer cannot pass as coverage.

    uv run bench/coverage.py                 # markdown table
    uv run bench/coverage.py --verbose       # plus the generated filters
    uv run bench/coverage.py --json out.jsonl  # for bench/plot_pushdown.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa

from polars_pylance import _scan, scan_lance
from polars_pylance._predicate import to_lance_filter

ROWS = 2_000
WORDS = np.array(["alpha", "beta", "gamma", "delta"])

CASES: list[tuple[str, pl.Expr]] = [
    ("comparison", pl.col("id") > 7),
    ("string equality", pl.col("cat") == "beta"),
    ("boolean column", pl.col("flag")),
    ("and / or / not", ~((pl.col("id") > 5) & (pl.col("val") < 0.5)) | pl.col("flag")),
    (
        "deep nesting (4 levels)",
        (pl.col("id") > 5)
        & ((pl.col("val") < 0.5) | ((pl.col("cat") == "beta") & (pl.col("id") < 900))),
    ),
    ("is_null", pl.col("opt").is_null()),
    ("is_between", pl.col("id").is_between(10, 20)),
    ("column vs column", pl.col("id") > pl.col("opt")),
    ("date literal", pl.col("day") == dt.date(2024, 2, 1)),
    ("datetime literal", pl.col("ts") > dt.datetime(2024, 1, 5)),
    ("eq_missing", pl.col("opt").eq_missing(3)),
    ("xor", (pl.col("id") > 5) ^ pl.col("flag")),
    ("is_in (list)", pl.col("id").is_in([1, 2, 3])),
    ("is_in (Series)", pl.col("id").is_in(pl.Series([1, 2, 3]))),
    ("is_in (500 values)", pl.col("id").is_in(list(range(500)))),
    ("is_nan", pl.col("val").is_nan()),
    ("str.starts_with", pl.col("cat").str.starts_with("be")),
    ("str.ends_with", pl.col("cat").str.ends_with("ta")),
    ("str.contains (literal)", pl.col("text").str.contains("gamma", literal=True)),
    ("str.contains (regex)", pl.col("text").str.contains(r"row-\d+-al")),
    ("str.len_chars", pl.col("cat").str.len_chars() > 4),
    ("str.to_lowercase", pl.col("cat").str.to_lowercase() == "beta"),
    ("str.slice", pl.col("cat").str.slice(0, 2) == "be"),
    ("str.contains_any", pl.col("text").str.contains_any(["alpha", "beta"])),
    ("arithmetic", (pl.col("id") + 1) * 2 > 100),
    ("modulo", (pl.col("id") % 2) == 0),
    ("true division", (pl.col("id") / 2) > 10.5),
    ("floor division", (pl.col("id") // 2) == 5),
    ("abs", pl.col("val").abs() > 0.5),
    ("round", pl.col("val").round(2) == 0.5),
    ("dt.year", pl.col("ts").dt.year() == 2024),
    ("dt.hour", pl.col("ts").dt.hour() == 3),
    ("dt.date", pl.col("ts").dt.date() == dt.date(2024, 1, 2)),
    ("dt.weekday", pl.col("ts").dt.weekday() == 1),
    ("cast", pl.col("id").cast(pl.Int32) > 5),
    ("cast (non-strict)", pl.col("id").cast(pl.Int32, strict=False) > 5),
    ("list.contains", pl.col("tags").list.contains(3)),
    ("list.len", pl.col("tags").list.len() == 2),
    ("struct.field", pl.col("meta").struct.field("k") > 5),
    ("fill_null", pl.col("opt").fill_null(0) > 5),
    ("all_horizontal", pl.all_horizontal(pl.col("id") > 5, pl.col("val") < 0.5)),
    ("when/then", pl.when(pl.col("id") > 5).then(True).otherwise(False)),
    ("hash", pl.col("id").hash() % 2 == 0),
    ("aggregate", pl.col("val") > pl.col("val").mean()),
    (
        "AND: one side unlowerable",
        pl.col("cat").str.slice(0, 2).eq("be") & (pl.col("id") > 5),
    ),
    (
        "OR: one side unlowerable",
        pl.col("cat").str.slice(0, 2).eq("be") | (pl.col("id") > 5),
    ),
    (
        "NOT: unlowerable inside",
        ~(pl.col("cat").str.slice(0, 2).eq("be") & (pl.col("id") > 5)),
    ),
]


def build(directory: Path) -> str:
    base = dt.datetime(2024, 1, 1)
    table = pa.table(
        {
            "id": pa.array(np.arange(ROWS, dtype=np.int64)),
            "cat": pa.array(WORDS[np.arange(ROWS) % 4]),
            "text": pa.array([f"row-{i:05d}-{WORDS[i % 4]}" for i in range(ROWS)]),
            "val": pa.array(np.random.default_rng(0).random(ROWS)),
            "flag": pa.array(np.arange(ROWS) % 3 == 0),
            "opt": pa.array(
                [None if i % 7 == 0 else i for i in range(ROWS)], pa.int64()
            ),
            "ts": pa.array([base + dt.timedelta(hours=i) for i in range(ROWS)]),
            "day": pa.array(
                [dt.date(2024, 1, 1) + dt.timedelta(days=i % 365) for i in range(ROWS)]
            ),
            "tags": pa.array(
                [[i % 5, i % 3] for i in range(ROWS)], pa.list_(pa.int64())
            ),
            "meta": pa.array(
                [{"k": i % 10, "s": WORDS[i % 4]} for i in range(ROWS)],
                pa.struct([("k", pa.int64()), ("s", pa.string())]),
            ),
        }
    )
    uri = str(directory / "coverage.lance")
    lance.write_dataset(table, uri, max_rows_per_file=500)
    return uri


def pyarrow_lowering(uri: str, predicate: pl.Expr) -> str | None:
    """What Polars itself hands the dataset provider for `predicate`."""
    warnings.simplefilter("ignore")
    seen: dict[str, Any] = {}
    original = _scan.LanceDatasetProvider.to_dataset_scan

    def spy(self: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return original(self, **kwargs)

    _scan.LanceDatasetProvider.to_dataset_scan = spy  # type: ignore[method-assign]
    try:
        scan_lance(uri).filter(predicate).select(pl.len()).collect(engine="streaming")
    finally:
        _scan.LanceDatasetProvider.to_dataset_scan = original  # type: ignore[method-assign]
    return seen.get("pyarrow_predicate")


def _evaluation_error(pyarrow_predicate: str) -> str | None:
    """Whether the string Polars generated can be evaluated at all."""
    try:
        eval(pyarrow_predicate, _scan._pyarrow_eval_namespace())
    except Exception as exc:
        return type(exc).__name__
    return None


def _rows_kept(dataset: lance.LanceDataset, filter: Any) -> set[int] | str:
    """The `id`s Lance keeps for `filter`, or why it could not be used."""
    try:
        table = dataset.scanner(columns=["id"], filter=filter).to_table()
    except Exception as exc:
        return type(exc).__name__
    return set(table["id"].to_pylist())


def _verdict(kept: set[int], pushed: set[int] | str, partial: bool) -> str:
    """Grade a lowering against the rows the predicate itself keeps.

    A lowering is allowed to be a *superset* -- the caller leaves the predicate
    with the engine -- so dropping a row is the only real failure. A lowering
    that claimed to cover every column but is not exact is worth flagging too,
    since that is the case a caller might be tempted to trust.
    """
    if isinstance(pushed, str):
        return f"!! Lance {pushed}"
    if not kept <= pushed:
        return "!! DROPS ROWS"
    if kept != pushed:
        return "partial" if partial else "!! NOT EXACT"
    return "yes"


def coverage(uri: str, verbose: bool, out: Path | None = None) -> None:
    dataset = lance.dataset(uri)
    truth = pl.from_arrow(dataset.to_table())
    assert isinstance(truth, pl.DataFrame)

    print(f"polars {pl.__version__} ({os.environ.get('BENCH_BUILD', 'unknown')})\n")
    print("| predicate | Polars -> PyArrow | visitor -> Lance SQL |")
    print("| --- | --- | --- |")
    totals = {"pyarrow": 0, "sql": 0}
    details: list[str] = []
    records: list[dict[str, Any]] = []
    for label, predicate in CASES:
        pa_pred = pyarrow_lowering(uri, predicate)
        lowered = to_lance_filter(predicate)
        kept = set(truth.filter(predicate)["id"].to_list())

        # Both lowerings are graded the same way, and both against the data:
        # "does Lance, given this filter, hand back at least the rows the
        # predicate keeps". Polars lowers only whole conjuncts, so a lowering
        # that misses a column is a superset by construction.
        columns = set(predicate.meta.root_names())
        if pa_pred is None:
            left = "-"
        elif (broken := _evaluation_error(pa_pred)) is not None:
            # Polars produced something, but not something PyArrow accepts.
            left = f"!! {broken}"
        else:
            expression = eval(pa_pred, _scan._pyarrow_eval_namespace())
            left = _verdict(
                kept,
                _rows_kept(dataset, expression),
                partial=not all(f"'{c}'" in pa_pred for c in columns),
            )
            totals["pyarrow"] += 1

        if lowered is None:
            right = "-"
        else:
            right = _verdict(
                kept, _rows_kept(dataset, lowered.sql), partial=not lowered.exact
            )
            totals["sql"] += 1
            details.append(f"- `{label}`: `{lowered.sql[:120]}`")
        print(f"| {label} | {left} | {right} |")
        records.append(
            {
                "build": os.environ.get("BENCH_BUILD", "unknown"),
                "polars": pl.__version__,
                "case": label,
                "pyarrow": left,
                "pyarrow_expr": pa_pred,
                "visitor": right,
                "sql": None if lowered is None else lowered.sql,
            }
        )
    print(
        f"\n{totals['pyarrow']}/{len(CASES)} reach Lance through the provider path, "
        f"{totals['sql']}/{len(CASES)} through the visitor."
    )
    if verbose:
        print("\nLowered filters:\n")
        print("\n".join(details))
    if out is not None:
        out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    target = (
        Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None
    )
    directory = Path(tempfile.mkdtemp(prefix="coverage-"))
    try:
        coverage(build(directory), "--verbose" in sys.argv, target)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
