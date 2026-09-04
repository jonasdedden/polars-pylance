"""Measure what predicate pushdown is worth, with and without scalar indices.

    uv run --group bench bench/polars_lance/pushdown.py \
        --out bench/polars_lance/plots/static

Runs each case twice, once with `predicate_pushdown=True` and once with it
off, and does that over an unindexed dataset and an indexed one. Every
measurement is a fresh subprocess, so peak RSS is that query's alone and no
page cache or Lance handle is shared between runs.

Unlike `run_matrix.py` this needs no particular machine: it compares two paths
through the same library on the same data, so the ratio is meaningful even
where the absolute numbers are not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROWS = 2_000_000
PAYLOAD_BYTES = 256

# `id` and `payload` are projected in every case, so the wide column is what a
# filter can avoid reading. Selectivity is in the label because it is the thing
# that decides whether pushdown pays.
CASES: dict[str, str] = {
    "selective": "val > 0.999  (0.1% of rows)",
    "membership": "id.is_in(100 ids)",
    "category": "cat == 'c3'  (12.5% of rows)",
    "substring": "text.str.contains('needle')  (0.02% of rows)",
    "computed": "val * 2 > 1.9995  (0.05% of rows)",
}
INDEXED = {"none": "no indices", "indexed": "scalar indices"}


def build(root: Path, *, indexed: bool) -> Path:
    """Write the dataset, optionally with a scalar index per filtered column."""
    import lance
    import numpy as np
    import pyarrow as pa

    uri = root / ("indexed.lance" if indexed else "plain.lance")
    if uri.exists():
        return uri
    rng = np.random.default_rng(0)
    text = np.array([f"row {i} filler" for i in range(ROWS)], dtype=object)
    text[rng.choice(ROWS, ROWS // 5000, replace=False)] += " needle"
    table = pa.table(
        {
            "id": pa.array(np.arange(ROWS, dtype=np.int64)),
            "val": pa.array(rng.random(ROWS)),
            "cat": pa.array(rng.choice([f"c{i}" for i in range(8)], ROWS)),
            "text": pa.array(text.tolist()),
            "payload": pa.array(["x" * PAYLOAD_BYTES] * ROWS),
        }
    )
    dataset = lance.write_dataset(table, str(uri), mode="overwrite")
    if indexed:
        dataset.create_scalar_index("val", "BTREE")
        dataset.create_scalar_index("id", "BTREE")
        dataset.create_scalar_index("cat", "BITMAP")
        dataset.create_scalar_index("text", "NGRAM")
    return uri


def predicate(case: str) -> Any:
    import polars as pl

    return {
        "selective": pl.col("val") > 0.999,
        "membership": pl.col("id").is_in(list(range(0, 1_000_000, 10_000))),
        "category": pl.col("cat") == "c3",
        "substring": pl.col("text").str.contains("needle"),
        "computed": pl.col("val") * 2 > 1.9995,
    }[case]


def measure(uri: str, case: str, *, pushdown: bool) -> dict[str, float]:
    """Run one query in this process and report its cost."""
    import polars_pylance as pll

    started = time.perf_counter()
    frame = (
        pll.scan_lance(uri, predicate_pushdown=pushdown)
        .filter(predicate(case))
        .select("id", "payload")
        .collect(engine="streaming")
    )
    seconds = time.perf_counter() - started
    peak_kib = next(
        int(line.split()[1])
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmHWM:")
    )
    return {"seconds": seconds, "peak_mib": peak_kib / 1024, "rows": frame.height}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/pll-pushdown"))  # noqa: S108
    parser.add_argument(
        "--out", type=Path, default=Path("bench/polars_lance/plots/static")
    )
    parser.add_argument(
        "--json", type=Path, default=Path("bench/polars_lance/pushdown.jsonl")
    )
    parser.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # The child branch: one query, one process, so VmHWM is this query's peak.
    if args.child:
        uri, case, pushdown = args.child
        print(json.dumps(measure(uri, case, pushdown=pushdown == "on")))
        return

    args.root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index_key in INDEXED:
        uri = build(args.root, indexed=index_key == "indexed")
        for case in CASES:
            for pushdown in ("on", "off"):
                proc = subprocess.run(
                    [sys.executable, __file__, "--child", str(uri), case, pushdown],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                row |= {"case": case, "pushdown": pushdown, "indices": index_key}
                results.append(row)
                print(
                    f"{index_key:8s} {case:11s} pushdown={pushdown:3s} "
                    f"{row['seconds']:6.3f}s {row['peak_mib']:7.1f} MiB "
                    f"{row['rows']:>8,} rows"
                )

    args.json.write_text("".join(json.dumps(r) + "\n" for r in results))
    plot(results, args.out)


def plot(results: list[dict[str, Any]], out: Path) -> None:
    """One grouped bar chart per index pass: pushdown on against pushdown off."""
    import plotly.graph_objects as go

    out.mkdir(parents=True, exist_ok=True)
    colour = {"on": "#2a9d8f", "off": "#d1495b"}
    for index_key, index_label in INDEXED.items():
        rows = [r for r in results if r["indices"] == index_key]
        figure = go.Figure()
        for pushdown in ("off", "on"):
            series = [
                next(r for r in rows if r["case"] == c and r["pushdown"] == pushdown)
                for c in CASES
            ]
            figure.add_bar(
                name=f"pushdown {pushdown}",
                x=[CASES[c] for c in CASES],
                y=[r["seconds"] for r in series],
                marker_color=colour[pushdown],
                text=[
                    f"{r['seconds']:.2f}s<br>{r['peak_mib']:.0f} MiB" for r in series
                ],
                textposition="outside",
            )
        figure.update_layout(
            title=f"Predicate pushdown, {index_label} "
            f"({ROWS // 1_000_000}M rows, {PAYLOAD_BYTES}-byte payload)",
            yaxis_title="runtime (s), lower is better",
            barmode="group",
            width=900,
            height=460,
            template="plotly_white",
            legend={"orientation": "h", "y": 1.1, "x": 0},
            margin={"t": 90},
        )
        path = out / f"pushdown-{index_key}.svg"
        figure.write_image(str(path))
        print("wrote", path)


if __name__ == "__main__":
    main()
