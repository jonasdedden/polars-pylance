"""Render the predicate-pushdown results as Plotly pages.

    uv run --group bench bench/plot_pushdown.py --static

Reads the files `bench/pushdown.py` and `bench/coverage.py` write and produces
four views:

``pushdown.html`` / ``pushdown-indexed.html``
    Per case, what each scan path handed Polars and how long it took. The two
    metrics get their own panel rather than a second y-axis: they are different
    quantities, and one axis each is the only way the marks stay comparable.
``pushdown-upstream.html``
    The provider path on a Polars that passes the whole predicate, filter
    pushed vs not -- both from the same build, so only the two bars compare.
``pushdown-coverage.html``
    Which of 47 predicate shapes each lowering reaches Lance with.

Dot plots rather than bars because every measure here spans three or more
decades, and a bar on a log axis has no honest baseline. Values are printed
next to each mark, so the chart reads without hovering or colour vision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Categorical slots 1-3 of the reference palette, in order. Validated all-pairs
# in light mode: worst CVD dE 9.2, worst normal-vision dE 24.0.
COLOUR = {
    "provider": "#2a78d6",
    "io_plugin": "#eb6834",
    "engine": "#1baf7a",
}
LABEL = {
    "provider": "provider (Polars → PyArrow)",
    "io_plugin": "io_plugin (visitor → Lance SQL)",
    "engine": "no pushdown",
}
# On a Polars carrying the `serialized_predicate` patch the provider path lowers
# the predicate itself, so it is no longer the PyArrow subset that is measured.
PATCHED_LABEL = dict(LABEL, provider="provider (visitor → Lance SQL)")

# Drawn back to front, so the series that usually sits alone goes last.
ORDER = ["engine", "provider", "io_plugin"]
OFFSET = {"engine": 0.24, "provider": 0.0, "io_plugin": -0.24}

# (draw order, colour per key, vertical offset per key). `dumbbell` takes one so
# a figure can compare something other than the three shipped paths.
Series = tuple[list[str], dict[str, str], dict[str, float]]

SURFACE = "#fcfcfb"
INK = "#52514e"
RULE = "#d9d8d3"
STATUS = {"yes": "#0ca30c", "partial": "#fab219", "broken": "#d03b3b"}
ABSENT = "#8a8984"

CASES: dict[str, str] = {
    "full": "full scan, no filter",
    "proj": "projection only",
    "head": "head(10)",
    "topk": "top-k sort",
    "numeric": "val > 0.999",
    "numeric_payload": "val > 0.999, reads payload",
    "prefix": "cat.str.starts_with",
    "prefix_payload": "cat.str.starts_with, reads payload",
    "contains": "text.str.contains, 1 in 10k",
    "is_in": "id.is_in(200 values)",
    "is_in_small": "id.is_in(100 values)",
    "arith": "val * 2 > 1.9995",
    "eq_missing": "cat.eq_missing('beta')",
    "temporal": "ts.dt.year() == 2024",
    "mixed": "unlowerable AND numeric",
}
# Everything below the line is a predicate Polars cannot lower to PyArrow.
FIRST_VISITOR_ONLY = "prefix"


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _log_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """1-2-5 ticks across the data's decades, labelled with the full number.

    Plotly labels log minor ticks with the mantissa alone, so the tick at 20
    reads "2" next to the one reading "10". Supplying them avoids that.
    """
    finite = [v for v in values if v and v > 0]
    lo, hi = min(finite), max(finite)
    ticks: list[float] = []
    decade = math.floor(math.log10(lo))
    while decade <= math.ceil(math.log10(hi)):
        ticks.append(10.0**decade)
        decade += 1
    return ticks, [_compact(t) for t in ticks]


def _seconds(value: float) -> str:
    return f"{value:.3f}" if value < 1 else f"{value:.2f}"


def _compact(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1e6:.3g}M"
    if value >= 1_000:
        return f"{value / 1e3:.3g}k"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def dumbbell(
    rows: list[dict[str, Any]],
    *,
    cases: dict[str, str],
    key: str,
    title: str,
    label: str,
    value_fmt: Any = _compact,
    labels: dict[str, str] | None = None,
    series: Series | None = None,
) -> tuple[list[go.Scatter], list[float], list[str]]:
    """One row per case, one dot per implementation, on a log axis.

    `series` names which implementations to draw and how; the default is the
    three a user can choose between. A figure comparing something else -- two
    Polars builds, or the variants in `bench/_variants.py` -- passes its own.
    """
    labels = labels or LABEL
    order, colour, offset = series or (ORDER, COLOUR, OFFSET)
    traces: list[go.Scatter] = []
    names = list(cases)
    values = [r[key] for r in rows if r.get(key)]

    # A recessive rule spanning each row, so the gap is legible before the dots
    # are read individually.
    for i, case in enumerate(names):
        spread = [r[key] for r in rows if r["case"] == case and r.get(key)]
        if len(spread) < 2:
            continue
        traces.append(
            go.Scatter(
                x=[min(spread), max(spread)],
                y=[i, i],
                mode="lines",
                line={"color": RULE, "width": 6},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for impl in order:
        pts = [
            (names.index(r["case"]), r[key], r)
            for r in rows
            if r["impl"] == impl and r["case"] in cases and r.get(key) is not None
        ]
        if not pts:
            continue
        traces.append(
            go.Scatter(
                x=[v for _, v, _ in pts],
                y=[i + offset[impl] for i, _, _ in pts],
                mode="markers+text",
                name=labels[impl],
                legendgroup=impl,
                marker={
                    "size": 11,
                    "color": colour[impl],
                    "line": {"color": SURFACE, "width": 2},
                },
                text=[value_fmt(v) for _, v, _ in pts],
                textposition="middle right",
                textfont={"size": 10, "color": INK},
                cliponaxis=False,
                customdata=[
                    [labels[impl], r.get("pushed") or "nothing"] for _, _, r in pts
                ],
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b><br>{label} %{{x:,}}"
                    "<br>pushed: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    ticks, ticktext = _log_ticks(values)
    return traces, ticks, ticktext


def scan_figure(
    rows: list[dict[str, Any]], subtitle: str, labels: dict[str, str] | None = None
) -> go.Figure:
    cases = {k: v for k, v in CASES.items() if any(r["case"] == k for r in rows)}
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=("rows handed to Polars", "wall time"),
    )

    for col, (key, label, fmt) in enumerate(
        (
            ("rows_from_lance", "rows — fewer is better", _compact),
            ("seconds", "seconds — lower is better", _seconds),
        ),
        start=1,
    ):
        traces, ticks, ticktext = dumbbell(
            rows,
            cases=cases,
            key=key,
            title=label,
            label=label,
            value_fmt=fmt,
            labels=labels,
        )
        for trace in traces:
            if col == 2 and trace.showlegend is not False:
                trace.showlegend = False
            fig.add_trace(trace, row=1, col=col)
        fig.update_xaxes(
            type="log",
            tickvals=ticks,
            ticktext=ticktext,
            tickangle=0,
            title_text=label,
            gridcolor=RULE,
            row=1,
            col=col,
            # room for the value printed beside the right-most dot
            range=[math.log10(min(ticks) / 2.2), math.log10(max(ticks) * 4.5)],
        )

    order = list(cases)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(order))),
        ticktext=[cases[c] for c in order],
        autorange="reversed",
        gridcolor=RULE,
    )
    if FIRST_VISITOR_ONLY in order:
        for col in (1, 2):
            fig.add_hline(
                y=order.index(FIRST_VISITOR_ONLY) - 0.5,
                line={"color": INK, "width": 1, "dash": "dot"},
                row=1,
                col=col,
            )

    fig.update_layout(
        template="plotly_white",
        height=170 + 46 * len(order),
        margin={"l": 260, "r": 40, "t": 115, "b": 110},
        title={
            "text": "What reaches Lance, and what it costs<br>"
            f"<span style='font-size:12px;color:{INK}'>{subtitle}<br>"
            "Below the dotted rule: predicates Polars cannot lower to PyArrow."
            "</span>",
            "x": 0.01,
            "y": 0.97,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.10,
            "xanchor": "left",
            "x": 0,
        },
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
    )
    return fig


def upstream_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """The provider path on a patched Polars: filter pushed vs not."""
    queries = list(dict.fromkeys(r["query"] for r in rows))
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=("rows handed to Polars", "wall time"),
    )
    series = {
        False: ("nothing pushed", ABSENT),
        True: ("lowered to Lance SQL", COLOUR["provider"]),
    }

    for col, (key, label, fmt) in enumerate(
        (
            ("rows_from_lance", "rows — fewer is better", _compact),
            ("seconds", "seconds — lower is better", _seconds),
        ),
        start=1,
    ):
        for i, query in enumerate(queries):
            spread = [r[key] for r in rows if r["query"] == query]
            fig.add_trace(
                go.Scatter(
                    x=[min(spread), max(spread)],
                    y=[i, i],
                    mode="lines",
                    line={"color": RULE, "width": 6},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        for pushed, (name, colour) in series.items():
            pts = [
                (queries.index(r["query"]), r[key])
                for r in rows
                if r["pushdown"] is pushed
            ]
            fig.add_trace(
                go.Scatter(
                    x=[v for _, v in pts],
                    y=[i for i, _ in pts],
                    mode="markers+text",
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    marker={
                        "size": 11,
                        "color": colour,
                        "line": {"color": SURFACE, "width": 2},
                    },
                    text=[fmt(v) for _, v in pts],
                    textposition="middle right",
                    textfont={"size": 10, "color": INK},
                    cliponaxis=False,
                    hovertemplate=f"<b>{name}</b><br>{label} %{{x:,}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        ticks, ticktext = _log_ticks([r[key] for r in rows])
        fig.update_xaxes(
            type="log",
            tickvals=ticks,
            ticktext=ticktext,
            title_text=label,
            gridcolor=RULE,
            range=[math.log10(min(ticks) / 1.6), math.log10(max(ticks) * 3.2)],
            row=1,
            col=col,
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(queries))),
        ticktext=queries,
        autorange="reversed",
        gridcolor=RULE,
    )
    fig.update_layout(
        template="plotly_white",
        height=220 + 60 * len(queries),
        margin={"l": 290, "r": 60, "t": 150, "b": 110},
        title={
            "text": "The provider path, on a Polars that passes the whole "
            f"predicate<br><span style='font-size:12px;color:{INK}'>"
            "1M rows, 128-byte payload, impl='provider'. Both marks come from "
            "the same build,<br>so they compare to each other and not to the "
            "other pages.</span>",
            "x": 0.01,
            "y": 0.95,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
        },
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
    )
    return fig


def coverage_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Which lowering reaches Lance, one row per predicate shape."""
    shape = {"yes": "circle", "partial": "diamond", "broken": "x", "-": "circle-open"}
    colour = {
        "yes": STATUS["yes"],
        "partial": STATUS["partial"],
        "broken": STATUS["broken"],
        "-": ABSENT,
    }
    name = {
        "yes": "reaches Lance",
        "partial": "partly reaches Lance",
        "broken": "lowered, but not evaluable",
        "-": "left to the engine",
    }

    def state(value: str) -> str:
        if value.startswith("!!"):
            return "broken"
        return value if value in shape else "-"

    names = [r["case"] for r in rows]
    fig = go.Figure()

    for column, (key, x) in enumerate((("pyarrow", 0), ("visitor", 1))):
        for status in ("yes", "partial", "broken", "-"):
            pts = [i for i, r in enumerate(rows) if state(r[key]) == status]
            if not pts:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[x] * len(pts),
                    y=pts,
                    mode="markers",
                    name=name[status],
                    legendgroup=status,
                    showlegend=column == 0,
                    marker={
                        "size": 12,
                        "symbol": shape[status],
                        "color": colour[status],
                        "line": {"color": colour[status], "width": 2},
                    },
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[f"{names[i]} — {key}: {name[status]}" for i in pts],
                )
            )

    counts = {
        key: sum(1 for r in rows if state(r[key]) in ("yes", "partial"))
        for key in ("pyarrow", "visitor")
    }
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=[
            f"Polars → PyArrow<br><b>{counts['pyarrow']} of {len(rows)}</b>",
            f"visitor → Lance SQL<br><b>{counts['visitor']} of {len(rows)}</b>",
        ],
        range=[-0.6, 1.6],
        side="top",
        gridcolor=SURFACE,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(names))),
        ticktext=names,
        autorange="reversed",
        gridcolor=RULE,
    )
    fig.update_layout(
        template="plotly_white",
        height=190 + 21 * len(names),
        width=780,
        margin={"l": 250, "r": 40, "t": 160, "b": 90},
        title={
            "text": "Predicate coverage: what each lowering can express<br>"
            f"<span style='font-size:12px;color:{INK}'>47 predicate shapes, "
            "each checked against the data; see bench/coverage.py</span>",
            "x": 0.01,
            "y": 0.97,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.04,
            "xanchor": "left",
            "x": 0,
        },
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
    )
    return fig


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=here / "results-pushdown-4m.jsonl")
    ap.add_argument(
        "--indexed", type=Path, default=here / "results-pushdown-4m-indexed.jsonl"
    )
    ap.add_argument(
        "--patched", type=Path, default=here / "results-pushdown-4m-patched.jsonl"
    )
    ap.add_argument(
        "--patched-indexed",
        type=Path,
        default=here / "results-pushdown-4m-patched-indexed.jsonl",
    )
    ap.add_argument(
        "--upstream", type=Path, default=here / "results-pushdown-upstream-1m.jsonl"
    )
    ap.add_argument("--coverage", type=Path, default=here / "results-coverage.jsonl")
    ap.add_argument("--out", type=Path, default=here / "plots")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="drop a sibling plotly.min.js so the pages need no network",
    )
    ap.add_argument(
        "--static",
        action="store_true",
        help="also write PNG and SVG for embedding in the docs (needs kaleido)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pages: list[tuple[str, go.Figure]] = []
    if args.results.exists():
        pages.append(
            (
                "pushdown",
                scan_figure(
                    load(args.results),
                    "4M rows, 256-byte payload, best of 3 — no scalar indices",
                ),
            )
        )
    if args.indexed.exists():
        pages.append(
            (
                "pushdown-indexed",
                scan_figure(
                    load(args.indexed),
                    "the same dataset, with BTREE on id, BITMAP on cat, NGRAM on text",
                ),
            )
        )
    for name, path, note in (
        (
            "pushdown-patched",
            args.patched,
            "no scalar indices",
        ),
        (
            "pushdown-patched-indexed",
            args.patched_indexed,
            "with BTREE on id, BITMAP on cat, NGRAM on text",
        ),
    ):
        if path.exists():
            pages.append(
                (
                    name,
                    scan_figure(
                        load(path),
                        "the same matrix on a Polars that passes the provider the "
                        f"whole predicate — 4M rows, {note}.<br>An unoptimised "
                        "build, so these compare to each other and not to the "
                        "pages above",
                        labels=PATCHED_LABEL,
                    ),
                )
            )
    if args.upstream.exists():
        pages.append(("pushdown-upstream", upstream_figure(load(args.upstream))))
    if args.coverage.exists():
        pages.append(("pushdown-coverage", coverage_figure(load(args.coverage))))

    for name, fig in pages:
        path = args.out / f"{name}.html"
        fig.write_html(
            path,
            include_plotlyjs=("directory" if args.offline else "cdn"),
            full_html=True,
        )
        print(f"wrote {path}")

    if args.static:
        static = args.out / "static"
        static.mkdir(parents=True, exist_ok=True)
        for name, fig in pages:
            for ext in ("png", "svg"):
                path = static / f"{name}.{ext}"
                fig.write_image(path, scale=2 if ext == "png" else 1)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
