"""Render results.jsonl as interactive Plotly pages.

    uv run --group bench bench/plot.py bench/results-m8id4xl.jsonl

Writes one HTML per view into `--out` (default: next to the results file). The
pages link plotly from the CDN so they stay a few KB; pass ``--offline`` to
drop a sibling ``plotly.min.js`` instead and make them work without a network.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

GIB_PER_MROW = 0.4915  # measured: 190.7 GiB / 388M rows

PHASES = {
    "scaling": "generous cap — how peak RSS and runtime grow with dataset size",
    "fixed-budget": "memory pinned, data grown past it — does the job finish?",
}

CASES: dict[str, str] = {
    "r_full": "full scan + payload aggregate",
    "r_proj": "projection-only aggregate",
    "r_filter_lo": "selective filter (val > 0.999)",
    "r_filter_hi": "50% filter + payload aggregate",
    "r_cat": "string predicate (cat == 'a')",
    "r_head": "head(10) — limit pushdown",
    "r_topk": "sort + head(10) — top-k",
    "w_sink": "write filtered projection",
    "w_frag": "fragment-parallel write (polars-pylance only)",
    "l_version": "pinned version (polars-pylance only)",
    "l_rowid": "_rowid column (polars-pylance only)",
    "l_fragments": "sharded fragment scan (polars-pylance only)",
}
THEIRS, OURS = "polars-lance", "polars-pylance"
COLOUR = {THEIRS: "#d1495b", OURS: "#2a9d8f"}
FAIL_SYMBOL = {"OOM-killed": "x", "panic-unsendable": "triangle-down"}


def _log_ticks(
    values: list[float],
) -> tuple[list[float] | None, list[str] | None]:
    """1-2-5 ticks across the data's decades, labelled with the full number.

    Plotly's default log minor ticks are labelled with the mantissa alone, so the
    tick at 20 reads "2" sitting just past the one reading "10". Supplying the
    values explicitly avoids that.
    """
    finite = [v for v in values if v and v > 0]
    if not finite:
        return None, None  # let plotly choose rather than showing no ticks
    lo, hi = min(finite), max(finite)
    decade = math.floor(math.log10(lo))
    vals: list[float] = []
    while decade <= math.ceil(math.log10(hi)):
        for mant in (1, 2, 5):
            vals.append(mant * 10.0**decade)
        decade += 1
    vals = [v for v in vals if lo / 2.5 <= v <= hi * 2.5]

    def fmt(v: float) -> str:
        if v >= 1000:
            return f"{v:,.0f}"
        if v >= 1:
            return f"{v:g}"
        return f"{v:.10f}".rstrip("0")

    return vals, [fmt(v) for v in vals]


def load(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            # a partial trailing line appears when the matrix is still running
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def gib(rows: int) -> float:
    return rows * GIB_PER_MROW / 1e6


def grid(rows_of_data: list[dict[str, Any]], phase: str) -> go.Figure:
    """One subplot per case, with both costs on shared x = dataset size.

    Runtime is on the left axis (solid, circles) and peak memory on the right
    (dashed, diamonds); colour is the implementation. Keeping them together is
    the point of the comparison -- the interesting cases are the ones where the
    two metrics disagree, and side-by-side pages hide exactly that.
    """
    cases = [
        c
        for c in CASES
        if any(r["case"] == c and r.get("phase") == phase for r in rows_of_data)
    ]
    ncols = 3
    nrows = -(-len(cases) // ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[CASES[c] for c in cases],
        specs=[[{"secondary_y": True} for _ in range(ncols)] for _ in range(nrows)],
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )

    METRICS = (
        ("seconds", "runtime (s)", "solid", "circle", False),
        ("peak_gib", "peak RSS (GiB)", "dot", "diamond", True),
    )

    for i, case in enumerate(cases):
        r_, c_ = divmod(i, ncols)
        case_rows = [
            r
            for r in rows_of_data
            if r["case"] == case and r.get("phase") == phase and r["status"] == "ok"
        ]

        for impl in (THEIRS, OURS):
            pts = sorted(
                (
                    r
                    for r in rows_of_data
                    if r["case"] == case
                    and r["impl"] == impl
                    and r.get("phase") == phase
                ),
                key=lambda r: r["rows"],
            )
            ok = [pt for pt in pts if pt["status"] == "ok"]
            for key, label, dash, symbol, secondary in METRICS:
                if not ok:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[gib(pt["rows"]) for pt in ok],
                        y=[pt[key] for pt in ok],
                        mode="lines+markers",
                        name=f"{impl} — {label}",
                        legendgroup=f"{impl}-{key}",
                        showlegend=(i == 0),
                        line={"color": COLOUR[impl], "width": 2, "dash": dash},
                        marker={"size": 7, "symbol": symbol},
                        opacity=1.0 if not secondary else 0.75,
                        hovertemplate=(
                            f"<b>{impl}</b><br>source %{{x:.1f}} GiB<br>"
                            f"{label} %{{y:.3g}}<extra></extra>"
                        ),
                    ),
                    row=r_ + 1,
                    col=c_ + 1,
                    secondary_y=secondary,
                )

            # failures sit on the runtime axis so a gap is never mistaken for
            # "not measured"
            for status, symbol in FAIL_SYMBOL.items():
                bad = [pt for pt in pts if pt["status"] == status]
                if not bad:
                    continue
                floor = min(
                    (r["seconds"] for r in case_rows),
                    default=1.0,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[gib(pt["rows"]) for pt in bad],
                        y=[floor] * len(bad),
                        mode="markers",
                        name=f"{impl}: {status}",
                        legendgroup=f"{impl}-{status}",
                        showlegend=(i == 0),
                        marker={
                            "symbol": symbol,
                            "size": 13,
                            "color": COLOUR[impl],
                            "line": {"width": 1, "color": "#222"},
                        },
                        hovertemplate=(
                            f"<b>{impl}</b><br>%{{x:.1f}} GiB<br>"
                            f"{status}<extra></extra>"
                        ),
                    ),
                    row=r_ + 1,
                    col=c_ + 1,
                    secondary_y=False,
                )

        xv, xt = _log_ticks([gib(r["rows"]) for r in case_rows])
        tv, tt = _log_ticks([r["seconds"] for r in case_rows])
        mv, mt = _log_ticks([r["peak_gib"] for r in case_rows])
        fig.update_xaxes(
            type="log",
            title_text="source GiB",
            tickvals=xv,
            ticktext=xt,
            row=r_ + 1,
            col=c_ + 1,
        )
        fig.update_yaxes(
            type="log",
            title_text="runtime (s)",
            tickvals=tv,
            ticktext=tt,
            row=r_ + 1,
            col=c_ + 1,
            secondary_y=False,
        )
        fig.update_yaxes(
            type="log",
            title_text="peak RSS (GiB)",
            tickvals=mv,
            ticktext=mt,
            showgrid=False,  # two grids on one panel is unreadable
            row=r_ + 1,
            col=c_ + 1,
            secondary_y=True,
        )

    fig.update_layout(
        height=380 * nrows,
        title=(
            f"{phase} — runtime (solid, left) and peak memory (dotted, right)"
            f"<br><sub>{PHASES.get(phase, '')}</sub>"
        ),
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.04 / nrows},
    )
    return fig


# The panels worth embedding in COMPARISON.md: (phase, case, filename stem).
FOCUS = [
    ("scaling", "w_sink", "write-scaling"),
    ("scaling", "r_full", "full-scan-scaling"),
    ("scaling", "r_cat", "string-predicate-scaling"),
    ("scaling", "r_filter_lo", "selective-filter-scaling"),
    ("scaling", "r_filter_hi", "half-filter-scaling"),
    ("fixed-budget", "w_sink", "write-fixed-budget"),
]


def focus_figure(
    rows_of_data: list[dict[str, Any]], phase: str, case: str
) -> go.Figure:
    """A single case as one standalone panel, for static export."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    case_rows = [
        r
        for r in rows_of_data
        if r["case"] == case and r.get("phase") == phase and r["status"] == "ok"
    ]
    metrics = (
        ("seconds", "runtime (s)", "solid", "circle", False),
        ("peak_gib", "peak RSS (GiB)", "dot", "diamond", True),
    )
    for impl in (THEIRS, OURS):
        pts = sorted(
            (
                r
                for r in rows_of_data
                if r["case"] == case and r["impl"] == impl and r.get("phase") == phase
            ),
            key=lambda r: r["rows"],
        )
        ok = [pt for pt in pts if pt["status"] == "ok"]
        for key, label, dash, symbol, secondary in metrics:
            if not ok:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[gib(pt["rows"]) for pt in ok],
                    y=[pt[key] for pt in ok],
                    mode="lines+markers",
                    name=f"{impl} — {label}",
                    line={"color": COLOUR[impl], "width": 2.5, "dash": dash},
                    marker={"size": 9, "symbol": symbol},
                ),
                secondary_y=secondary,
            )
        for status, symbol in FAIL_SYMBOL.items():
            bad = [pt for pt in pts if pt["status"] == status]
            if not bad:
                continue
            floor = min((r["seconds"] for r in case_rows), default=1.0)
            fig.add_trace(
                go.Scatter(
                    x=[gib(pt["rows"]) for pt in bad],
                    y=[floor] * len(bad),
                    mode="markers+text",
                    name=f"{impl}: {status}",
                    text=["OOM" if status == "OOM-killed" else "crash"] * len(bad),
                    textposition="top center",
                    textfont={"size": 11, "color": COLOUR[impl]},
                    marker={
                        "symbol": symbol,
                        "size": 15,
                        "color": COLOUR[impl],
                        "line": {"width": 1, "color": "#222"},
                    },
                ),
                secondary_y=False,
            )

    xv, xt = _log_ticks([gib(r["rows"]) for r in case_rows])
    tv, tt = _log_ticks([r["seconds"] for r in case_rows])
    mv, mt = _log_ticks([r["peak_gib"] for r in case_rows])
    fig.update_xaxes(
        type="log", title_text="source dataset (GiB)", tickvals=xv, ticktext=xt
    )
    fig.update_yaxes(
        type="log",
        title_text="runtime (s)",
        tickvals=tv,
        ticktext=tt,
        secondary_y=False,
    )
    fig.update_yaxes(
        type="log",
        title_text="peak RSS (GiB)",
        tickvals=mv,
        ticktext=mt,
        showgrid=False,
        secondary_y=True,
    )
    fig.update_layout(
        width=900,
        height=480,
        title=f"{CASES[case]} — {phase}",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 70, "r": 70, "t": 60, "b": 90},
    )
    return fig


def ratio_figure(rows_of_data: list[dict[str, Any]]) -> go.Figure:
    """polars-pylance / polars-lance: 1.0 is parity, below 1.0 means we win."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("runtime ratio", "peak-memory ratio")
    )
    for col, key in ((1, "seconds"), (2, "peak_gib")):
        all_ratios: list[float] = []
        for case in CASES:
            pairs = []
            for n in sorted({r["rows"] for r in rows_of_data if r["case"] == case}):
                d = {
                    r["impl"]: r
                    for r in rows_of_data
                    if r["case"] == case
                    and r["rows"] == n
                    and r.get("phase") == "scaling"
                }
                t, o = d.get(THEIRS), d.get(OURS)
                if t and o and t["status"] == "ok" and o["status"] == "ok":
                    pairs.append((gib(n), o[key] / t[key]))
            if len(pairs) < 2:
                continue
            all_ratios.extend(p[1] for p in pairs)
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in pairs],
                    y=[p[1] for p in pairs],
                    mode="lines+markers",
                    name=CASES[case],
                    legendgroup=case,
                    showlegend=(col == 1),
                    hovertemplate=f"<b>{CASES[case]}</b><br>%{{x:.1f}} GiB<br>"
                    f"ratio %{{y:.2f}}x<extra></extra>",
                ),
                row=1,
                col=col,
            )
        xv, xt = _log_ticks([gib(r["rows"]) for r in rows_of_data])
        yv, yt = _log_ticks(all_ratios)
        fig.update_xaxes(
            type="log",
            title_text="source GiB",
            tickvals=xv,
            ticktext=xt,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            type="log",
            title_text="polars-pylance / polars-lance",
            tickvals=yv,
            ticktext=yt,
            row=1,
            col=col,
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888")
    fig.update_layout(
        height=560,
        title="polars-pylance ÷ polars-lance — below 1.0 means polars-pylance wins",
        template="plotly_white",
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="drop a sibling plotly.min.js (4.7 MB) so the pages need no "
        "network; the default links the CDN and keeps them a few KB",
    )
    ap.add_argument(
        "--static",
        action="store_true",
        help="also write the FOCUS panels as PNG and SVG for embedding in "
        "COMPARISON.md (needs kaleido)",
    )
    args = ap.parse_args()
    out = args.out or args.results.parent
    out.mkdir(parents=True, exist_ok=True)

    data = load(args.results)
    seen = {str(r["phase"]) for r in data if r.get("phase")}
    phases = [p for p in PHASES if p in seen] + sorted(seen - set(PHASES))

    pages: list[tuple[str, go.Figure]] = []
    for phase in phases:
        pages.append((f"{phase}.html", grid(data, phase)))
    pages.append(("ratios.html", ratio_figure(data)))

    for name, fig in pages:
        fig.write_html(
            out / name,
            include_plotlyjs=("directory" if args.offline else "cdn"),
            full_html=True,
        )
        print(f"wrote {out / name}")
    if args.static:
        static = out / "static"
        static.mkdir(parents=True, exist_ok=True)
        for phase, case, stem in FOCUS:
            fig = focus_figure(data, phase, case)
            for ext in ("png", "svg"):
                path = static / f"{stem}.{ext}"
                fig.write_image(path, scale=2 if ext == "png" else 1)
                print(f"wrote {path}")

    print(f"({len(data)} datapoints)")


if __name__ == "__main__":
    main()
