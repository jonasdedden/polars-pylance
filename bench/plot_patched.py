"""Render the patched-Polars results as Plotly pages.

    uv run --group bench bench/plot_patched.py --static

Reads what ``bench/pushdown.py`` writes for each (Polars build x dataset) and
produces three views, all in the same idiom as ``bench/plot_pushdown.py`` --
dots on a log axis, values printed beside each mark, one panel per measure:

``patched-provider.html``
    The default scan path on the two builds. One row per case, one dot per
    build, so the length of each rule is what the three upstream PRs moved.
``patched-attribution.html``
    The same path on the patched build with one lowering pinned at a time:
    Polars' own PyArrow lowering (what #28994 and #28996 buy) against the
    visitor's Lance SQL (what #28995 buys).
``patched-hooks.html``
    Provider against IO plugin with the same filter behind both, plus the
    IO-plugin variant that lets the engine re-apply the predicate.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from plot_pushdown import (
    CASES,
    INK,
    RULE,
    SURFACE,
    _compact,
    _seconds,
    dumbbell,
    load,
)

RESULTS = Path(__file__).parent
PLOTS = RESULTS / "plots"

# Categorical slots of the same reference palette `plot_pushdown.py` uses.
BLUE, ORANGE, GREEN, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8a5fbf"

BUILD_SERIES = (
    ["base", "patched"],
    {"base": "#8a8984", "patched": BLUE},
    {"base": 0.2, "patched": -0.2},
)
BUILD_LABEL = {
    "base": "upstream main",
    "patched": "with #28994 + #28995 + #28996",
}

ATTRIBUTION_SERIES = (
    ["provider_pa", "provider_sql", "provider"],
    {"provider_pa": ORANGE, "provider_sql": GREEN, "provider": BLUE},
    {"provider_pa": 0.26, "provider_sql": 0.0, "provider": -0.26},
)
ATTRIBUTION_LABEL = {
    "provider_pa": "Polars → PyArrow only (#28994, #28996)",
    "provider_sql": "visitor → Lance SQL only (#28995)",
    "provider": "both, as shipped",
}

HOOK_SERIES = (
    ["engine", "io_plugin", "io_plugin_rust", "provider"],
    {
        "engine": "#8a8984",
        "io_plugin": ORANGE,
        "io_plugin_rust": PLUM,
        "provider": BLUE,
    },
    {"engine": 0.3, "io_plugin": 0.1, "io_plugin_rust": -0.1, "provider": -0.3},
)
HOOK_LABEL = {
    "engine": "no pushdown",
    "io_plugin": "io_plugin (re-filters in Python)",
    "io_plugin_rust": "io_plugin (engine re-filters)",
    "provider": "provider",
}


def _panels(
    rows: list[dict[str, Any]],
    *,
    series: Any,
    labels: dict[str, str],
    title: str,
    subtitle: str,
    key: str = "impl",
) -> go.Figure:
    """Two panels -- rows out of Lance, then wall time -- over the same cases."""
    cases = {k: v for k, v in CASES.items() if any(r["case"] == k for r in rows)}
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=("rows handed to Polars", "wall time"),
    )
    # `dumbbell` groups by "impl"; a build comparison groups by "build", so the
    # column is renamed rather than the helper duplicated.
    if key != "impl":
        rows = [dict(r, impl=r[key]) for r in rows]

    for col, (metric, label, fmt) in enumerate(
        (
            ("rows_from_lance", "rows — fewer is better", _compact),
            ("seconds", "seconds — lower is better", _seconds),
        ),
        start=1,
    ):
        traces, ticks, ticktext = dumbbell(
            rows,
            cases=cases,
            key=metric,
            title=label,
            label=label,
            value_fmt=fmt,
            labels=labels,
            series=series,
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
    fig.update_layout(
        template="plotly_white",
        height=170 + 46 * len(order),
        margin={"l": 260, "r": 40, "t": 115, "b": 110},
        title={
            "text": f"{title}<br>"
            f"<span style='font-size:12px;color:{INK}'>{subtitle}</span>",
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


def write(fig: go.Figure, name: str, static: bool) -> None:
    PLOTS.mkdir(exist_ok=True)
    (PLOTS / f"{name}.html").write_text(
        fig.to_html(include_plotlyjs="cdn", full_html=True)
    )
    print(f"wrote {PLOTS / f'{name}.html'}")
    if static:
        (PLOTS / "static").mkdir(exist_ok=True)
        for suffix, scale in ((".png", 2), (".svg", 1)):
            fig.write_image(PLOTS / "static" / f"{name}{suffix}", scale=scale)
        print(f"wrote {PLOTS / 'static' / name}.png/.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="also write PNG and SVG")
    parser.add_argument("--suffix", default="", help="e.g. -indexed")
    args = parser.parse_args()
    suffix = args.suffix

    base = load(RESULTS / f"results-rel-base-4m{suffix}.jsonl")
    patched = load(RESULTS / f"results-rel-patched-4m{suffix}.jsonl")
    dataset = "4M rows, 256-byte payload" + (
        ", scalar indices on id/cat/text" if suffix else ", no indices"
    )

    provider_only = [r for r in base + patched if r["impl"] == "provider"]
    write(
        _panels(
            provider_only,
            series=BUILD_SERIES,
            labels=BUILD_LABEL,
            title="What the three upstream PRs move on the default scan path",
            subtitle=(
                f"{dataset}. Same release build of Polars either side of the "
                "patches, best of three, one process per measurement."
            ),
            key="build",
        ),
        f"patched-provider{suffix}",
        args.static,
    )

    write(
        _panels(
            [r for r in patched if r["impl"] in ATTRIBUTION_SERIES[0]],
            series=ATTRIBUTION_SERIES,
            labels=ATTRIBUTION_LABEL,
            title="Which patch does the work",
            subtitle=(
                f"{dataset}. The provider path on the patched build with one "
                "lowering pinned at a time."
            ),
        ),
        f"patched-attribution{suffix}",
        args.static,
    )

    write(
        _panels(
            [r for r in patched if r["impl"] in HOOK_SERIES[0]],
            series=HOOK_SERIES,
            labels=HOOK_LABEL,
            title="IO plugin against dataset provider, same filter behind both",
            subtitle=(
                f"{dataset}. On the patched build both hooks push the identical "
                "filter, so what is left is the hook."
            ),
        ),
        f"patched-hooks{suffix}",
        args.static,
    )


if __name__ == "__main__":
    main()
