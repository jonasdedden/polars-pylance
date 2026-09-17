"""What actually reaches Lance, and that the answer is right either way.

These catch a silent regression to "read everything, filter in Polars", and in
the other direction a filter pushed so far that rows go missing.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Literal

import lance
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from conftest import ScannerCall, spy_on_scanner
from polars_pylance import LanceScanOptions, scan_lance
from polars_pylance._predicate import LanceFilter

if TYPE_CHECKING:
    from pathlib import Path


def _scan_calls(calls: list[ScannerCall]) -> list[ScannerCall]:
    """Drop the schema-probing call, which projects nothing and filters nothing."""
    return [c for c in calls if c.columns is not None or c.filter is not None]


# ---------------------------------------------------------------------------
# projection and limit
# ---------------------------------------------------------------------------


def test_projection_reaches_lance(
    lance_uri: str, scanner_calls: list[ScannerCall]
) -> None:
    scan_lance(lance_uri).select("id", "val").collect(engine="streaming")
    calls = _scan_calls(scanner_calls)
    assert calls, "no projected scan reached Lance"
    for call in calls:
        assert call.columns is not None
        assert set(call.columns) <= {"id", "val"}


def test_filter_columns_are_read_but_not_returned(
    lance_uri: str, scanner_calls: list[ScannerCall]
) -> None:
    got = (
        scan_lance(lance_uri)
        .filter(pl.col("cat") == "b")
        .select("id")
        .collect(engine="streaming")
    )
    assert got.columns == ["id"]
    for call in _scan_calls(scanner_calls):
        assert call.columns is not None
        assert "payload" not in call.columns


def test_limit_reaches_lance_without_filter(
    lance_uri: str, scanner_calls: list[ScannerCall]
) -> None:
    scan_lance(lance_uri).select("id").head(5).collect(engine="streaming")
    assert 5 in [c.limit for c in _scan_calls(scanner_calls)]


@pytest.mark.parametrize(
    ("predicate", "pushdown"),
    [
        pytest.param(pl.col("cat") == "d", True, id="pushed filter"),
        pytest.param(pl.col("cat").str.slice(0, 1) == "d", True, id="residual filter"),
        pytest.param(pl.col("cat") == "d", False, id="pushdown disabled"),
    ],
)
def test_head_after_a_filter_counts_surviving_rows(
    lance_uri: str,
    expected: pl.DataFrame,
    scanner_calls: list[ScannerCall],
    predicate: pl.Expr,
    pushdown: bool,  # noqa: FBT001 - a pytest parameter
) -> None:
    """A limit pushed while a filter is still to run downstream would truncate early."""
    got = (
        scan_lance(lance_uri, predicate_pushdown=pushdown)
        .filter(predicate)
        .sort("id")
        .head(3)
        .collect(engine="streaming")
    )
    want = expected.filter(predicate).sort("id").head(3)
    assert_frame_equal(got, want)
    for call in _scan_calls(scanner_calls):
        assert not (call.limit and call.filter is None), (
            "limit pushed into Lance while the filter was handled downstream"
        )


def test_scan_options_reach_lance(
    lance_uri: str, scanner_calls: list[ScannerCall]
) -> None:
    options = LanceScanOptions(batch_size=1_234, io_buffer_size=8 * 1024 * 1024)
    scan_lance(lance_uri, options=options).select("id").collect(engine="streaming")
    calls = _scan_calls(scanner_calls)
    assert calls
    assert all(c.io_buffer_size == 8 * 1024 * 1024 for c in calls)
    # The option beats the engine's batch-size hint, which is sized in rows with
    # no idea how wide they are.
    assert all(c.batch_size == 1_234 for c in calls)


def test_the_engine_batch_size_hint_is_used_when_no_option_asks_otherwise(
    lance_uri: str, scanner_calls: list[ScannerCall]
) -> None:
    options = LanceScanOptions(batch_size=None)
    scan_lance(lance_uri, options=options).select("id").collect(engine="streaming")
    sizes = {c.batch_size for c in _scan_calls(scanner_calls)}
    assert sizes
    assert None not in sizes


# ---------------------------------------------------------------------------
# what the filter is, and whether it gets there
# ---------------------------------------------------------------------------


# Most of these are silently dropped by Polars' own PyArrow lowering (as of 1.44.2),
# which is the whole argument for translating the predicate here instead.
PUSHED: list[tuple[str, pl.Expr, str]] = [
    ("comparison", (pl.col("cat") == "b") & (pl.col("val") > 0.9), "`cat` = 'b'"),
    # Polars lowers `is_in` to PyArrow only up to 100 values.
    ("long is_in", pl.col("id").is_in(list(range(0, 1_000, 5))), "IN"),
    ("starts_with", pl.col("cat").str.starts_with("b"), "starts_with"),
    ("contains", pl.col("cat").str.contains("b", literal=True), "contains"),
    ("contains_any", pl.col("cat").str.contains_any(["b", "c"]), "contains"),
    ("arithmetic", (pl.col("val") * 2) > 1.5, "*"),
    ("string length", pl.col("cat").str.len_chars() == 1, "length"),
    ("is_nan", pl.col("val").is_nan(), "isnan"),
    ("xor", (pl.col("id") > 5) ^ (pl.col("cat") == "b"), "NOT"),
    ("min_horizontal", pl.min_horizontal("id", pl.col("id") * 2) > 100, "least"),
    # The optimizer casts one side before the plugin sees the comparison.
    ("mixed types", pl.col("id") > pl.col("val"), "TRY_CAST"),
    # The pushed half narrows the read; the rest is finished in Polars.
    (
        "partly",
        (pl.col("val") > 0.9) & (pl.col("cat").str.slice(0, 1) == "b"),
        "`val` > 0.9",
    ),
]


@pytest.mark.parametrize(
    ("predicate", "fragment"),
    [pytest.param(e, f, id=name) for name, e, f in PUSHED],
)
def test_predicate_reaches_lance(
    predicate: pl.Expr,
    fragment: str,
    lance_uri: str,
    expected: pl.DataFrame,
    pushed_filters: list[str],
) -> None:
    got = (
        scan_lance(lance_uri)
        .filter(predicate)
        .select(pl.len())
        .collect(engine="streaming")
    )
    assert pushed_filters, "predicate did not reach Lance"
    assert all(fragment in f for f in pushed_filters)
    assert got.item() == expected.filter(predicate).height


@pytest.mark.parametrize(
    ("predicate", "pushdown", "with_row_id"),
    [
        pytest.param(pl.col("cat") == "b", False, False, id="disabled"),
        pytest.param(
            pl.col("cat").str.slice(0, 1) == "b", True, False, id="untranslatable"
        ),
        # `_rowid` does not exist while Lance evaluates the filter.
        pytest.param(pl.col("_rowid") < 10, True, True, id="generated column"),
    ],
)
def test_predicate_stays_in_polars(
    predicate: pl.Expr,
    pushdown: bool,  # noqa: FBT001 - a pytest parameter
    with_row_id: bool,  # noqa: FBT001 - a pytest parameter
    lance_uri: str,
    pushed_filters: list[str],
) -> None:
    lf = scan_lance(lance_uri, predicate_pushdown=pushdown, with_row_id=with_row_id)
    got = lf.filter(predicate).select(pl.len()).collect(engine="streaming")
    assert not pushed_filters
    frame = pl.from_arrow(lance.dataset(lance_uri).to_table(with_row_id=True))
    assert isinstance(frame, pl.DataFrame)
    assert got.item() == frame.filter(predicate).height


def test_a_filter_lance_refuses_is_dropped_rather_than_raised(
    lance_uri: str, expected: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unplannable filter costs speed, never the query."""
    from polars_pylance import _scan

    monkeypatch.setattr(
        _scan,
        "to_lance_filter",
        lambda *a, **k: LanceFilter(sql="no_such_function(`cat`)", exact=True),
    )
    predicate = pl.col("cat") == "b"
    with pytest.warns(RuntimeWarning, match="rejected the pushed-down filter"):
        got = (
            scan_lance(lance_uri)
            .filter(predicate)
            .select(pl.len())
            .collect(engine="streaming")
        )
    assert got.item() == expected.filter(predicate).height


# ---------------------------------------------------------------------------
# end to end: the pushed query and the eager one agree
# ---------------------------------------------------------------------------

END_TO_END: list[tuple[str, pl.Expr]] = [
    ("is_in", pl.col("id").is_in([1, 2, 3, 1_999])),
    ("not in", ~pl.col("cat").is_in(["alpha", "beta"])),
    ("starts_with", pl.col("cat").str.starts_with("b")),
    ("contains regex", pl.col("text").str.contains(r"row-000\d\d-beta")),
    ("contains_any", pl.col("text").str.contains_any(["-0001", "-0002"])),
    ("strip and match", pl.col("text").str.strip_chars(" ").str.ends_with("delta")),
    ("replace_all", pl.col("cat").str.replace_all("a", "", literal=True) == "bet"),
    ("len_bytes", pl.col("cat").str.len_bytes() > 5),
    ("arithmetic", (pl.col("val") * 100) % 2 > 1),
    ("power", (pl.col("id") ** 2) > 3_500_000),
    ("is_nan or inf", pl.col("odd").is_nan() | pl.col("odd").is_infinite()),
    ("is_finite", pl.col("odd").is_finite() & (pl.col("odd") > 3.0)),
    ("null handling", pl.col("opt").fill_null(-1) < 0),
    ("eq_missing", pl.col("opt").eq_missing(14)),
    ("xor", (pl.col("id") > 1_000) ^ pl.col("flag")),
    ("min_horizontal", pl.min_horizontal("id", "opt") > 1_500),
    ("date part", pl.col("ts").dt.hour() < 3),
    ("weekday", pl.col("ts").dt.weekday() == 7),
    ("truncate", pl.col("ts").dt.truncate("1d") == dt.datetime(2024, 1, 3)),
    ("date literal", pl.col("day") > dt.date(2024, 11, 1)),
    ("list contains", pl.col("tags").list.contains(4)),
    ("list get", pl.col("tags").list.get(0, null_on_oob=True).is_null()),
    ("struct field", pl.col("meta").struct.field("s") == "gamma"),
    (
        "relaxed conjunction",
        (pl.col("id") > 500) & (pl.col("cat").str.slice(0, 1) == "b"),
    ),
    (
        "nested",
        ((pl.col("cat") == "beta") | pl.col("odd").is_nan())
        & pl.col("text").str.contains("row-001", literal=True),
    ),
    ("nothing pushable", pl.col("cat").str.slice(1, 2) == "et"),
]


@pytest.mark.parametrize(
    "predicate", [pytest.param(e, id=name) for name, e in END_TO_END]
)
def test_scan_matches_an_eager_read(
    predicate: pl.Expr, rich_uri: str, rich_frame: pl.DataFrame
) -> None:
    """The point of the whole exercise: pushdown must not change the answer."""
    got = (
        scan_lance(rich_uri)
        .filter(predicate)
        .select("id")
        .collect(engine="streaming")
        .sort("id")
    )
    want = rich_frame.filter(predicate).select("id").sort("id")
    assert_frame_equal(got, want)


def test_an_indexed_column_keeps_its_index(
    tmp_path: Path, rich_frame: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `CAST` around an indexed column costs the index, so it must not appear."""
    uri = str(tmp_path / "indexed.lance")
    lance.write_dataset(rich_frame.to_arrow(), uri)
    dataset = lance.dataset(uri)
    dataset.create_scalar_index("val", index_type="BTREE")

    # Recording starts here, rather than at fixture time, so building the index
    # above does not count as a pushed-down filter.
    filters: list[str] = []

    def record(call: ScannerCall) -> None:
        if call.filter is not None:
            filters.append(call.filter)

    spy_on_scanner(monkeypatch, record)
    got = (
        scan_lance(uri)
        .filter(pl.col("val") > 0.999)
        .select(pl.len())
        .collect(engine="streaming")
    )

    assert filters == ["((`val` > 0.999) OR `val` < CAST('-inf' AS double))"]
    assert "ScalarIndexQuery" in dataset.scanner(filter=filters[0]).explain_plan()
    assert got.item() == rich_frame.filter(pl.col("val") > 0.999).height


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("x").eq_missing(3),
        pl.col("x").ne_missing(3),
        ~pl.col("x").eq_missing(3),
        ~pl.col("x").ne_missing(3),
        pl.lit(3).ne_missing(pl.col("x")),
        pl.col("x").eq_missing(3) ^ pl.col("flag"),
        ~(pl.col("x").eq_missing(3) | pl.col("flag")),
        pl.col("x").ne_missing(3) & pl.col("flag"),
    ],
)
def test_null_safe_comparisons_preserve_null_rows(
    tmp_path: Path, predicate: pl.Expr, engine: Literal["streaming", "in-memory"]
) -> None:
    frame = pl.DataFrame(
        {
            "id": range(6),
            "x": [None, 3, 4, None, 3, 4],
            "flag": [True] * 3 + [False] * 3,
        }
    )
    uri = str(tmp_path / "nulls.lance")
    lance.write_dataset(frame.to_arrow(), uri)
    got = scan_lance(uri).filter(predicate).select("id").collect(engine=engine)
    assert_frame_equal(got, frame.filter(predicate).select("id"))
