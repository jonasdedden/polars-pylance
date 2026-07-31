"""What actually reaches Lance. These are the tests that would catch a silent
regression to "read everything, filter in Polars"."""

from __future__ import annotations

from typing import Any

import polars as pl
import pyarrow as pa
import pytest

from polars_lance import scan_lance


def _scan_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the schema-probing call, which projects nothing and filters nothing."""
    return [
        c for c in calls if c.get("columns") is not None or c.get("filter") is not None
    ]


def test_projection_reaches_lance(
    lance_uri: str, impl: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, impl=impl).select("id", "val").collect(engine="streaming")
    calls = _scan_calls(scanner_calls)
    assert calls, "no projected scan reached Lance"
    assert all(set(c["columns"]) <= {"id", "val"} for c in calls)
    assert all("payload" not in c["columns"] for c in calls)


def test_filter_columns_are_read_but_not_returned(
    lance_uri: str, impl: str, scanner_calls: list[dict[str, Any]]
) -> None:
    got = (
        scan_lance(lance_uri, impl=impl)
        .filter(pl.col("cat") == "b")
        .select("id")
        .collect(engine="streaming")
    )
    assert got.columns == ["id"]
    assert all("payload" not in c["columns"] for c in _scan_calls(scanner_calls))


def test_predicate_reaches_lance(
    lance_uri: str, impl: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, impl=impl).filter(
        (pl.col("cat") == "b") & (pl.col("val") > 0.9)
    ).select("id").collect(engine="streaming")

    filters = [
        c["filter"] for c in _scan_calls(scanner_calls) if c.get("filter") is not None
    ]
    assert filters, f"no filter pushed down (impl={impl})"
    if impl == "provider":
        assert all(isinstance(f, pa.compute.Expression) for f in filters)
    else:
        assert all(isinstance(f, str) for f in filters)
        assert "cat" in filters[0] and "val" in filters[0]


def test_predicate_pushdown_can_be_disabled(
    lance_uri: str, impl: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, impl=impl, predicate_pushdown=False).filter(
        pl.col("cat") == "b"
    ).select("id").collect(engine="streaming")
    assert all(c.get("filter") is None for c in scanner_calls)


def test_untranslatable_predicate_is_not_pushed(
    lance_uri: str, expected: pl.DataFrame, scanner_calls: list[dict[str, Any]]
) -> None:
    """The io_plugin path must stay correct when SQL translation gives up."""
    got = (
        scan_lance(lance_uri, impl="io_plugin")
        .filter(pl.col("cat").str.starts_with("b"))
        .select(pl.len())
        .collect(engine="streaming")
    )
    assert got.item() == expected.filter(pl.col("cat").str.starts_with("b")).height
    assert all(c.get("filter") is None for c in scanner_calls)


def test_limit_reaches_lance_without_filter(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    """Polars only offers a row limit when nothing will be filtered out later."""
    scan_lance(lance_uri, impl="provider").select("id").head(5).collect(
        engine="streaming"
    )
    limits = [c.get("limit") for c in _scan_calls(scanner_calls)]
    assert 5 in limits


def test_limit_is_not_pushed_past_a_filter(
    lance_uri: str, expected: pl.DataFrame, scanner_calls: list[dict[str, Any]]
) -> None:
    """A pushed limit combined with an unpushed filter would truncate too early."""
    got = (
        scan_lance(lance_uri, impl="io_plugin", predicate_pushdown=False)
        .filter(pl.col("cat") == "d")
        .sort("id")
        .head(3)
        .collect(engine="streaming")
    )
    want = expected.filter(pl.col("cat") == "d").sort("id").head(3)
    assert got.height == 3
    assert got["id"].to_list() == want["id"].to_list()
    for call in _scan_calls(scanner_calls):
        assert not (call.get("limit") and call.get("filter") is None), (
            "limit pushed into Lance while the filter was handled downstream"
        )


def test_scan_options_reach_lance(
    lance_uri: str, impl: str, scanner_calls: list[dict[str, Any]]
) -> None:
    from polars_lance import LanceScanOptions

    options = LanceScanOptions(batch_size=1_234, io_buffer_size=8 * 1024 * 1024)
    scan_lance(lance_uri, impl=impl, options=options).select("id").collect(
        engine="streaming"
    )
    calls = _scan_calls(scanner_calls)
    assert calls
    assert all(c["io_buffer_size"] == 8 * 1024 * 1024 for c in calls)
    # The io_plugin path lets the engine's batch-size hint win; the provider path
    # has no hint to honour and uses ours.
    if impl == "provider":
        assert all(c["batch_size"] == 1_234 for c in calls)


def test_unknown_impl_rejected(lance_uri: str) -> None:
    with pytest.raises(ValueError, match="unknown impl"):
        scan_lance(lance_uri, impl="nope")  # type: ignore[arg-type]
