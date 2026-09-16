"""Test the resolver's execution contract even on stable Polars."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pylance import LanceScanSpec, _resolver, scan_lance
from polars_pylance._predicate import LanceFilter
from polars_pylance._resolver import (
    _planned_source,
    _resolve_filters,
)

if TYPE_CHECKING:
    from conftest import ScannerCall


def test_resolver_requires_compatible_polars() -> None:
    if hasattr(pl.LazyFrame, "from_lazyframe_resolver"):
        pytest.skip("resolver API available")
    with pytest.raises(ImportError, match="requires a Polars build"):
        scan_lance("not-opened.lance", backend="resolver")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scan backend"):
        scan_lance("not-opened.lance", backend="unknown")  # type: ignore[arg-type]


def test_planned_source_retains_residual_and_drops_exact_filter_columns(
    rich_uri: str, rich_frame: pl.DataFrame, scanner_calls: list[ScannerCall]
) -> None:
    exact = pl.col("val") > 0.9
    residual = pl.col("cat").str.slice(0, 1) == "b"
    plan, applied = _resolve_filters(
        LanceScanSpec(rich_uri), [exact, residual], rich_frame.schema
    )
    assert applied == {0}
    lf = _planned_source(plan, rich_frame.schema)
    got = lf.filter(residual).select("id").collect(engine="streaming")
    assert_frame_equal(got, rich_frame.filter(exact & residual).select("id"))
    assert all("val" not in c.columns for c in scanner_calls if c.columns is not None)


def test_acknowledged_filter_survives_execution_time_rejection(
    rich_uri: str, rich_frame: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = pl.col("val") > 0.9
    residual = pl.col("cat").str.slice(0, 1) == "b"
    monkeypatch.setattr(
        _resolver,
        "to_lance_filter",
        lambda *a, **k: LanceFilter("no_such_function(`val`)", exact=True),
    )
    plan, applied = _resolve_filters(
        LanceScanSpec(rich_uri), [exact], rich_frame.schema
    )
    assert applied == {0}
    lf = _planned_source(plan, rich_frame.schema)
    with pytest.warns(RuntimeWarning, match="rejected the pushed-down filter"):
        got = lf.filter(residual).select("id").head(3).collect(engine="streaming")
    assert_frame_equal(got, rich_frame.filter(exact & residual).select("id").head(3))


def test_search_filters_are_not_acknowledged(rich_frame: pl.DataFrame) -> None:
    expr = pl.col("id") > 3
    for spec in (
        LanceScanSpec("unused", nearest={"column": "vector", "q": [1.0]}),
        LanceScanSpec("unused", full_text_query="example"),
        LanceScanSpec("unused", prefilter="id > 0"),
        LanceScanSpec("unused", predicate_pushdown=False),
    ):
        plan, applied = _resolve_filters(spec, [expr], rich_frame.schema)
        assert not applied
        assert plan.sql == spec.prefilter
