"""What actually reaches Lance. These are the tests that would catch a silent
regression to "read everything, filter in Polars"."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
import pyarrow.compute as pc
import pytest

from polars_pylance import scan_lance


def _scan_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the schema-probing call, which projects nothing and filters nothing."""
    return [
        c for c in calls if c.get("columns") is not None or c.get("filter") is not None
    ]


def test_projection_reaches_lance(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri).select("id", "val").collect(engine="streaming")
    calls = _scan_calls(scanner_calls)
    assert calls, "no projected scan reached Lance"
    assert all(set(c["columns"]) <= {"id", "val"} for c in calls)
    assert all("payload" not in c["columns"] for c in calls)


def test_filter_columns_are_read_but_not_returned(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    got = (
        scan_lance(lance_uri)
        .filter(pl.col("cat") == "b")
        .select("id")
        .collect(engine="streaming")
    )
    assert got.columns == ["id"]
    assert all("payload" not in c["columns"] for c in _scan_calls(scanner_calls))


def test_predicate_reaches_lance(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri).filter((pl.col("cat") == "b") & (pl.col("val") > 0.9)).select(
        "id"
    ).collect(engine="streaming")

    filters = [
        c["filter"] for c in _scan_calls(scanner_calls) if c.get("filter") is not None
    ]
    assert filters, "no filter pushed down"
    # Polars hands the provider a ready-made PyArrow expression, which is
    # exactly what Lance's scanner accepts -- no translation in between.
    assert all(isinstance(f, pc.Expression) for f in filters)


def test_predicate_pushdown_can_be_disabled(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, predicate_pushdown=False).filter(pl.col("cat") == "b").select(
        "id"
    ).collect(engine="streaming")
    assert all(c.get("filter") is None for c in scanner_calls)


def test_untranslatable_predicate_is_not_pushed(
    lance_uri: str, expected: pl.DataFrame, scanner_calls: list[dict[str, Any]]
) -> None:
    """A predicate Polars cannot express for PyArrow stays in the engine."""
    got = (
        scan_lance(lance_uri)
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
    scan_lance(lance_uri).select("id").head(5).collect(engine="streaming")
    limits = [c.get("limit") for c in _scan_calls(scanner_calls)]
    assert 5 in limits


def test_limit_is_not_pushed_past_a_filter(
    lance_uri: str, expected: pl.DataFrame, scanner_calls: list[dict[str, Any]]
) -> None:
    """A pushed limit combined with an unpushed filter would truncate too early."""
    got = (
        scan_lance(lance_uri, predicate_pushdown=False)
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
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    from polars_pylance import LanceScanOptions

    options = LanceScanOptions(batch_size=1_234, io_buffer_size=8 * 1024 * 1024)
    scan_lance(lance_uri, options=options).select("id").collect(engine="streaming")
    calls = _scan_calls(scanner_calls)
    assert calls
    assert all(c["io_buffer_size"] == 8 * 1024 * 1024 for c in calls)
    # There is no engine batch-size hint on this path, so our option is used.
    assert all(c["batch_size"] == 1_234 for c in calls)


# ---------------------------------------------------------------------------
# io_plugin path: the whole predicate arrives, so more of it reaches Lance
# ---------------------------------------------------------------------------

IMPLS: tuple[Literal["provider", "io_plugin"], ...] = ("provider", "io_plugin")

# Predicates Polars cannot lower to PyArrow, so the provider path pushes
# nothing and the io_plugin path pushes Lance SQL.
BEYOND_PYARROW: list[pl.Expr] = [
    pl.col("cat").str.starts_with("b"),
    pl.col("id").is_in([1, 2, 3]),
    (pl.col("id") % 2) == 0,
    pl.col("val").abs() > 0.5,
]


@pytest.mark.parametrize("predicate", BEYOND_PYARROW)
def test_io_plugin_pushes_what_polars_cannot_lower(
    predicate: pl.Expr, lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, impl="io_plugin").filter(predicate).select(pl.len()).collect(
        engine="streaming"
    )
    pushed = [c["filter"] for c in _scan_calls(scanner_calls) if c.get("filter")]
    assert pushed, "nothing reached Lance"
    assert all(isinstance(f, str) for f in pushed), "expected a Lance SQL filter"


@pytest.mark.parametrize("predicate", BEYOND_PYARROW)
def test_provider_pushes_none_of_it(
    predicate: pl.Expr, lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    """The measurement behind `impl=`: Polars' own lowering declines these."""
    scan_lance(lance_uri, impl="provider").filter(predicate).select(pl.len()).collect(
        engine="streaming"
    )
    assert all(c.get("filter") is None for c in scanner_calls)


@pytest.mark.parametrize("predicate", BEYOND_PYARROW)
def test_both_paths_agree(
    predicate: pl.Expr, lance_uri: str, expected: pl.DataFrame
) -> None:
    want = expected.filter(predicate).sort("id")["id"].to_list()
    for impl in IMPLS:
        got = (
            scan_lance(lance_uri, impl=impl)
            .filter(predicate)
            .sort("id")
            .select("id")
            .collect(engine="streaming")["id"]
            .to_list()
        )
        assert got == want, impl


def test_io_plugin_pushes_only_the_translatable_conjunct(
    lance_uri: str, expected: pl.DataFrame, scanner_calls: list[dict[str, Any]]
) -> None:
    """A predicate that is half-translatable still narrows the scan."""
    predicate = pl.col("cat").str.strip_chars() == "b"
    got = (
        scan_lance(lance_uri, impl="io_plugin")
        .filter(predicate & (pl.col("val") > 0.5))
        .select(pl.len())
        .collect(engine="streaming")
    )
    assert got.item() == expected.filter(predicate & (pl.col("val") > 0.5)).height
    pushed = [c["filter"] for c in _scan_calls(scanner_calls) if c.get("filter")]
    assert pushed and all("val" in f and "strip" not in f for f in pushed)


def test_io_plugin_pushdown_can_be_disabled(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, impl="io_plugin", predicate_pushdown=False).filter(
        pl.col("cat").str.starts_with("b")
    ).select(pl.len()).collect(engine="streaming")
    assert all(c.get("filter") is None for c in scanner_calls)


def test_io_plugin_pushes_the_limit(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    """The old io-plugin implementation kept the limit in Python; this one does not."""
    scan_lance(lance_uri, impl="io_plugin").select("id").head(5).collect(
        engine="streaming"
    )
    assert 5 in [c.get("limit") for c in _scan_calls(scanner_calls)]


def test_io_plugin_stops_early(lance_uri: str, frames_yielded: list[int]) -> None:
    scan_lance(lance_uri, impl="io_plugin").select("id").head(5).collect(
        engine="streaming"
    )
    assert frames_yielded[0] == 1


def test_a_filter_lance_refuses_is_dropped_not_raised(
    lance_uri: str, expected: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lance rejecting a filter must cost speed, never correctness.

    The lowering is not schema-aware, so it can emit SQL that Lance refuses to
    plan (a literal outside the column's type, say). The predicate is applied in
    the engine regardless, so the scan can simply drop the hint.
    """
    from polars_pylance import _scan
    from polars_pylance._predicate import LanceFilter

    monkeypatch.setattr(
        _scan,
        "to_lance_filter",
        lambda predicate, **_: LanceFilter(sql="no_such_fn(1)", exact=False),
    )
    with pytest.warns(RuntimeWarning, match="rejected the pushed-down filter"):
        got = (
            scan_lance(lance_uri, impl="io_plugin")
            .filter(pl.col("cat") == "b")
            .select(pl.len())
            .collect(engine="streaming")
        )
    assert got.item() == expected.filter(pl.col("cat") == "b").height


def test_unknown_impl_is_rejected(lance_uri: str) -> None:
    with pytest.raises(ValueError, match="unknown scan impl"):
        scan_lance(lance_uri, impl="nonesuch")  # type: ignore[arg-type]
