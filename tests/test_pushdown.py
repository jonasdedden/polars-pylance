"""What actually reaches Lance. These are the tests that would catch a silent
regression to "read everything, filter in Polars"."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from polars_pylance import scan_lance


def _provider_is_handed_the_whole_predicate() -> bool:
    """Whether this Polars passes a dataset provider its serialized predicate.

    Polars passes only its own PyArrow lowering today, so the provider path
    pushes nothing for a predicate outside that subset. A Polars that also
    passes the expression lets the provider lower it as widely as the io_plugin
    path does -- see docs/PREDICATE_PUSHDOWN.md -- which flips the expected
    answer of several tests below.
    """
    from polars._plr import PyLazyFrame
    from polars._utils.wrap import wrap_ldf

    seen: set[str] = set()
    schema = pa.schema([pa.field("a", pa.int64())])

    class Probe:
        def schema(self) -> pa.Schema:
            return schema

        def to_dataset_scan(self, **kwargs: Any) -> tuple[pl.LazyFrame, str]:
            seen.update(kwargs)

            def impl(*_args: Any, **_kwargs: Any) -> tuple[Any, bool]:
                return iter([pl.DataFrame({"a": [1]})]), False

            lf = pl.LazyFrame._scan_python_function(
                schema, impl, pyarrow=True, is_pure=True
            )
            return lf, "v1"

    lf = wrap_ldf(PyLazyFrame.new_from_dataset_object(Probe()))
    # A predicate with no PyArrow lowering, so only the new argument can carry it.
    lf.filter(pl.col("a").cast(pl.String).str.starts_with("x")).collect(
        engine="streaming"
    )
    return "serialized_predicate" in seen


PROVIDER_LOWERS_THE_PREDICATE = _provider_is_handed_the_whole_predicate()

needs_pyarrow_only_provider = pytest.mark.skipif(
    PROVIDER_LOWERS_THE_PREDICATE,
    reason="this Polars hands the provider the whole predicate, so it pushes more",
)


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
    # exactly what Lance's scanner accepts -- no translation in between. Where
    # it also hands over the predicate itself, the exact SQL lowering wins.
    expected = str if PROVIDER_LOWERS_THE_PREDICATE else pc.Expression
    assert all(isinstance(f, expected) for f in filters)


def test_predicate_pushdown_can_be_disabled(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, predicate_pushdown=False).filter(pl.col("cat") == "b").select(
        "id"
    ).collect(engine="streaming")
    assert all(c.get("filter") is None for c in scanner_calls)


@needs_pyarrow_only_provider
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


@needs_pyarrow_only_provider
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


# ---------------------------------------------------------------------------
# provider path, given a predicate Polars does not pass today
# ---------------------------------------------------------------------------


def _provider_scan(uri: str, predicate: pl.Expr | None, **kwargs: Any) -> str | None:
    """Resolve a provider scan by hand and report the filter it would push.

    Stock Polars only ever passes `pyarrow_predicate`. Calling the interface
    directly is how the *other* argument gets exercised until a Polars ships
    that offers it -- see docs/PREDICATE_PUSHDOWN.md.
    """
    from polars_pylance import LanceDatasetProvider, LanceScanSpec

    provider = LanceDatasetProvider(LanceScanSpec(uri=uri, **kwargs))
    serialized = None if predicate is None else predicate.meta.serialize()
    seen: list[Any] = []
    original = LanceScanSpec.iter_frames

    def spy(self: Any, dataset: Any, **kw: Any) -> Any:
        seen.append(kw.get("filter"))
        return original(self, dataset, **kw)

    LanceScanSpec.iter_frames = spy  # type: ignore[method-assign]
    try:
        # No projection: the LazyFrame a provider returns declares the full
        # schema, and Polars, not the provider, narrows it afterwards.
        result = provider.to_dataset_scan(serialized_predicate=serialized)
        assert result is not None
        result[0].collect(engine="streaming")
    finally:
        LanceScanSpec.iter_frames = original  # type: ignore[method-assign]
    return seen[0] if seen else None


def test_provider_lowers_a_serialized_predicate(lance_uri: str) -> None:
    pushed = _provider_scan(lance_uri, pl.col("cat").str.starts_with("b"))
    assert pushed == "starts_with(`cat`, 'b')"


def test_provider_ignores_a_serialized_predicate_when_pushdown_is_off(
    lance_uri: str,
) -> None:
    pushed = _provider_scan(
        lance_uri, pl.col("cat").str.starts_with("b"), predicate_pushdown=False
    )
    assert pushed is None


def test_provider_survives_an_unreadable_serialized_predicate(lance_uri: str) -> None:
    from polars_pylance import LanceDatasetProvider, LanceScanSpec

    provider = LanceDatasetProvider(LanceScanSpec(uri=lance_uri))
    with pytest.warns(RuntimeWarning, match="could not read the predicate"):
        result = provider.to_dataset_scan(serialized_predicate=b"not an expression")
    assert result is not None


def test_provider_drops_a_filter_lance_refuses(
    lance_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lowering Lance will not plan must not take the query down with it.

    The provider path reports the predicate as unapplied, so Polars filters
    above the scan and the pushed filter is only ever an IO hint.
    """
    from polars_pylance import LanceDatasetProvider, LanceFilter, LanceScanSpec, _scan

    monkeypatch.setattr(
        _scan,
        "to_lance_filter",
        lambda predicate, **_: LanceFilter(sql="no_such_fn(1)", exact=True),
    )
    provider = LanceDatasetProvider(LanceScanSpec(uri=lance_uri))
    result = provider.to_dataset_scan(
        serialized_predicate=(pl.col("cat") == "b").meta.serialize()
    )
    assert result is not None
    with pytest.warns(RuntimeWarning, match="rejected the pushed-down filter"):
        got = result[0].collect(engine="streaming")
    assert got.height == 60_000


# ---------------------------------------------------------------------------
# the Lance filter Lance answers wrongly
# ---------------------------------------------------------------------------


def test_lance_mishandles_a_pyarrow_timestamp_filter(rich_uri: str) -> None:
    """The bug `_unsafe_for_lance` exists for, pinned so its repair is visible.

    Lance takes a PyArrow expression through Substrait, and a comparison against
    a timezone-naive timestamp column comes back with the wrong rows -- silently,
    unlike `date64` / `time64` / `duration` / `timestamp(tz=...)`, which raise.
    Measured on pylance 9.0.0 and 10.0.0.

    When this test starts failing, Lance has fixed it: drop `_unsafe_for_lance`
    and this test with it.
    """
    import datetime as dt

    import lance

    dataset = lance.dataset(rich_uri)
    cut = dt.datetime(2024, 1, 5)
    expected = dataset.scanner(
        columns=["id"], filter=f"ts > timestamp '{cut:%Y-%m-%d %H:%M:%S}'"
    ).to_table()
    through_pyarrow = dataset.scanner(
        columns=["id"], filter=pc.field("ts") > pa.scalar(cut, pa.timestamp("us"))
    ).to_table()

    assert expected.num_rows > 0
    assert through_pyarrow.num_rows == 0


def test_timestamp_predicate_is_not_pushed_as_a_pyarrow_expression(
    rich_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    import datetime as dt

    scan_lance(rich_uri).filter(pl.col("ts") > dt.datetime(2024, 1, 5)).select(
        pl.len()
    ).collect(engine="streaming")

    pushed = [call["filter"] for call in _scan_calls(scanner_calls)]
    assert not any(isinstance(f, pc.Expression) for f in pushed)


def test_timestamp_predicate_still_returns_the_right_rows(
    rich_uri: str, rich_frame: pl.DataFrame
) -> None:
    import datetime as dt

    predicate = pl.col("ts") > dt.datetime(2024, 1, 5)
    impls: tuple[Literal["provider", "io_plugin"], ...] = ("provider", "io_plugin")
    for impl in impls:
        got = (
            scan_lance(rich_uri, impl=impl)
            .filter(predicate)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        assert got == rich_frame.filter(predicate).height


def test_the_guard_leaves_the_sql_lowering_alone(rich_uri: str) -> None:
    """Only the PyArrow expression is refused; Lance's own SQL is fine with it."""
    import datetime as dt

    pushed = _provider_scan(rich_uri, pl.col("ts") > dt.datetime(2024, 1, 5))
    assert pushed is not None
    assert "ts" in pushed


def test_a_column_named_like_a_timestamp_column_still_pushes(rich_uri: str) -> None:
    """The guard matches column names in the generated source, so check a
    predicate over other columns of the same dataset is unaffected."""
    from polars_pylance import _scan

    schema = _scan.LanceScanSpec(uri=rich_uri).arrow_schema()
    assert not _scan._unsafe_for_lance("(pa.compute.field('id') > 5)", schema)
    assert _scan._unsafe_for_lance("(pa.compute.field('ts') > 5)", schema)
    # `day` is a date32, which Lance answers correctly.
    assert not _scan._unsafe_for_lance("(pa.compute.field('day') > 5)", schema)
