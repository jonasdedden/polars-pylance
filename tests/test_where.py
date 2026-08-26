"""The `where=` escape hatch: a Lance SQL filter for predicates Polars will not lower.

Polars only hands the dataset provider a predicate when it can render it as a
PyArrow expression. `is_in`, the `str.*` predicates and arithmetic are not
lowered, so without this the engine filters after Lance has read every row.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from polars_pylance import scan_lance


def test_where_reaches_lance(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    scan_lance(lance_uri, where="cat = 'b'").select("id").collect(engine="streaming")
    assert any(call.get("filter") == "cat = 'b'" for call in scanner_calls), (
        f"the SQL filter never reached Lance: {scanner_calls}"
    )


def test_where_matches_the_equivalent_polars_filter(lance_uri: str) -> None:
    """The point of the hatch is a faster route to the *same* rows."""
    via_sql = (
        scan_lance(lance_uri, where="cat IN ('a', 'b')")
        .select("id")
        .collect(engine="streaming")
        .sort("id")
    )
    via_polars = (
        scan_lance(lance_uri)
        .filter(pl.col("cat").is_in(["a", "b"]))
        .select("id")
        .collect(engine="streaming")
        .sort("id")
    )
    assert via_sql.equals(via_polars)
    assert via_sql.height > 0


def test_where_composes_with_a_polars_filter(lance_uri: str) -> None:
    """Both filters apply; they are conjunctive, so order does not change the answer."""
    out = (
        scan_lance(lance_uri, where="cat IN ('a', 'b')")
        .filter(pl.col("val") > 0.5)
        .select("cat", "val")
        .collect(engine="streaming")
    )
    assert out.height > 0
    assert set(out["cat"].unique().to_list()) <= {"a", "b"}
    assert (out["val"] > 0.5).all()


def test_where_takes_precedence_over_the_pushed_predicate(
    lance_uri: str, scanner_calls: list[dict[str, Any]]
) -> None:
    """Only one filter reaches Lance; Polars re-applies its own on the result."""
    out = (
        scan_lance(lance_uri, where="cat = 'a'")
        .filter(pl.col("cat") == "b")
        .select("id")
        .collect(engine="streaming")
    )
    # 'a' AND 'b' is empty, which is exactly what both filters applying means
    assert out.height == 0
    filters = [c.get("filter") for c in scanner_calls if c.get("filter") is not None]
    assert filters and all(f == "cat = 'a'" for f in filters), filters


def test_invalid_where_is_reported(lance_uri: str) -> None:
    with pytest.raises(Exception, match=r"(?i)column|field|parse|schema|no_such"):
        scan_lance(lance_uri, where="no_such_column = 1").select("id").collect(
            engine="streaming"
        )


def test_where_survives_serialization(lance_uri: str) -> None:
    """It has to reach the workers, so it must be part of the picklable spec."""
    import pickle

    from polars_pylance import LanceScanSpec

    spec = LanceScanSpec(uri=lance_uri, where="cat = 'a'")
    assert pickle.loads(pickle.dumps(spec)).where == "cat = 'a'"
