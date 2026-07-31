"""Unit tests for the Polars-expression -> Lance SQL translator."""

from __future__ import annotations

import json

import polars as pl
import pytest

from polars_lance import to_lance_filter
from polars_lance._predicate import prune_unevaluable

TRANSLATABLE = [
    (pl.col("cat") == "b", "(cat = 'b')"),
    (pl.col("cat") != "b", "(cat != 'b')"),
    (pl.col("id") >= 10, "(id >= 10)"),
    (pl.col("id") < -3, "(id < -3)"),
    (pl.col("val") > 0.9, "(val > 0.9)"),
    (pl.col("flag") == True, "(flag = TRUE)"),  # noqa: E712
    (pl.col("cat").is_null(), "(cat IS NULL)"),
    (pl.col("cat").is_not_null(), "(cat IS NOT NULL)"),
    (~(pl.col("cat") == "a"), "(NOT (cat = 'a'))"),
    (
        (pl.col("cat") == "b") & (pl.col("val") > 0.9),
        "((cat = 'b') AND (val > 0.9))",
    ),
    (
        (pl.col("cat") == "a") | (pl.col("val") <= 0.1),
        "((cat = 'a') OR (val <= 0.1))",
    ),
    (
        (pl.col("id") >= 10) & (pl.col("id") < 20) & (pl.col("cat") != "d"),
        "(((id >= 10) AND (id < 20)) AND (cat != 'd'))",
    ),
]


@pytest.mark.parametrize(("expr", "sql"), TRANSLATABLE)
def test_translatable(expr: pl.Expr, sql: str) -> None:
    assert to_lance_filter(expr) == sql


def test_quotes_are_escaped() -> None:
    assert to_lance_filter(pl.col("cat") == "O'Brien") == "(cat = 'O''Brien')"


DECLINED = [
    pl.col("cat").str.starts_with("a"),  # string functions
    pl.col("cat").is_in(["a", "b"]),  # IPC-encoded list literal
    pl.col("val").is_between(0.1, 0.2),  # lowered to an unsupported function
    pl.col("id") + 1 == 5,  # arithmetic
    pl.col("val").sum() > 1,  # aggregation
    pl.col("cat"),  # bare column, not a boolean expression
    pl.col("id") == pl.lit(None),  # null literal
    pl.col("val") > float("inf"),  # unrepresentable literal
]


@pytest.mark.parametrize("expr", DECLINED)
def test_declined(expr: pl.Expr) -> None:
    assert to_lance_filter(expr) is None


def test_unsafe_identifier_declined() -> None:
    # Quoting rules are dialect-specific, so odd names are not translated.
    assert to_lance_filter(pl.col("weird name") == 1) is None
    assert to_lance_filter(pl.col("a.b") == 1) is None


def test_virtual_columns_declined() -> None:
    assert to_lance_filter(pl.col("_rowid") == 1) is None
    assert to_lance_filter(pl.col("_distance") < 0.5) is None


# -- engine-internal placeholders -------------------------------------------
#
# When Polars pushes a top-k (sort + head) into a scan it injects an opaque
# `dynamic_pred` node into the predicate handed to an IO plugin. It cannot be
# evaluated from Python by any route -- DataFrame.filter, lazy().filter, select
# and with_columns all panic with "internal error: entered unreachable code" --
# so it has to be stripped before use.

DYNAMIC = {
    "Display": {
        "inputs": [{"Column": "id"}],
        "fmt_str": "dynamic_pred: 0f9c2b1e-0000-4000-8000-000000000000",
    }
}
CAT_EQ_C = {
    "BinaryExpr": {
        "left": {"Column": "cat"},
        "op": "Eq",
        "right": {"Literal": {"Scalar": {"String": "c"}}},
    }
}


def _expr(tree: dict) -> pl.Expr:
    return pl.Expr.deserialize(json.dumps(tree).encode(), format="json")


def _and(left: dict, right: dict) -> dict:
    return {"BinaryExpr": {"left": left, "op": "And", "right": right}}


def test_dynamic_conjunct_is_stripped() -> None:
    pruned = prune_unevaluable(_expr(_and(CAT_EQ_C, DYNAMIC)))
    assert pruned is not None
    frame = pl.DataFrame({"id": [1, 2, 3], "cat": ["c", "d", "c"]})
    assert frame.filter(pruned).height == 2


def test_dynamic_only_predicate_becomes_none() -> None:
    assert prune_unevaluable(_expr(DYNAMIC)) is None


def test_ordinary_predicate_is_returned_unchanged() -> None:
    expr = pl.col("cat") == "c"
    assert prune_unevaluable(expr) is expr


def test_dynamic_under_or_is_refused() -> None:
    """Weakening a disjunct is not the same as dropping a conjunct: fail loudly."""
    tree = {"BinaryExpr": {"left": CAT_EQ_C, "op": "Or", "right": DYNAMIC}}
    with pytest.raises(RuntimeError, match="cannot be safely ignored"):
        prune_unevaluable(_expr(tree))
