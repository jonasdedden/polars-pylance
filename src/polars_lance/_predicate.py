"""Translate a subset of Polars expressions into Lance SQL filter strings.

Only used by the ``register_io_source`` fallback path. The dataset-provider path
gets a ready-made PyArrow predicate from Polars itself and needs none of this.

The translation is deliberately partial: anything not on the allowlist yields
``None``, meaning "no filter pushed down". Callers must therefore *always* also
apply the original Polars predicate to each batch — this function exists to let
Lance skip pages and use scalar indices, never to establish correctness.

Every construct on the allowlist has the same three-valued semantics in
DataFusion SQL as in Polars, so a translated filter never drops a row that
Polars would have kept.
"""

from __future__ import annotations

import json
import re
from typing import Any

import polars as pl

_BINARY_OPS = {
    "Eq": "=",
    "NotEq": "!=",
    "Lt": "<",
    "LtEq": "<=",
    "Gt": ">",
    "GtEq": ">=",
    "And": "AND",
    "Or": "OR",
}

_COMPARISONS = frozenset({"Eq", "NotEq", "Lt", "LtEq", "Gt", "GtEq"})

_INT_SCALARS = frozenset(
    {"Int8", "Int16", "Int32", "Int64", "Int128", "UInt8", "UInt16", "UInt32", "UInt64"}
)
_FLOAT_SCALARS = frozenset({"Float32", "Float64"})

# Unquoted-safe identifier. Anything else (spaces, dots, unicode, leading digit)
# would need dialect-specific quoting, so we decline to translate instead.
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Lance-generated columns: not filterable in the SQL predicate.
_VIRTUAL_COLUMNS = frozenset(
    {"_rowid", "_rowaddr", "_distance", "_score", "query_index"}
)


class _Untranslatable(Exception):
    """Raised internally when a node falls outside the allowlist."""


def to_lance_filter(predicate: pl.Expr) -> str | None:
    """Return an SQL filter for `predicate`, or None if it cannot be translated.

    Examples
    --------
    >>> to_lance_filter((pl.col("cat") == "b") & (pl.col("val") > 0.9))
    "(cat = 'b') AND (val > 0.9)"
    >>> to_lance_filter(pl.col("cat").str.starts_with("a")) is None
    True
    """
    try:
        tree = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        return None

    try:
        return _boolean(tree)
    except _Untranslatable:
        return None
    except RecursionError:
        return None


def _boolean(node: Any) -> str:
    """Translate a node that must evaluate to a boolean."""
    if not isinstance(node, dict) or len(node) != 1:
        raise _Untranslatable
    kind, body = next(iter(node.items()))

    if kind == "BinaryExpr":
        op = body.get("op")
        if op in _COMPARISONS:
            left = _value(body["left"])
            right = _value(body["right"])
            return f"({left} {_BINARY_OPS[op]} {right})"
        if op in ("And", "Or"):
            left = _boolean(body["left"])
            right = _boolean(body["right"])
            return f"({left} {_BINARY_OPS[op]} {right})"
        raise _Untranslatable

    if kind == "Function":
        name = _boolean_function_name(body)
        (inner,) = body["input"] if len(body["input"]) == 1 else (None,)
        if inner is None:
            raise _Untranslatable
        if name == "IsNull":
            return f"({_value(inner)} IS NULL)"
        if name == "IsNotNull":
            return f"({_value(inner)} IS NOT NULL)"
        if name == "Not":
            return f"(NOT {_boolean(inner)})"
        raise _Untranslatable

    raise _Untranslatable


def _boolean_function_name(body: Any) -> str:
    function = body.get("function")
    if not isinstance(function, dict) or len(function) != 1:
        raise _Untranslatable
    namespace, name = next(iter(function.items()))
    if namespace != "Boolean" or not isinstance(name, str):
        raise _Untranslatable
    return name


def _value(node: Any) -> str:
    """Translate a node in value position (a column or a literal)."""
    if not isinstance(node, dict) or len(node) != 1:
        raise _Untranslatable
    kind, body = next(iter(node.items()))

    if kind == "Column":
        return _column(body)
    if kind == "Literal":
        return _literal(body)
    raise _Untranslatable


def _column(name: Any) -> str:
    if (
        not isinstance(name, str)
        or not _SAFE_IDENT.match(name)
        or name in _VIRTUAL_COLUMNS
    ):
        raise _Untranslatable
    return name


def _literal(body: Any) -> str:
    if not isinstance(body, dict) or len(body) != 1:
        raise _Untranslatable
    wrapper, inner = next(iter(body.items()))
    # "Scalar" carries a concrete dtype; "Dyn" carries an as-yet-untyped literal.
    if wrapper not in ("Scalar", "Dyn") or not isinstance(inner, dict):
        raise _Untranslatable
    if len(inner) != 1:
        raise _Untranslatable

    dtype, value = next(iter(inner.items()))

    if dtype in ("String", "Str") and isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if dtype == "Boolean" and isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if (dtype in _INT_SCALARS or dtype == "Int") and isinstance(value, int):
        # bool is an int subclass; it must not fall through to here.
        if isinstance(value, bool):
            raise _Untranslatable
        return str(value)
    if (dtype in _FLOAT_SCALARS or dtype == "Float") and isinstance(
        value, (int, float)
    ):
        if isinstance(value, bool):
            raise _Untranslatable
        text = repr(float(value))
        if text in ("inf", "-inf", "nan"):
            raise _Untranslatable
        return text

    # Temporal, binary, nested, decimal and null literals are all declined: the
    # SQL spelling is dialect-sensitive and getting it wrong risks dropping rows.
    raise _Untranslatable
