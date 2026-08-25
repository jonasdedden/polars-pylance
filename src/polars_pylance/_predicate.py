"""Lower Polars predicates into Lance SQL filter strings.

Polars hands an IO plugin the *whole* predicate as a :class:`polars.Expr`, and
hands the dataset provider only the part it could itself translate into a
PyArrow expression -- a much narrower language than Lance's SQL filter (no
string functions, no arithmetic, no ``is_in``, no temporal parts, no list or
struct access). This module closes that gap: it walks the serialized expression
tree and emits the Lance equivalent, so those predicates reach the scanner
instead of being evaluated on batches Lance already decoded.

Two properties make it safe to be aggressive:

*Relaxation.* A lowering may return a **superset** filter -- one that keeps every
row the Polars predicate would keep, and possibly more. An untranslatable
conjunct of an ``AND`` is simply dropped, so ``a > 5 & b.str.contains("x")``
still gets ``a > 5`` pushed even before the string half is understood. That is
only sound in positive position: a dropped conjunct under ``NOT``, or a dropped
branch of an ``OR``, would remove rows, so those cases decline instead. Every
lowering therefore carries an :attr:`~LanceFilter.exact` flag saying whether it
is equivalent or merely a superset.

*Re-application.* The caller is expected to leave the original predicate with
the Polars engine (see :mod:`polars_pylance._scan`). The SQL filter exists to let
Lance skip pages, use scalar indices and defer materialising wide columns --
never to establish correctness. A wrong translation would still be a bug, but a
*missing* one only costs speed.

Where SQL and Polars disagree, the lowering declines rather than guesses:
``Time`` literals (Lance has no time type), ``xor`` (Lance rejects boolean
operands), ``dt.weekday`` (1..7 in Polars, 0..6 in SQL), non-strict casts, and
anything whose spelling is not in the tested allowlist.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

# Columns Lance synthesises during the scan. They do not exist while the filter
# is evaluated, so a predicate touching one cannot be pushed.
VIRTUAL_COLUMNS = frozenset(
    {"_rowid", "_rowaddr", "_distance", "_score", "query_index"}
)

# Beyond this many elements an `IN` list stops being worth the SQL round trip;
# the conjunct is dropped and Polars evaluates it instead.
MAX_IN_LIST = 4096

_COMPARISONS = {
    "Eq": "=",
    "NotEq": "!=",
    "Lt": "<",
    "LtEq": "<=",
    "Gt": ">",
    "GtEq": ">=",
}
# `eq_missing` / `ne_missing`. Null-safe against a non-null literal is plain
# equality; against anything else Lance has no spelling (`IS DISTINCT FROM` is
# rejected), so those decline.
_NULL_SAFE = {"EqValidity": "=", "NotEqValidity": "!="}

_ARITHMETIC = {"Plus": "+", "Minus": "-", "Multiply": "*", "Modulus": "%"}

_CONJUNCTIONS = {"And": "AND", "LogicalAnd": "AND", "Or": "OR", "LogicalOr": "OR"}

# `date_part` field names that mean the same thing in both systems. Polars'
# `weekday` (Mon=1..Sun=7) and the sub-second parts do not, and are left out.
_DATE_PARTS = {
    "Year": "year",
    "Quarter": "quarter",
    "Month": "month",
    "Day": "day",
    "OrdinalDay": "doy",
    "Week": "week",
    "Hour": "hour",
    "Minute": "minute",
    "Second": "second",
}

_CAST_TYPES = {
    "Int8": "tinyint",
    "Int16": "smallint",
    "Int32": "int",
    "Int64": "bigint",
    "Float32": "float",
    "Float64": "double",
    "String": "string",
    "Boolean": "boolean",
    "Date": "date",
}


class _Decline(Exception):
    """Raised internally when a node has no Lance equivalent."""


@dataclass(frozen=True)
class LanceFilter:
    """A Lance SQL filter lowered from a Polars predicate.

    Attributes
    ----------
    sql
        The filter string, ready for ``LanceDataset.scanner(filter=...)``.
    exact
        True when the filter keeps exactly the rows the Polars predicate keeps.
        False when it is a superset, because part of the predicate was dropped;
        the caller must then keep evaluating the original predicate.
    """

    sql: str
    exact: bool


@dataclass(frozen=True)
class _Value:
    """A lowered value-position expression."""

    sql: str
    # Float literals need the other side of a comparison cast: Lance rejects
    # `int_col > 1.5` outright rather than coercing, the way Polars does.
    is_float_literal: bool = False
    # Set when the expression is already floating point, so the cast can be
    # skipped rather than nested.
    is_double: bool = False

    def as_double(self) -> _Value:
        if self.is_float_literal or self.is_double:
            return self
        return _Value(f"CAST({self.sql} AS double)", is_double=True)


def to_lance_filter(
    predicate: pl.Expr, *, max_in_list: int = MAX_IN_LIST
) -> LanceFilter | None:
    """Lower `predicate` to a Lance SQL filter, or None if nothing can be pushed.

    Parameters
    ----------
    predicate
        Any boolean Polars expression, however deeply nested.
    max_in_list
        Largest ``is_in`` membership list to spell out as SQL ``IN``.

    Examples
    --------
    >>> to_lance_filter(pl.col("cat").str.starts_with("b"))
    LanceFilter(sql="starts_with(`cat`, 'b')", exact=True)
    >>> to_lance_filter(
    ...     pl.col("cat").str.extract(r"(\\d+)").is_null() & (pl.col("id") > 3)
    ... )
    LanceFilter(sql='(`id` > 3)', exact=False)
    >>> to_lance_filter(pl.col("id").hash() > 3) is None
    True
    """
    try:
        tree = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        # A predicate that cannot even be serialized (a Python UDF, say) is one
        # we could not have translated either.
        return None

    lowering = _Lowering(max_in_list=max_in_list)
    try:
        sql, exact = lowering.predicate(tree)
    except (_Decline, RecursionError):
        return None
    if sql is None:
        return None
    return LanceFilter(sql=sql, exact=exact)


class _Lowering:
    """One translation pass. Holds the knobs; carries no state between nodes."""

    def __init__(self, *, max_in_list: int) -> None:
        self.max_in_list = max_in_list

    # -- boolean position --------------------------------------------------

    def predicate(self, node: Any) -> tuple[str | None, bool]:
        """Lower a node that evaluates to a boolean.

        Returns ``(sql, exact)``. ``sql is None`` means "no constraint": the
        caller must treat it as ``TRUE`` and keep the row-level predicate. A
        node that declines never propagates out of here -- relaxation only works
        if one unlowerable branch costs that branch and nothing else.
        """
        try:
            return self._predicate(node)
        except _Decline:
            return None, False

    def _predicate(self, node: Any) -> tuple[str | None, bool]:
        kind, body = _unpack(node)

        if kind == "Alias":
            # `(pl.col("a") > 1).alias("x")` as a filter: the name is noise.
            return self.predicate(body[0])
        if kind == "BinaryExpr":
            return self._binary_predicate(body)
        if kind == "Function":
            return self._function_predicate(body)
        if kind == "Column":
            # A bare boolean column.
            return _column(body), True
        if kind == "Literal":
            value = _literal_series(node)
            if value.dtype == pl.Boolean and value.len() == 1:
                item = value.item()
                if item is None:
                    return "FALSE", True
                return ("TRUE" if item else "FALSE"), True
            return None, False
        # Ternary (when/then/otherwise) has no Lance spelling: CASE is rejected.
        return None, False

    def _binary_predicate(self, body: Any) -> tuple[str | None, bool]:
        op = body.get("op")

        if op in _CONJUNCTIONS:
            left, left_exact = self.predicate(body["left"])
            right, right_exact = self.predicate(body["right"])
            if _CONJUNCTIONS[op] == "AND":
                # Dropping a conjunct only widens the result, which the caller
                # narrows again by keeping the Polars predicate.
                if left is None:
                    return right, False
                if right is None:
                    return left, False
                return f"({left} AND {right})", left_exact and right_exact
            # A dropped OR branch would be a filter that removes rows the
            # predicate keeps, so both sides have to translate.
            if left is None or right is None:
                return None, False
            return f"({left} OR {right})", left_exact and right_exact

        if op in _COMPARISONS:
            return self._compare(_COMPARISONS[op], body["left"], body["right"])
        if op in _NULL_SAFE:
            # Null-safe equality collapses to plain equality only when the other
            # operand cannot be null.
            pairs = (
                (body["right"], body["left"]),
                (body["left"], body["right"]),
            )
            for value, other in pairs:
                if _is_non_null_literal(value):
                    return self._compare(_NULL_SAFE[op], other, value)
            return None, False
        # Xor: Lance rejects boolean operands to `=` / `!=`.
        return None, False

    def _compare(self, op: str, left: Any, right: Any) -> tuple[str | None, bool]:
        try:
            lhs = self.value(left)
            rhs = self.value(right)
        except _Decline:
            return None, False
        if lhs.is_float_literal != rhs.is_float_literal:
            # Polars compares int against float by promoting to float; Lance
            # refuses the mixed comparison, so promote it here.
            lhs, rhs = lhs.as_double(), rhs.as_double()
        return f"({lhs.sql} {op} {rhs.sql})", True

    def _function_predicate(self, body: Any) -> tuple[str | None, bool]:
        (name, payload), args = _function(body), body.get("input", [])
        if not name or not isinstance(args, list) or not args:
            return None, False

        if name == ("Boolean", "Not"):
            inner, exact = self.predicate(args[0])
            # Negating a superset gives a subset, which would drop rows.
            if inner is None or not exact:
                return None, False
            return f"(NOT {inner})", True
        if name == ("Boolean", "IsNull"):
            return self._unary(args[0], "{} IS NULL")
        if name == ("Boolean", "IsNotNull"):
            return self._unary(args[0], "{} IS NOT NULL")
        if name == ("Boolean", "IsNan"):
            # No is_nan() in Lance, but NaN is the only value unequal to itself.
            return self._unary(args[0], "{0} != {0}")
        if name == ("Boolean", "IsNotNan"):
            return self._unary(args[0], "{0} = {0}")
        if name in (("Boolean", "AllHorizontal"), ("Boolean", "AnyHorizontal")):
            return self._horizontal(name[1] == "AllHorizontal", args)
        if name == ("Boolean", "IsIn"):
            return self._is_in(args)
        if name == ("Boolean", "IsBetween"):
            return self._is_between(_options(payload), args)
        if name[0] == "StringExpr":
            return self._string_predicate(name, _options(payload), args)
        if name[0] in ("ListExpr", "ArrayExpr") and name[1:] == ("Contains",):
            return self._contains(args)
        return None, False

    def _unary(self, arg: Any, template: str) -> tuple[str | None, bool]:
        try:
            value = self.value(arg)
        except _Decline:
            return None, False
        return f"({template.format(value.sql)})", True

    def _horizontal(self, all_: bool, args: Sequence[Any]) -> tuple[str | None, bool]:
        parts: list[str] = []
        exact = True
        for arg in args:
            sql, arg_exact = self.predicate(arg)
            if sql is None:
                if all_:
                    exact = False
                    continue
                return None, False
            exact = exact and arg_exact
            parts.append(sql)
        if not parts:
            return None, False
        joined = f" {'AND' if all_ else 'OR'} ".join(parts)
        return f"({joined})", exact

    def _is_in(self, args: Sequence[Any]) -> tuple[str | None, bool]:
        if len(args) != 2:
            return None, False
        try:
            column = self.value(args[0])
            values = _literal_elements(args[1])
        except _Decline:
            return None, False
        if values.len() > self.max_in_list:
            return None, False
        if values.is_empty():
            # `is_in([])` is false everywhere; `IN ()` is a syntax error.
            return "FALSE", True
        try:
            rendered = [_scalar(v, values.dtype) for v in values]
        except _Decline:
            return None, False
        if values.dtype in (pl.Float32, pl.Float64):
            column = column.as_double()
        return f"({column.sql} IN ({', '.join(rendered)}))", True

    def _is_between(
        self, options: dict[str, Any], args: Sequence[Any]
    ) -> tuple[str | None, bool]:
        if len(args) != 3:
            return None, False
        closed = options.get("closed", "Both")
        lower = ">=" if closed in ("Both", "Left") else ">"
        upper = "<=" if closed in ("Both", "Right") else "<"
        low, low_exact = self._compare(lower, args[0], args[1])
        high, high_exact = self._compare(upper, args[0], args[2])
        if low is None or high is None:
            return None, False
        return f"({low} AND {high})", low_exact and high_exact

    def _string_predicate(
        self, path: tuple[str, ...], options: dict[str, Any], args: Sequence[Any]
    ) -> tuple[str | None, bool]:
        name = path[1] if len(path) > 1 else ""
        if name in ("StartsWith", "EndsWith") and len(args) == 2:
            try:
                column, needle = self.value(args[0]), self.value(args[1])
            except _Decline:
                return None, False
            fn = "starts_with" if name == "StartsWith" else "ends_with"
            return f"{fn}({column.sql}, {needle.sql})", True
        if name == "Contains" and len(args) == 2:
            try:
                column, needle = self.value(args[0]), self.value(args[1])
            except _Decline:
                return None, False
            if options.get("literal"):
                # Substring search, not the token search the name suggests.
                return f"contains({column.sql}, {needle.sql})", True
            # Polars and Lance both match with the Rust `regex` crate.
            return f"regexp_like({column.sql}, {needle.sql})", True
        return None, False

    def _contains(self, args: Sequence[Any]) -> tuple[str | None, bool]:
        if len(args) != 2:
            return None, False
        try:
            column, needle = self.value(args[0]), self.value(args[1])
        except _Decline:
            return None, False
        return f"array_has({column.sql}, {needle.sql})", True

    # -- value position ----------------------------------------------------

    def value(self, node: Any) -> _Value:
        """Lower a node that evaluates to a value. Raises `_Decline` if it cannot.

        Value position has no relaxed form: a value is translated exactly or not
        at all.
        """
        kind, body = _unpack(node)

        if kind == "Alias":
            return self.value(body[0])
        if kind == "Column":
            return _Value(_column(body))
        if kind == "Literal":
            return _literal(node)
        if kind == "Cast":
            return self._cast(body)
        if kind == "BinaryExpr":
            return self._arithmetic(body)
        if kind == "Function":
            return self._function_value(body)
        raise _Decline

    def _cast(self, body: Any) -> _Value:
        if body.get("options") != "Strict":
            # A non-strict Polars cast yields null where Lance would fail the
            # whole scan.
            raise _Decline
        dtype = body.get("dtype")
        name = dtype.get("Literal") if isinstance(dtype, dict) else None
        if name not in _CAST_TYPES:
            raise _Decline
        return _Value(f"CAST({self.value(body['expr']).sql} AS {_CAST_TYPES[name]})")

    def _arithmetic(self, body: Any) -> _Value:
        op = body.get("op")
        if op in _ARITHMETIC:
            left, right = self.value(body["left"]), self.value(body["right"])
            return _Value(f"({left.sql} {_ARITHMETIC[op]} {right.sql})")
        if op == "TrueDivide":
            # Polars' `/` is always float division; SQL's is integer division
            # between integers.
            left = self.value(body["left"]).as_double()
            return _Value(
                f"({left.sql} / {self.value(body['right']).sql})", is_double=True
            )
        # FloorDivide has no Lance spelling (`floor()` is rejected).
        raise _Decline

    def _function_value(self, body: Any) -> _Value:
        (name, payload), args = _function(body), body.get("input", [])
        if not name or not isinstance(args, list) or not args:
            raise _Decline

        if name == ("Abs",):
            return _Value(f"abs({self.value(args[0]).sql})")
        if name == ("Negate",):
            return _Value(f"(- {self.value(args[0]).sql})")
        if name == ("FillNull",) and len(args) == 2:
            left, right = self.value(args[0]), self.value(args[1])
            return _Value(f"coalesce({left.sql}, {right.sql})")
        # `Round` is deliberately absent: Polars breaks ties to even, Lance
        # away from zero, and nothing in the IR lets us ask for the other one.
        if name[0] == "StringExpr":
            return self._string_value(name[1] if len(name) > 1 else "", args)
        if name[0] == "TemporalExpr":
            return self._temporal_value(name[1] if len(name) > 1 else "", args)
        if name == ("StructExpr", "FieldByName") and len(args) == 1:
            if not isinstance(payload, str):
                raise _Decline
            return _Value(f"{self.value(args[0]).sql}.{_quote(payload)}")
        if name[0] in ("ListExpr", "ArrayExpr") and name[-1] == "Length":
            return _Value(f"array_length({self.value(args[0]).sql})")
        raise _Decline

    def _string_value(self, name: str, args: Sequence[Any]) -> _Value:
        column = self.value(args[0]).sql
        if name == "Lowercase":
            return _Value(f"lower({column})")
        if name == "Uppercase":
            return _Value(f"upper({column})")
        if name in ("LenChars", "LenBytes") and len(args) == 1:
            # Lance has no octet_length; byte length is only equal to character
            # length for ASCII, so only the character form is lowered.
            if name == "LenBytes":
                raise _Decline
            return _Value(f"length({column})")
        raise _Decline

    def _temporal_value(self, name: str, args: Sequence[Any]) -> _Value:
        column = self.value(args[0]).sql
        if name == "Date":
            return _Value(f"CAST({column} AS date)")
        part = _DATE_PARTS.get(name)
        if part is None:
            raise _Decline
        return _Value(f"date_part('{part}', {column})")


# ---------------------------------------------------------------------------
# tree and literal helpers
# ---------------------------------------------------------------------------


def _unpack(node: Any) -> tuple[str, Any]:
    """Split a single-key IR node into its tag and body."""
    if not isinstance(node, dict) or len(node) != 1:
        raise _Decline
    kind, body = next(iter(node.items()))
    if not isinstance(kind, str):
        raise _Decline
    return kind, body


def _function(body: Any) -> tuple[tuple[str, ...], Any]:
    """Split a `function` field into its name path and its payload.

    The IR spells a function four ways, and telling them apart is the only
    fiddly part of reading it::

        "Abs"                            -> ("Abs",),               None
        {"Boolean": "IsNull"}            -> ("Boolean", "IsNull"),  None
        {"Boolean": {"IsIn": {...}}}     -> ("Boolean", "IsIn"),    {...}
        {"Round": {"decimals": 2, ...}}  -> ("Round",),             {...}

    A namespaced name is a single key whose value is the payload; a bare
    function with options is a single key whose value *is* the options.
    """
    function = body.get("function") if isinstance(body, dict) else None
    if isinstance(function, str):
        return (function,), None
    if not isinstance(function, dict) or len(function) != 1:
        return (), None
    namespace, inner = next(iter(function.items()))
    if isinstance(inner, str):
        return (namespace, inner), None
    if isinstance(inner, dict) and len(inner) == 1:
        name, payload = next(iter(inner.items()))
        if isinstance(payload, (dict, str)):
            return (namespace, name), payload
    return (namespace,), inner


def _options(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _column(name: Any) -> str:
    if not isinstance(name, str) or not name or name in VIRTUAL_COLUMNS:
        raise _Decline
    return _quote(name)


def _quote(name: str) -> str:
    """Backtick-quote an identifier; Lance rejects the double-quoted form."""
    if "`" in name:
        raise _Decline
    return f"`{name}`"


def _literal_series(node: Any) -> pl.Series:
    """Evaluate a literal subtree back to a Series.

    Round-tripping through Polars rather than reading the IR's own encoding is
    what keeps this robust: the literal payload is a dtype-dependent mix of
    inline values and Arrow IPC blobs, and which one is used has changed
    between Polars releases.
    """
    try:
        expr = pl.Expr.deserialize(io.BytesIO(json.dumps(node).encode()), format="json")
        return pl.select(expr).to_series()
    except Exception as exc:
        raise _Decline from exc


def _literal(node: Any) -> _Value:
    series = _literal_series(node)
    if series.len() != 1:
        raise _Decline
    dtype = series.dtype
    return _Value(
        _scalar(series.item(), dtype),
        is_float_literal=dtype in (pl.Float32, pl.Float64),
    )


def _literal_elements(node: Any) -> pl.Series:
    """The membership list of an `is_in`, whatever shape the IR gave it.

    Polars spells the right-hand side either as a Series literal or as a
    one-element List literal, depending on how the caller wrote it.
    """
    series = _literal_series(node)
    if not isinstance(series.dtype, (pl.List, pl.Array)):
        return series
    if series.len() != 1:
        raise _Decline
    inner = series.item()
    if not isinstance(inner, pl.Series):
        raise _Decline
    return inner


def _scalar(value: Any, dtype: pl.DataType) -> str:
    """Render a Python value as a Lance SQL literal."""
    if value is None:
        return "NULL"
    if dtype == pl.Boolean:
        return "TRUE" if value else "FALSE"
    if dtype in (pl.String, pl.Categorical) or isinstance(
        dtype, (pl.Enum, pl.Categorical)
    ):
        return _string_literal(str(value))
    if dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        return str(int(value))
    if dtype in (pl.Float32, pl.Float64):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise _Decline
        return repr(number)
    if dtype == pl.Date:
        return f"date '{value.isoformat()}'"
    if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
        # Lance parses a bare timestamp literal against the column's own time
        # zone, so an aware literal is normalised to UTC first.
        stamp = value
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return f"timestamp '{stamp.isoformat(sep=' ')}'"
    if dtype == pl.Binary:
        return f"X'{bytes(value).hex()}'"
    # Time, Duration, Decimal, and every nested dtype: no dependable spelling.
    raise _Decline


def _string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_non_null_literal(node: Any) -> bool:
    try:
        kind, _ = _unpack(node)
    except _Decline:
        return False
    if kind != "Literal":
        return False
    try:
        series = _literal_series(node)
    except _Decline:
        return False
    return series.len() == 1 and series.item() is not None
