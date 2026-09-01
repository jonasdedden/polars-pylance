"""Lower Polars predicates into Lance SQL filter strings.

An IO plugin is handed the whole predicate as a `polars.Expr`. Polars'
own lowering, behind `scan_pyarrow_dataset` and `scan_delta`, produces a
PyArrow expression instead and drops what will not fit, which is most of the
language. This module walks the serialized expression tree and emits the Lance
equivalent.

A lowering may be a superset: an untranslatable conjunct of an `AND` is dropped, so `a >
5 & b.str.contains("x")` still pushes `a > 5`. That is only sound in positive position,
so a dropped conjunct under `NOT` or a dropped branch of an `OR` declines instead.
[`LanceFilter.exact`][polars_pylance.LanceFilter.exact] says which of the two the caller
holds; a relaxed filter narrows the read and `polars_pylance._scan` still evaluates the
predicate, while an exact one decides the answer on its own.

Constructs whose SQL meaning differs decline rather than guess; each decline is
pinned by a test.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

# Lance synthesises these during the scan, so a filter cannot reference them.
VIRTUAL_COLUMNS = frozenset(
    {"_rowid", "_rowaddr", "_distance", "_score", "query_index"}
)

# Past this many elements an `IN` list stops being worth the round trip.
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
# equality; Lance has no spelling for the rest.
_NULL_SAFE = {"EqValidity": "=", "NotEqValidity": "!="}

_ARITHMETIC = {"Plus": "+", "Minus": "-", "Multiply": "*", "Modulus": "%"}

_CONJUNCTIONS = {"And": "AND", "LogicalAnd": "AND", "Or": "OR", "LogicalOr": "OR"}

# `date_part` fields that mean the same in both. Sub-second parts do not;
# `weekday` is renumbered in `_temporal_value`.
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

# `dt.truncate` windows `date_trunc` can express. Polars also accepts multiples
# ("2d"), which decline.
_TRUNCATE_UNITS = {
    "1s": "second",
    "1m": "minute",
    "1h": "hour",
    "1d": "day",
    "1w": "week",
    "1mo": "month",
    "3mo": "quarter",
    "1q": "quarter",
    "1y": "year",
}

# Narrower integers are absent on purpose: Polars raises on a value that does
# not fit and Lance wraps.
_CAST_TYPES = {
    "Int64": "bigint",
    "Float32": "float",
    "Float64": "double",
    "String": "string",
    "Boolean": "boolean",
    "Date": "date",
}

# A namespace and a bare function's options are both single-key objects, so
# `{"ListExpr": {"Get": false}}` and `{"Round": {...}}` are told apart by name.
_NAMESPACES = frozenset(
    {
        "Boolean",
        "StringExpr",
        "TemporalExpr",
        "ListExpr",
        "ArrayExpr",
        "StructExpr",
        "Pow",
    }
)

# No literal syntax for these, but a cast of the spelled-out name parses.
_POS_INF = "CAST('inf' AS double)"
_NEG_INF = "CAST('-inf' AS double)"


# One node of Polars' serialized expression IR, as `json.loads` hands it over.
# The IR is versioned by Polars and its shape has changed between releases, so
# the walk below narrows every node before reading it rather than assuming.
Json: TypeAlias = "str | int | float | bool | list[Json] | dict[str, Json] | None"


class _Decline(Exception):
    """Raised internally when a node has no Lance equivalent."""


@dataclass(frozen=True)
class LanceFilter:
    """A Lance SQL filter lowered from a Polars predicate.

    Attributes:
        sql: The filter string, ready for `LanceDataset.scanner(filter=...)`.
        exact: True when the filter keeps exactly the rows the predicate keeps. False
            when part of the predicate was dropped, leaving a superset; the caller must
            then keep evaluating the predicate.
    """

    sql: str
    exact: bool


@dataclass(frozen=True)
class _Value:
    """A lowered value-position expression."""

    sql: str
    # Lance rejects `int_col > 1.5` rather than coercing, so a float literal
    # needs the other side cast.
    is_float_literal: bool = False
    # Already floating point, so that cast would be a no-op.
    is_double: bool = False
    # Known to be text, which is how a `Plus` node is told from concatenation.
    is_string: bool = False

    def as_double(self) -> _Value:
        if self.is_float_literal or self.is_double:
            return self
        return _Value(f"CAST({self.sql} AS double)", is_double=True)


def to_lance_filter(
    predicate: pl.Expr,
    *,
    max_in_list: int = MAX_IN_LIST,
    schema: pl.Schema | None = None,
) -> LanceFilter | None:
    r"""Lower `predicate` to a Lance SQL filter, or None if nothing can be pushed.

    Args:
        predicate: Any boolean Polars expression, however deeply nested.
        max_in_list: Largest `is_in` membership list to spell out as SQL `IN`.
        schema: The scanned schema, when the caller has it. Used to drop a promotion the
            schema shows is a no-op, since a `CAST` around an indexed column costs its
            scalar index, and to tell a string `+` from an arithmetic one.

    Examples:
        >>> import polars as pl
        >>> from polars_pylance import to_lance_filter
        >>> to_lance_filter(pl.col("cat").str.starts_with("b"))
        LanceFilter(sql="starts_with(`cat`, 'b')", exact=True)
        >>> to_lance_filter(
        ...     pl.col("cat").str.extract(r"(\d+)").is_null() & (pl.col("id") > 3)
        ... )
        LanceFilter(sql='(`id` > 3)', exact=False)
        >>> to_lance_filter(pl.col("id").hash() > 3) is None
        True
    """
    try:
        tree = json.loads(predicate.meta.serialize(format="json"))
    except Exception:  # noqa: BLE001 - see below
        # No tree means nothing to lower, and pushdown is optional, so this
        # declines rather than reaching the caller. The failure this is known to
        # catch is a `ComputeError` from a UDF closing over something
        # unpicklable; the family polars raises here is not documented, and a
        # wrong guess would turn a missed optimization into a failed query. A
        # plain UDF does serialize: it is declined by the walk, as an
        # `AnonymousFunction` node it has no spelling for.
        return None

    lowering = _Lowering(max_in_list=max_in_list, schema=schema)
    try:
        sql, exact = lowering.predicate(tree)
    except (_Decline, RecursionError):
        return None
    if sql is None:
        return None
    return LanceFilter(sql=sql, exact=exact)


class _Lowering:
    """One translation pass. Holds the knobs; carries no state between nodes."""

    def __init__(self, *, max_in_list: int, schema: pl.Schema | None = None) -> None:
        self.max_in_list = max_in_list
        self.schema = schema

    # -- boolean position --------------------------------------------------

    def predicate(self, node: Json) -> tuple[str | None, bool]:
        """Lower a boolean node to `(sql, exact)`.

        `sql is None` means no constraint. Declines are swallowed here so that
        one unlowerable branch costs only that branch.
        """
        try:
            return self._predicate(node)
        except _Decline:
            return None, False

    def _predicate(self, node: Json) -> tuple[str | None, bool]:
        kind, body = _unpack(node)

        if kind == "Alias":
            # `(pl.col("a") > 1).alias("x")` as a filter: the name is noise.
            return self.predicate(_inputs(body)[0])
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

    def _binary_predicate(self, node: Json) -> tuple[str | None, bool]:
        body = _fields(node)
        op = body.get("op")
        if not isinstance(op, str):
            return None, False

        if op in _CONJUNCTIONS:
            left, left_exact = self.predicate(_field(body, "left"))
            right, right_exact = self.predicate(_field(body, "right"))
            if _CONJUNCTIONS[op] == "AND":
                # Dropping a conjunct only widens the result.
                if left is None:
                    return right, False
                if right is None:
                    return left, False
                return f"({left} AND {right})", left_exact and right_exact
            # A dropped OR branch would remove rows, so both sides must lower.
            if left is None or right is None:
                return None, False
            return f"({left} OR {right})", left_exact and right_exact

        if op in _COMPARISONS:
            return self._compare(
                _COMPARISONS[op], _field(body, "left"), _field(body, "right")
            )
        if op in _NULL_SAFE:
            # Collapses to plain equality only against a non-null operand.
            left_node, right_node = _field(body, "left"), _field(body, "right")
            pairs = ((right_node, left_node), (left_node, right_node))
            for value, other in pairs:
                if _is_non_null_literal(value):
                    return self._compare(_NULL_SAFE[op], other, value)
            return None, False

        if op == "Xor":
            # Lance rejects boolean operands to `!=`, so expand it. Both halves
            # must be exact, since the expansion negates each of them.
            left, left_exact = self.predicate(_field(body, "left"))
            right, right_exact = self.predicate(_field(body, "right"))
            if left is None or right is None or not (left_exact and right_exact):
                return None, False
            return (
                f"(({left} AND NOT {right}) OR (NOT {left} AND {right}))",
                True,
            )
        return None, False

    def _compare(self, op: str, left: Json, right: Json) -> tuple[str | None, bool]:
        try:
            lhs = self.value(left)
            rhs = self.value(right)
        except _Decline:
            return None, False
        if lhs.is_float_literal != rhs.is_float_literal:
            # Lance refuses the mixed comparison Polars promotes through.
            lhs, rhs = lhs.as_double(), rhs.as_double()
        return f"({lhs.sql} {op} {rhs.sql})", True

    def _function_predicate(self, node: Json) -> tuple[str | None, bool]:
        body = _fields(node)
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
            # Not `x != x`: Lance compares by SQL's total ordering, under which
            # NaN equals itself, so that spelling drops every NaN row.
            return self._unary(args[0], "isnan({})")
        if name == ("Boolean", "IsNotNan"):
            return self._unary(args[0], "NOT isnan({})")
        if name == ("Boolean", "IsInfinite"):
            return self._unary(args[0], f"{{0}} = {_POS_INF} OR {{0}} = {_NEG_INF}")
        if name == ("Boolean", "IsFinite"):
            return self._unary(
                args[0],
                f"NOT isnan({{0}}) AND {{0}} != {_POS_INF} AND {{0}} != {_NEG_INF}",
            )
        if name in (("Boolean", "AllHorizontal"), ("Boolean", "AnyHorizontal")):
            return self._horizontal(args, all_=name[1] == "AllHorizontal")
        if name == ("Boolean", "IsIn"):
            return self._is_in(_options(payload), args)
        if name == ("Boolean", "IsBetween"):
            return self._is_between(_options(payload), args)
        if name[0] == "StringExpr":
            return self._string_predicate(name, _options(payload), args)
        if name[0] in ("ListExpr", "ArrayExpr") and name[1:] == ("Contains",):
            return self._contains(args)
        return None, False

    def _unary(self, arg: Json, template: str) -> tuple[str | None, bool]:
        try:
            value = self.value(arg)
        except _Decline:
            return None, False
        return f"({template.format(value.sql)})", True

    def _horizontal(
        self, args: Sequence[Json], *, all_: bool
    ) -> tuple[str | None, bool]:
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

    def _is_in(
        self, options: dict[str, Json], args: Sequence[Json]
    ) -> tuple[str | None, bool]:
        if len(args) != 2:
            return None, False
        if options.get("nulls_equal"):
            # `IN` propagates NULL, so null matching null has no spelling.
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
        if isinstance(values.dtype, (pl.Float32, pl.Float64)):
            column = column.as_double()
        return f"({column.sql} IN ({', '.join(rendered)}))", True

    def _is_between(
        self, options: dict[str, Json], args: Sequence[Json]
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
        self, path: tuple[str, ...], options: dict[str, Json], args: Sequence[Json]
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
        if name == "ContainsAny" and len(args) == 2:
            return self._contains_any(options, args)
        return None, False

    def _contains_any(
        self, options: dict[str, Json], args: Sequence[Json]
    ) -> tuple[str | None, bool]:
        """`str.contains_any` as a disjunction of substring tests.

        The case-insensitive form declines: Polars folds ASCII only.
        """
        if options.get("ascii_case_insensitive"):
            return None, False
        try:
            column = self.value(args[0])
            patterns = _literal_elements(args[1])
        except _Decline:
            return None, False
        if patterns.is_empty() or patterns.len() > self.max_in_list:
            return None, False
        try:
            rendered = [_scalar(v, patterns.dtype) for v in patterns]
        except _Decline:
            return None, False
        if any(r == "NULL" for r in rendered):
            return None, False
        joined = " OR ".join(f"contains({column.sql}, {r})" for r in rendered)
        return f"({joined})", True

    def _contains(self, args: Sequence[Json]) -> tuple[str | None, bool]:
        if len(args) != 2:
            return None, False
        try:
            column, needle = self.value(args[0]), self.value(args[1])
        except _Decline:
            return None, False
        return f"array_has({column.sql}, {needle.sql})", True

    def _is_floating(self, name: Json) -> bool:
        """Whether `name` is already a float column, so a promotion is a no-op."""
        if self.schema is None or not isinstance(name, str):
            return False
        return isinstance(self.schema.get(name), (pl.Float32, pl.Float64))

    def _is_text(self, name: Json) -> bool:
        """Whether `name` is a text column, so a `+` on it means concatenation."""
        if self.schema is None or not isinstance(name, str):
            return False
        dtype = self.schema.get(name)
        return dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))

    # -- value position ----------------------------------------------------

    def value(self, node: Json) -> _Value:
        """Lower a value node, or raise `_Decline`.

        Value position has no relaxed form.
        """
        kind, body = _unpack(node)

        if kind == "Alias":
            return self.value(_inputs(body)[0])
        if kind == "Column":
            return _Value(
                _column(body),
                is_double=self._is_floating(body),
                is_string=self._is_text(body),
            )
        if kind == "Literal":
            return _literal(node)
        if kind == "Cast":
            return self._cast(body)
        if kind == "BinaryExpr":
            return self._arithmetic(body)
        if kind == "Function":
            return self._function_value(body)
        raise _Decline

    def _cast(self, node: Json) -> _Value:
        body = _fields(node)
        dtype = body.get("dtype")
        name = dtype.get("Literal") if isinstance(dtype, dict) else None
        if not isinstance(name, str) or name not in _CAST_TYPES:
            raise _Decline
        if body.get("options") != "Strict" and not self._is_widening(body, name):
            # A non-strict Polars cast yields null where Lance fails the scan,
            # unless it cannot fail, which is the case the optimizer creates.
            raise _Decline
        return _Value(
            f"CAST({self.value(_field(body, 'expr')).sql} AS {_CAST_TYPES[name]})",
            is_double=name in ("Float32", "Float64"),
        )

    def _is_widening(self, body: dict[str, Json], target: str) -> bool:
        """Whether this non-strict cast cannot produce a null.

        Comparing an int column to a float one makes Polars' optimizer insert
        `col.cast(Float64, strict=False)` before the plugin sees the predicate.
        Widening a number to a float never fails; nothing else here is checked,
        hence the schema lookup.
        """
        if target not in ("Float32", "Float64"):
            return False
        inner = body.get("expr")
        if not isinstance(inner, dict) or list(inner) != ["Column"]:
            return False
        column = inner["Column"]
        if not isinstance(column, str):
            return False
        source = self.schema.get(column) if self.schema is not None else None
        return source is not None and source.is_numeric()

    def _arithmetic(self, node: Json) -> _Value:
        body = _fields(node)
        op = body.get("op")
        if not isinstance(op, str):
            raise _Decline
        if op in _ARITHMETIC:
            left = self.value(_field(body, "left"))
            right = self.value(_field(body, "right"))
            if op == "Plus" and (left.is_string or right.is_string):
                # Polars overloads `+` for text; SQL spells that `||`. A string
                # literal settles it on its own; two columns need the schema,
                # without which this stays `+` and Lance declines to plan it.
                return _Value(f"({left.sql} || {right.sql})", is_string=True)
            return _Value(f"({left.sql} {_ARITHMETIC[op]} {right.sql})")
        if op == "TrueDivide":
            # Polars' `/` is always float division; SQL's is integer division
            # between integers.
            left = self.value(_field(body, "left")).as_double()
            divisor = self.value(_field(body, "right"))
            return _Value(f"({left.sql} / {divisor.sql})", is_double=True)
        # FloorDivide has no Lance spelling (`floor()` is rejected).
        raise _Decline

    def _function_value(self, node: Json) -> _Value:
        body = _fields(node)
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
        if name in (("MinHorizontal",), ("MaxHorizontal",)):
            # `least` / `greatest` skip nulls, which is what the Polars
            # horizontal reductions do too.
            fn = "least" if name[0] == "MinHorizontal" else "greatest"
            rendered = ", ".join(self.value(a).sql for a in args)
            return _Value(f"{fn}({rendered})")
        if name == ("Pow", "Generic") and len(args) == 2:
            return self._power(args)
        # `Round` is deliberately absent: Polars breaks ties to even, Lance
        # away from zero, and nothing in the IR lets us ask for the other one.
        # So are `sqrt` / `ln` / `log10` / `cbrt`: outside their domain Polars
        # produces NaN and Lance produces NULL, which sort differently.
        if name[0] == "StringExpr":
            return self._string_value(name[1] if len(name) > 1 else "", payload, args)
        if name[0] == "TemporalExpr":
            return self._temporal_value(name[1] if len(name) > 1 else "", args)
        if name == ("StructExpr", "FieldByName") and len(args) == 1:
            if not isinstance(payload, str):
                raise _Decline
            return _Value(f"{self.value(args[0]).sql}.{_quote(payload)}")
        if name[0] in ("ListExpr", "ArrayExpr") and name[-1] == "Length":
            return _Value(f"array_length({self.value(args[0]).sql})")
        if name[0] in ("ListExpr", "ArrayExpr") and name[-1] == "Get":
            return self._list_get(payload, args)
        raise _Decline

    def _power(self, args: Sequence[Json]) -> _Value:
        """`a ** b`, restricted to a whole non-negative exponent.

        Outside `power`'s domain Polars yields NaN and Lance yields NULL.
        """
        exponent = _literal(args[1])
        try:
            whole = int(exponent.sql)
        except ValueError as exc:
            raise _Decline from exc
        if whole < 0:
            raise _Decline
        return _Value(f"power({self.value(args[0]).sql}, {whole})")

    def _list_get(self, payload: Json, args: Sequence[Json]) -> _Value:
        """`list.get(i)`, only in its null-on-out-of-bounds spelling.

        The payload is the `null_on_oob` flag; unset, Polars raises where
        `array_element` returns null. Polars indexes from 0 and SQL from 1, and
        a negative index counts from the end in both.
        """
        if payload is not True or len(args) != 2:
            raise _Decline
        try:
            index = int(_literal(args[1]).sql)
        except ValueError as exc:
            # A non-integer index has no `array_element` spelling.
            raise _Decline from exc
        position = index + 1 if index >= 0 else index
        return _Value(f"array_element({self.value(args[0]).sql}, {position})")

    def _string_value(self, name: str, payload: Json, args: Sequence[Json]) -> _Value:
        column = self.value(args[0]).sql
        if name == "Lowercase":
            return _Value(f"lower({column})", is_string=True)
        if name == "Uppercase":
            return _Value(f"upper({column})", is_string=True)
        if name == "LenChars" and len(args) == 1:
            return _Value(f"length({column})")
        if name == "LenBytes" and len(args) == 1:
            return _Value(f"octet_length({column})")
        if name in ("StripChars", "StripCharsStart", "StripCharsEnd"):
            return self._strip(name, column, args)
        if name == "Replace" and len(args) == 3:
            return self._replace(_options(payload), column, args)
        if name == "ConcatHorizontal":
            return self._concat(_options(payload), args)
        raise _Decline

    def _concat(self, options: dict[str, Json], args: Sequence[Json]) -> _Value:
        """`concat_str`, which is what the optimizer rewrites a string `+` into.

        `||` propagates null and `concat` skips it, matching `ignore_nulls`.
        `concat_ws` also skips null, so a separator with `ignore_nulls=False`
        has no equivalent.
        """
        parts = [self.value(a).sql for a in args]
        if not parts:
            raise _Decline
        delimiter = options.get("delimiter", "")
        if not isinstance(delimiter, str):
            raise _Decline
        ignore_nulls = bool(options.get("ignore_nulls"))
        if not delimiter:
            joined = ", ".join(parts)
            if ignore_nulls:
                return _Value(f"concat({joined})", is_string=True)
            return _Value(f"({' || '.join(parts)})", is_string=True)
        if not ignore_nulls:
            raise _Decline
        separator = _string_literal(delimiter)
        return _Value(f"concat_ws({separator}, {', '.join(parts)})", is_string=True)

    def _strip(self, name: str, column: str, args: Sequence[Json]) -> _Value:
        """`str.strip_chars` and friends, only with an explicit character set.

        With no argument Polars strips Unicode whitespace and `btrim` strips
        spaces. The one-argument form is a character set on both sides.
        """
        if len(args) != 2:
            raise _Decline
        chars = _literal(args[1])
        if chars.sql == "NULL":
            raise _Decline
        fn = {"StripChars": "btrim", "StripCharsStart": "ltrim"}.get(name, "rtrim")
        return _Value(f"{fn}({column}, {chars.sql})", is_string=True)

    def _replace(
        self, options: dict[str, Json], column: str, args: Sequence[Json]
    ) -> _Value:
        """`str.replace` / `str.replace_all`.

        `n` is 1 for `replace` and -1 for `replace_all`. SQL's `replace` is
        literal and replaces every occurrence; `regexp_replace` replaces the
        first unless given the `g` flag. Literal-and-first-only has no spelling.
        """
        pattern, replacement = self.value(args[1]), self.value(args[2])
        every = options.get("n") == -1
        if options.get("literal"):
            if not every:
                raise _Decline
            return _Value(
                f"replace({column}, {pattern.sql}, {replacement.sql})", is_string=True
            )
        flags = ", 'g'" if every else ""
        return _Value(
            f"regexp_replace({column}, {pattern.sql}, {replacement.sql}{flags})",
            is_string=True,
        )

    def _temporal_value(self, name: str, args: Sequence[Json]) -> _Value:
        column = self.value(args[0]).sql
        if name == "Date":
            return _Value(f"CAST({column} AS date)")
        if name == "WeekDay":
            # Polars counts Monday as 1; `dow` counts Sunday as 0.
            return _Value(f"((date_part('dow', {column}) + 6) % 7 + 1)")
        if name == "Truncate" and len(args) == 2:
            every = _literal(args[1]).sql
            unit = _TRUNCATE_UNITS.get(every.strip("'"))
            if unit is None:
                raise _Decline
            return _Value(f"date_trunc('{unit}', {column})")
        part = _DATE_PARTS.get(name)
        if part is None:
            raise _Decline
        return _Value(f"date_part('{part}', {column})")


# ---------------------------------------------------------------------------
# tree and literal helpers
# ---------------------------------------------------------------------------


def _inputs(node: Json) -> list[Json]:
    """The argument list of an IR node, declining an empty or non-list one."""
    if not isinstance(node, list) or not node:
        raise _Decline
    return node


def _fields(node: Json) -> dict[str, Json]:
    """`node` as an IR object, declining anything else."""
    if not isinstance(node, dict):
        raise _Decline
    return node


def _field(body: dict[str, Json], name: str) -> Json:
    """One named field of an IR object, declining if it is absent."""
    if name not in body:
        raise _Decline
    return body[name]


def _unpack(node: Json) -> tuple[str, Json]:
    """Split a single-key IR node into its tag and body."""
    if not isinstance(node, dict) or len(node) != 1:
        raise _Decline
    return next(iter(node.items()))


def _function(node: Json) -> tuple[tuple[str, ...], Json]:
    """Split a `function` field into its name path and its payload.

    The IR spells a function four ways::

        "Abs"                            -> ("Abs",),               None
        {"Boolean": "IsNull"}            -> ("Boolean", "IsNull"),  None
        {"Boolean": {"IsIn": {...}}}     -> ("Boolean", "IsIn"),    {...}
        {"Round": {"decimals": 2, ...}}  -> ("Round",),             {...}

    """
    function = node.get("function") if isinstance(node, dict) else None
    if isinstance(function, str):
        return (function,), None
    if not isinstance(function, dict) or len(function) != 1:
        return (), None
    namespace, inner = next(iter(function.items()))
    if isinstance(inner, str):
        return (namespace, inner), None
    if isinstance(inner, dict) and len(inner) == 1:
        name, payload = next(iter(inner.items()))
        if namespace in _NAMESPACES or isinstance(payload, (dict, str)):
            return (namespace, name), payload
    return (namespace,), inner


def _options(payload: Json) -> dict[str, Json]:
    return payload if isinstance(payload, dict) else {}


def _column(name: Json) -> str:
    if not isinstance(name, str) or not name or name in VIRTUAL_COLUMNS:
        raise _Decline
    return _quote(name)


def _quote(name: str) -> str:
    """Backtick-quote an identifier; Lance rejects the double-quoted form."""
    if "`" in name:
        raise _Decline
    return f"`{name}`"


def _literal_series(node: Json) -> pl.Series:
    """Evaluate a literal subtree back to a Series.

    Round-tripping through Polars beats decoding the IR by hand: the payload is
    a dtype-dependent mix of inline values and Arrow IPC blobs, and which one
    is used has changed between releases.
    """
    try:
        expr = pl.Expr.deserialize(io.BytesIO(json.dumps(node).encode()), format="json")
        return pl.select(expr).to_series()
    except Exception as exc:
        raise _Decline from exc


def _literal(node: Json) -> _Value:
    series = _literal_series(node)
    if series.len() != 1:
        raise _Decline
    dtype = series.dtype
    return _Value(
        _scalar(series.item(), dtype),
        is_float_literal=isinstance(dtype, (pl.Float32, pl.Float64)),
        is_string=isinstance(dtype, (pl.String, pl.Categorical, pl.Enum)),
    )


def _literal_elements(node: Json) -> pl.Series:
    """The membership list of an `is_in`.

    Spelled either as a Series literal or as a one-element List literal.
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


def _scalar(value: object, dtype: pl.DataType) -> str:
    """Render a Python value as a Lance SQL literal."""
    if value is None:
        return "NULL"
    if dtype == pl.Boolean:
        return "TRUE" if value else "FALSE"
    if isinstance(dtype, (pl.String, pl.Categorical, pl.Enum)):
        return _string_literal(str(value))
    if isinstance(
        dtype,
        (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ),
    ):
        if not isinstance(value, (int, float)):
            raise _Decline
        return str(int(value))
    if isinstance(dtype, (pl.Float32, pl.Float64)):
        if not isinstance(value, (int, float)):
            raise _Decline
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise _Decline
        return repr(number)
    if dtype == pl.Date:
        if not isinstance(value, dt.date):
            raise _Decline
        return f"date '{value.isoformat()}'"
    if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
        if not isinstance(value, dt.datetime):
            raise _Decline
        # Lance parses a bare timestamp literal against the column's own time
        # zone, so an aware literal is normalised to UTC first.
        stamp = value
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return f"timestamp '{stamp.isoformat(sep=' ')}'"
    if dtype == pl.Binary:
        if not isinstance(value, (bytes, bytearray)):
            raise _Decline
        return f"X'{bytes(value).hex()}'"
    # Time, Duration, Decimal, and every nested dtype: no dependable spelling.
    raise _Decline


def _string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_non_null_literal(node: Json) -> bool:
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
