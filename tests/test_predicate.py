"""What the Polars to Lance SQL lowering does, and what it refuses to do.

A lowering must keep at least the rows the predicate keeps, and an `exact` one
must keep precisely them, since `scan_lance` then stops filtering. Three
families of tests hold it to that: shapes that pin the SQL, refusals, and
differential tests that run both against a real dataset.
"""

from __future__ import annotations

import datetime as dt
import math
import operator
import random
import threading
from typing import TYPE_CHECKING

import lance
import polars as pl
import pytest

from conftest import RICH_SCHEMA
from polars_pylance._predicate import (
    Json,
    LanceFilter,
    _Decline,
    _Lowering,
    to_lance_filter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa


# The `rich` fixture's columns as `scan_lance` sees them. The scan always lowers
# with the schema, so the shapes below are pinned with it.
def _polars_schema(schema: pa.Schema) -> pl.Schema:
    frame = pl.from_arrow(schema.empty_table())
    assert isinstance(frame, pl.DataFrame)
    return frame.schema


RICH = _polars_schema(RICH_SCHEMA)

# ---------------------------------------------------------------------------
# shape: one construct at a time
# ---------------------------------------------------------------------------

TRANSLATIONS: list[tuple[str, pl.Expr, str]] = [
    ("comparison", pl.col("id") > 7, "(`id` > 7)"),
    ("string equality", pl.col("cat") == "beta", "(`cat` = 'beta')"),
    ("bare boolean column", pl.col("flag"), "`flag`"),
    ("negation", ~(pl.col("id") > 7), "(NOT (`id` > 7))"),
    ("conjunction", (pl.col("id") > 7) & pl.col("flag"), "((`id` > 7) AND `flag`)"),
    ("disjunction", (pl.col("id") > 7) | pl.col("flag"), "((`id` > 7) OR `flag`)"),
    (
        "xor",
        (pl.col("id") > 7) ^ pl.col("flag"),
        "(((`id` > 7) AND NOT `flag`) OR (NOT (`id` > 7) AND `flag`))",
    ),
    ("is_null", pl.col("opt").is_null(), "(`opt` IS NULL)"),
    ("is_not_null", pl.col("opt").is_not_null(), "(`opt` IS NOT NULL)"),
    # Not `x != x`: Lance compares NaN by SQL's total ordering, under which NaN
    # does equal itself, so that spelling would silently drop every NaN row.
    ("is_nan", pl.col("odd").is_nan(), "(isnan(`odd`))"),
    ("is_not_nan", pl.col("odd").is_not_nan(), "(NOT isnan(`odd`))"),
    (
        "is_infinite",
        pl.col("odd").is_infinite(),
        "(`odd` = CAST('inf' AS double) OR `odd` = CAST('-inf' AS double))",
    ),
    (
        "is_finite",
        pl.col("odd").is_finite(),
        (
            "(NOT isnan(`odd`) AND `odd` != CAST('inf' AS double) "
            "AND `odd` != CAST('-inf' AS double))"
        ),
    ),
    ("is_in", pl.col("id").is_in([1, 2]), "(`id` IN (1, 2))"),
    # False, except null for a null input, so that negation still drops it.
    ("is_in empty", pl.col("id").is_in([]), "(`id` IS NULL AND CAST(NULL AS boolean))"),
    # Polars never matches a null element; SQL's `IN` would turn null instead.
    (
        "is_in with a null element",
        pl.col("id").is_in(pl.Series([1, None])),
        "(`id` IN (1))",
    ),
    # Polars matches `-0.0` and `0.0` to each other; Lance's total order does not.
    (
        "is_in with a zero",
        pl.col("val").is_in([0.0, 0.5]),
        "(`val` IN (0.5, -0.0, 0.0))",
    ),
    ("is_between", pl.col("id").is_between(1, 2), "((`id` >= 1) AND (`id` <= 2))"),
    ("starts_with", pl.col("cat").str.starts_with("b"), "starts_with(`cat`, 'b')"),
    ("ends_with", pl.col("cat").str.ends_with("a"), "ends_with(`cat`, 'a')"),
    (
        "contains, literal",
        pl.col("text").str.contains("x", literal=True),
        "contains(`text`, 'x')",
    ),
    (
        "contains, regex",
        pl.col("text").str.contains("x.y"),
        "regexp_like(`text`, 'x.y')",
    ),
    (
        "contains_any",
        pl.col("text").str.contains_any(["x", "y"]),
        "(contains(`text`, 'x') OR contains(`text`, 'y'))",
    ),
    ("len_chars", pl.col("cat").str.len_chars() > 4, "(length(`cat`) > 4)"),
    ("len_bytes", pl.col("cat").str.len_bytes() > 4, "(octet_length(`cat`) > 4)"),
    ("lowercase", pl.col("cat").str.to_lowercase() == "b", "(lower(`cat`) = 'b')"),
    ("uppercase", pl.col("cat").str.to_uppercase() == "B", "(upper(`cat`) = 'B')"),
    (
        "strip_chars",
        pl.col("text").str.strip_chars(" ") == "b",
        "(btrim(`text`, ' ') = 'b')",
    ),
    (
        "strip_chars_start",
        pl.col("text").str.strip_chars_start(" ") == "b",
        "(ltrim(`text`, ' ') = 'b')",
    ),
    (
        "strip_chars_end",
        pl.col("text").str.strip_chars_end(" ") == "b",
        "(rtrim(`text`, ' ') = 'b')",
    ),
    (
        "replace_all, literal",
        pl.col("cat").str.replace_all("a", "X", literal=True) == "X",
        "(replace(`cat`, 'a', 'X') = 'X')",
    ),
    (
        "replace, regex",
        pl.col("cat").str.replace("a.", "X") == "X",
        "(regexp_replace(`cat`, 'a.', 'X') = 'X')",
    ),
    (
        "replace_all, regex",
        pl.col("cat").str.replace_all("a.", "X") == "X",
        "(regexp_replace(`cat`, 'a.', 'X', 'g') = 'X')",
    ),
    (
        "string concatenation",
        (pl.col("cat") + "!") == "beta!",
        "((`cat` || '!') = 'beta!')",
    ),
    (
        "concat_str",
        pl.concat_str(pl.col("cat"), pl.col("text")) == "x",
        "((`cat` || `text`) = 'x')",
    ),
    (
        "concat_str ignoring nulls",
        pl.concat_str(pl.col("cat"), pl.col("text"), ignore_nulls=True) == "x",
        "(concat(`cat`, `text`) = 'x')",
    ),
    (
        "concat_str with a separator",
        pl.concat_str(pl.col("cat"), pl.col("text"), separator="-", ignore_nulls=True)
        == "x",
        "(concat_ws('-', `cat`, `text`) = 'x')",
    ),
    ("arithmetic", (pl.col("id") + 1) > 3, "((`id` + 1) > 3)"),
    # Polars' remainder takes the divisor's sign, SQL's the dividend's.
    (
        "modulo",
        (pl.col("id") % 2) == 0,
        (
            "(((`id` % NULLIF(2, 0)) + NULLIF(2, 0) * CAST("
            "(((`id` % NULLIF(2, 0)) < 0 AND NULLIF(2, 0) > 0)"
            " OR ((`id` % NULLIF(2, 0)) > 0 AND NULLIF(2, 0) < 0)) AS bigint)) = 0)"
        ),
    ),
    (
        "float against zero",
        pl.col("val") == 0.0,
        "(`val` IN (-0.0, 0.0))",
    ),
    # `x < -inf` holds for exactly a NaN with its sign bit set, which Lance
    # orders below every number and Polars above.
    (
        "float below zero",
        pl.col("val") < 0.0,
        "((`val` < -0.0) AND `val` >= CAST('-inf' AS double))",
    ),
    (
        "float above zero",
        pl.col("val") > 0.0,
        "((`val` > 0.0) OR `val` < CAST('-inf' AS double))",
    ),
    (
        "zero on the left",
        pl.lit(0.0) <= pl.col("val"),
        "((`val` >= -0.0) OR `val` < CAST('-inf' AS double))",
    ),
    ("negate", -pl.col("id") < 0, "((- `id`) < 0)"),
    ("power", (pl.col("id") ** 2) > 9, "(power(`id`, 2) > 9)"),
    ("abs", pl.col("id").abs() > 3, "(abs(`id`) > 3)"),
    # `cbrt` is defined everywhere, so of an integer it cannot be NaN.
    ("cbrt", pl.col("id").cbrt() > 0.5, "(cbrt(`id`) > 0.5)"),
    (
        "ln",
        pl.col("val").log() > 0.5,
        "((ln(`val`) > 0.5) OR ln(`val`) < CAST('-inf' AS double))",
    ),
    (
        "min_horizontal",
        pl.min_horizontal(pl.col("id"), pl.col("opt")) > 3,
        "(least(`id`, `opt`) > 3)",
    ),
    (
        "max_horizontal",
        pl.max_horizontal(pl.col("id"), pl.col("opt")) > 3,
        "(greatest(`id`, `opt`) > 3)",
    ),
    ("column vs column", pl.col("id") > pl.col("opt"), "(`id` > `opt`)"),
    # An integer column cast to a float holds no NaN.
    ("cast", pl.col("id").cast(pl.Float64) > 1, "(CAST(`id` AS double) > 1.0)"),
    (
        "non-strict cast",
        pl.col("cat").cast(pl.Int64, strict=False) == 1,
        "(TRY_CAST(`cat` AS bigint) = 1)",
    ),
    ("date part", pl.col("ts").dt.year() == 2024, "(date_part('year', `ts`) = 2024)"),
    # Polars counts Monday as 1; SQL's `dow` counts Sunday as 0.
    (
        "weekday",
        pl.col("ts").dt.weekday() == 1,
        "(((date_part('dow', `ts`) + 6) % 7 + 1) = 1)",
    ),
    (
        "truncate",
        pl.col("ts").dt.truncate("1d") == dt.datetime(2024, 1, 2),
        "(date_trunc('day', `ts`) = timestamp '2024-01-02 00:00:00')",
    ),
    (
        "date cast",
        pl.col("ts").dt.date() == dt.date(2024, 1, 2),
        "(CAST(`ts` AS date) = date '2024-01-02')",
    ),
    ("list contains", pl.col("tags").list.contains(3), "array_has(`tags`, 3)"),
    (
        "list contains a zero",
        pl.col("tags").list.contains(0.0),
        "(array_has(`tags`, -0.0) OR array_has(`tags`, 0.0))",
    ),
    ("list length", pl.col("tags").list.len() == 2, "(array_length(`tags`) = 2)"),
    # Polars indexes lists from 0, SQL from 1.
    (
        "list get",
        pl.col("tags").list.get(0, null_on_oob=True) == 3,
        "(array_element(`tags`, 1) = 3)",
    ),
    (
        "list get, from the end",
        pl.col("tags").list.get(-1, null_on_oob=True) == 3,
        "(array_element(`tags`, -1) = 3)",
    ),
    ("struct field", pl.col("meta").struct.field("k") > 5, "(`meta`.`k` > 5)"),
    ("fill_null", pl.col("opt").fill_null(0) > 5, "(coalesce(`opt`, 0) > 5)"),
    (
        "all_horizontal",
        pl.all_horizontal(pl.col("flag"), pl.col("id") > 1),
        "(`flag` AND (`id` > 1))",
    ),
    (
        "any_horizontal",
        pl.any_horizontal(pl.col("flag"), pl.col("id") > 1),
        "(`flag` OR (`id` > 1))",
    ),
    # Preserve Boolean values, including when composed beneath NOT or XOR.
    (
        "eq_missing",
        pl.col("opt").eq_missing(3),
        "((`opt` = 3) IS TRUE)",
    ),
    (
        "ne_missing",
        pl.col("opt").ne_missing(3),
        "((`opt` = 3) IS NOT TRUE)",
    ),
    ("alias is transparent", (pl.col("id") > 7).alias("x"), "(`id` > 7)"),
    ("quoted identifier", pl.col("odd name") > 1, "(`odd name` > 1)"),
    # Polars promotes to float to compare; Lance refuses the mixed comparison.
    (
        "int against float",
        pl.col("id") > 1.5,
        "(CAST(`id` AS double) > 1.5)",
    ),
    (
        "true division",
        pl.col("id") / 2 > 1.5,
        (
            "(((CAST(`id` AS double) / 2) > 1.5)"
            " OR (CAST(`id` AS double) / 2) < CAST('-inf' AS double))"
        ),
    ),
    ("string escaping", pl.col("cat") == "o'brien", "(`cat` = 'o''brien')"),
    (
        "date literal",
        pl.col("day") == dt.date(2024, 2, 1),
        "(`day` = date '2024-02-01')",
    ),
    (
        "datetime literal",
        pl.col("ts") == dt.datetime(2024, 1, 2, 3, 4, 5),
        "(`ts` = timestamp '2024-01-02 03:04:05')",
    ),
    ("binary literal", pl.col("bin") == b"abc", "(`bin` = X'616263')"),
    ("bool literal", pl.col("flag") == True, "(`flag` = TRUE)"),  # noqa: E712
    # Not `FALSE`, which negation would turn true.
    ("null literal", pl.lit(None, dtype=pl.Boolean), "CAST(NULL AS boolean)"),
]


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [pytest.param(e, s, id=name) for name, e, s in TRANSLATIONS],
)
def test_lowering_shape(predicate: pl.Expr, expected: str) -> None:
    lowered = to_lance_filter(predicate, schema=RICH)
    assert lowered is not None
    assert lowered.sql == expected
    assert lowered.exact


# Without a schema a number may be a float of either width, which Lance orders and
# rounds differently, so numeric predicates decline.
UNTYPED_TRANSLATIONS: list[tuple[str, pl.Expr, str | None]] = [
    ("integer literal", pl.col("id") > 7, None),
    ("literal on the left", pl.lit(0) < pl.col("id"), None),
    ("float literal", pl.col("val") > 0.5, None),
    ("is_in", pl.col("id").is_in([0, 2]), None),
    ("list contains", pl.col("tags").list.contains(0), None),
    # Nothing about text or dates needs guarding.
    ("string literal", pl.col("cat") == "beta", "(`cat` = 'beta')"),
    (
        "datetime literal",
        pl.col("ts") > dt.datetime(2024, 1, 2),
        "(`ts` > timestamp '2024-01-02 00:00:00')",
    ),
    # Two untyped columns may as well be strings, which have no such spelling.
    ("column against column", pl.col("id") > pl.col("opt"), None),
    ("min_horizontal", pl.min_horizontal("id", "opt") > 3, None),
    ("max_horizontal", pl.max_horizontal("id", "opt") > 3, None),
]


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [pytest.param(e, s, id=name) for name, e, s in UNTYPED_TRANSLATIONS],
)
def test_lowering_shape_without_a_schema(
    predicate: pl.Expr, expected: str | None
) -> None:
    lowered = to_lance_filter(predicate)
    if expected is None:
        assert lowered is None
        return
    assert lowered == LanceFilter(sql=expected, exact=True)


DECLINED: list[tuple[str, pl.Expr]] = [
    ("python-level hash", pl.col("id").hash() % 2 == 0),
    ("aggregate in predicate", pl.col("val") > pl.col("val").mean()),
    ("when/then", pl.when(pl.col("id") > 1).then(True).otherwise(False)),
    ("str.slice", pl.col("cat").str.slice(0, 2) == "be"),
    # Lance's `ln` of a Float32 is an ulp off, and the column may be one.
    ("ln, untyped", pl.col("val").log() > 0.5),
    # Polars strips every Unicode whitespace; `btrim` strips spaces.
    ("str.strip_chars, no argument", pl.col("cat").str.strip_chars() == "b"),
    # SQL's `replace` is literal but replaces every occurrence, so the one
    # combination that has no spelling is literal-and-first-only.
    ("str.replace, literal", pl.col("cat").str.replace("a", "X", literal=True) == "X"),
    # Polars folds ASCII only; Lance has no matching spelling.
    (
        "str.contains_any, case-insensitive",
        pl.col("cat").str.contains_any(["a"], ascii_case_insensitive=True),
    ),
    ("str.to_titlecase", pl.col("cat").str.to_titlecase() == "Beta"),
    # Polars raises where `array_element` returns null.
    ("list.get, strict", pl.col("tags").list.get(0) == 3),
    # Polars breaks ties to even, Lance away from zero.
    ("round", pl.col("val").round(2) == 0.5),
    # Lance folds `power(x, 0)` to 1, dropping nulls.
    ("zero power", (pl.col("id") ** 0) == 1),
    # `0 ** -1` is `inf` in Polars and fails the scan in Lance.
    ("negative power", (pl.col("id") ** -1) > 0),
    # `date_part('epoch', ...)` keeps the fraction; `dt.epoch` truncates.
    ("dt.epoch", pl.col("ts").dt.epoch("s") > 0),
    # `date_trunc` has no spelling for a multiple of a unit.
    (
        "dt.truncate, multiple",
        pl.col("ts").dt.truncate("2d") == dt.datetime(2024, 1, 2),
    ),
    # Without a schema either side may be a float, of either width.
    ("floor division", (pl.col("id") // 2) == 1),
    ("modulo", (pl.col("id") % 2) == 1),
    ("float modulo", (pl.col("val") % 2.0) == 1),
    # Not translated yet, though Polars and Lance both raise on overflow.
    ("narrowing cast", pl.col("id").cast(pl.Int32) > 1),
    # `concat_ws` skips nulls, so it cannot spell a null-propagating join.
    (
        "concat_str with a separator, keeping nulls",
        pl.concat_str(pl.col("cat"), pl.col("text"), separator="-") == "x",
    ),
    # Without a schema, `TRY_CAST` may not null what Polars nulls.
    ("non-strict cast", pl.col("id").cast(pl.Float64, strict=False) > 1),
    ("time literal", pl.col("t") == dt.time(12, 30)),
    ("duration literal", pl.col("d") > dt.timedelta(hours=1)),
    ("generated column", pl.col("_rowid") > 5),
    ("nan literal", pl.col("val") > float("nan")),
    ("backtick in name", pl.col("we`ird") > 1),
]


@pytest.mark.parametrize(
    "predicate", [pytest.param(e, id=name) for name, e in DECLINED]
)
def test_declined(predicate: pl.Expr) -> None:
    assert to_lance_filter(predicate) is None


def test_long_is_in_is_declined() -> None:
    """Past some size the SQL round trip stops paying for itself."""
    predicate = pl.col("id").is_in(list(range(10)))
    assert to_lance_filter(predicate, schema=RICH) is not None
    assert to_lance_filter(predicate, schema=RICH, max_in_list=5) is None


# ---------------------------------------------------------------------------
# relaxation: what happens when only part of a predicate lowers
# ---------------------------------------------------------------------------

UNTRANSLATABLE = pl.col("cat").str.slice(0, 2) == "beta"


def test_conjunct_is_dropped() -> None:
    """An AND keeps whatever lowered; the engine filters the rest."""
    lowered = to_lance_filter((pl.col("id") > 5) & UNTRANSLATABLE, schema=RICH)
    assert lowered == LanceFilter(sql="(`id` > 5)", exact=False)


def test_deep_conjunct_is_dropped() -> None:
    predicate = (
        (pl.col("id") > 5)
        & (pl.col("val") < 0.9)
        & UNTRANSLATABLE
        & pl.col("text").str.starts_with("row")
    )
    lowered = to_lance_filter(predicate, schema=RICH)
    assert lowered is not None
    assert not lowered.exact
    assert "`id` > 5" in lowered.sql
    assert "starts_with" in lowered.sql


def test_disjunct_is_not_dropped() -> None:
    """Dropping a branch of an OR would remove rows the predicate keeps."""
    assert to_lance_filter((pl.col("id") > 5) | UNTRANSLATABLE) is None


def test_negated_relaxation_is_declined() -> None:
    """NOT of a superset is a subset, so a relaxed child cannot be negated."""
    assert to_lance_filter(~((pl.col("id") > 5) & UNTRANSLATABLE)) is None


def test_relaxed_xor_is_declined() -> None:
    """The xor expansion negates both halves, so neither may be a superset."""
    assert (
        to_lance_filter((pl.col("id") > 5) ^ ((pl.col("id") < 9) & UNTRANSLATABLE))
        is None
    )


def test_relaxation_survives_nesting_in_a_conjunction() -> None:
    predicate = ((pl.col("id") > 5) | (pl.col("val") < 0.1)) & UNTRANSLATABLE
    lowered = to_lance_filter(predicate, schema=RICH)
    assert lowered is not None
    assert not lowered.exact
    assert lowered.sql == (
        "((`id` > 5) OR ((`val` < 0.1) AND `val` >= CAST('-inf' AS double)))"
    )


# ---------------------------------------------------------------------------
# differential: run both and compare row sets
# ---------------------------------------------------------------------------


def _ids(dataset: lance.LanceDataset, filter: str) -> set[int]:
    table = dataset.scanner(columns=["id"], filter=filter).to_table()
    return {i for i in table["id"].to_pylist() if i is not None}


DIFFERENTIAL: list[pl.Expr] = [
    predicate
    for _, predicate, _ in TRANSLATIONS
    # dtypes the shared fixture does not carry
    if not any(c in ("bin", "t", "d") for c in predicate.meta.root_names())
] + [
    (pl.col("id") > 5) & UNTRANSLATABLE,
    (pl.col("cat").str.starts_with("b") | (pl.col("val") < 0.2)) & (pl.col("id") < 900),
    ~(pl.col("cat").is_in(["beta", "gamma"])),
    pl.col("opt").fill_null(-1).is_between(3, 99),
    pl.col("meta").struct.field("s").str.starts_with("g"),
    (pl.col("ts").dt.hour() < 6) & pl.col("tags").list.contains(2),
    pl.col("odd").is_nan() | pl.col("odd").is_infinite(),
    pl.col("odd").is_finite() & (pl.col("odd") > 5.0),
    pl.col("text").str.strip_chars(" ").str.ends_with("beta"),
    pl.min_horizontal(pl.col("id"), pl.col("opt")) > 1_500,
    pl.col("ts").dt.truncate("1mo") == dt.datetime(2024, 1, 1),
    pl.col("tags").list.get(0, null_on_oob=True).is_null(),
]


@pytest.mark.parametrize("schema", [RICH, None], ids=["schema", "no schema"])
@pytest.mark.parametrize(
    "predicate", [pytest.param(p, id=str(p)[:60]) for p in DIFFERENTIAL]
)
def test_lance_keeps_every_row_polars_keeps(
    predicate: pl.Expr,
    schema: pl.Schema | None,
    rich_uri: str,
    rich_frame: pl.DataFrame,
) -> None:
    lowered = to_lance_filter(predicate, schema=schema)
    if lowered is None and schema is None:
        # Two untyped columns decline; the schema'd run holds the lowering to it.
        return
    assert lowered is not None, "expected this predicate to lower"
    dataset = lance.dataset(rich_uri)
    kept = set(rich_frame.filter(predicate)["id"].to_list())
    pushed = _ids(dataset, lowered.sql)
    assert kept <= pushed, "the pushed filter dropped rows the predicate keeps"
    if lowered.exact:
        assert pushed == kept


def _random_predicate(rng: random.Random, depth: int) -> pl.Expr:
    """A random predicate tree, including leaves that deliberately do not lower."""
    if depth == 0:
        leaves: list[Callable[[], pl.Expr]] = [
            lambda: pl.col("id") > rng.randrange(2000),
            lambda: pl.col("id").is_in([rng.randrange(2000) for _ in range(3)]),
            lambda: pl.col("id") % rng.choice([2, 3]) == 0,
            lambda: pl.col("val") < rng.random(),
            lambda: pl.col("val").abs() > rng.random(),
            lambda: pl.col("odd").is_nan(),
            lambda: pl.col("odd").is_finite(),
            lambda: pl.col("odd") > rng.random() * 10,
            lambda: pl.col("cat").str.starts_with(rng.choice(["a", "b", "z"])),
            lambda: pl.col("cat").str.contains_any(["bet", "gam"]),
            lambda: pl.col("text").str.contains("row-000", literal=True),
            lambda: pl.col("text").str.strip_chars(" ").str.ends_with("beta"),
            lambda: pl.col("opt").is_null(),
            lambda: pl.col("opt").fill_null(0) > rng.randrange(2000),
            lambda: pl.min_horizontal("id", "opt") > rng.randrange(2000),
            lambda: pl.col("ts").dt.hour() < rng.randrange(24),
            lambda: pl.col("ts").dt.weekday() == rng.randrange(1, 8),
            lambda: pl.col("ts").dt.truncate("1d") == dt.datetime(2024, 1, 2),
            lambda: pl.col("tags").list.contains(rng.randrange(5)),
            lambda: pl.col("tags").list.get(0, null_on_oob=True) == rng.randrange(5),
            lambda: pl.col("meta").struct.field("k") > rng.randrange(10),
            lambda: pl.col("flag"),
            # leaves with no Lance equivalent, to exercise relaxation
            lambda: pl.col("cat").str.slice(0, 2) == "be",
            lambda: pl.col("id").hash() % 3 == 0,
            lambda: pl.col("val").round(1) == 0.5,
        ]
        return rng.choice(leaves)()
    left, right = (_random_predicate(rng, depth - 1) for _ in range(2))
    choice = rng.random()
    if choice < 0.35:
        return left & right
    if choice < 0.6:
        return left | right
    if choice < 0.75:
        return ~left
    if choice < 0.85:
        return left ^ right
    horizontal: list[Callable[..., pl.Expr]] = [pl.all_horizontal, pl.any_horizontal]
    return rng.choice(horizontal)(left, right)


@pytest.mark.parametrize("schema", [RICH, None], ids=["schema", "no schema"])
def test_random_nested_predicates_are_sound(
    schema: pl.Schema | None, rich_uri: str, rich_frame: pl.DataFrame
) -> None:
    """No lowering of a randomly nested predicate may lose a row.

    Deep nesting is where a relaxation rule goes wrong, and the shape that
    breaks it is rarely one anybody would write by hand.
    """
    rng = random.Random(20260827)
    dataset = lance.dataset(rich_uri)
    lowered_count = 0
    for _ in range(150):
        predicate = _random_predicate(rng, rng.randrange(1, 4))
        lowered = to_lance_filter(predicate, schema=schema)
        if lowered is None:
            continue
        lowered_count += 1
        kept = set(rich_frame.filter(predicate)["id"].to_list())
        pushed = _ids(dataset, lowered.sql)
        assert kept <= pushed, f"{predicate}\n  lowered to {lowered.sql}"
        if lowered.exact:
            assert pushed == kept, f"{predicate}\n  lowered to {lowered.sql}"
    assert lowered_count > 50, "the generator stopped producing pushable predicates"


def test_untranslatable_predicate_does_not_raise() -> None:
    """Anything unexpected in the tree is a decline, never an exception.

    A UDF does serialize, so this is the walk declining an `AnonymousFunction`
    node rather than the serialization guard below.
    """
    opaque = pl.col("id").map_elements(lambda x: x, return_dtype=pl.Boolean)
    assert opaque.meta.serialize(format="json")
    assert to_lance_filter(opaque) is None


def test_a_predicate_that_will_not_serialize_declines() -> None:
    """The other way to end up with no tree: polars refuses to serialize it.

    A UDF closing over something unpicklable fails in `meta.serialize`, so the
    lowering never gets a tree to walk. It still costs only the pushdown.
    """
    lock = threading.Lock()
    predicate = pl.col("id").map_elements(
        lambda x: (lock, x)[1], return_dtype=pl.Boolean
    )
    with pytest.raises(pl.exceptions.ComputeError):
        predicate.meta.serialize(format="json")
    assert to_lance_filter(predicate) is None


# A node of the shape the walk expects, with one thing about it wrong. Polars
# does not emit these today; the IR is versioned and has changed shape between
# releases, and this is what the module promises to do when it next does.
MALFORMED: list[tuple[str, Json]] = [
    ("body is a list", {"BinaryExpr": ["not", "a", "dict"]}),
    ("function body is a list", {"Function": ["input"]}),
    ("operands missing", {"BinaryExpr": {"op": "Eq"}}),
    ("operator is not a name", {"BinaryExpr": {"op": 7, "left": 1, "right": 2}}),
    ("alias has no input", {"Alias": []}),
    ("input is not a list", {"Function": {"function": "Abs", "input": 3}}),
    (
        "dtype is not a name",
        {"Cast": {"dtype": {"Literal": 5}, "expr": {"Column": "a"}}},
    ),
    ("cast has no expression", {"Cast": {"dtype": {"Literal": "Int64"}}}),
    ("column is not a name", {"Column": 3}),
    ("node is not an object", ["BinaryExpr"]),
    ("node is empty", {}),
]


@pytest.mark.parametrize("node", [pytest.param(n, id=name) for name, n in MALFORMED])
def test_a_malformed_node_declines_rather_than_raising(node: Json) -> None:
    """A shape surprise costs the pushdown, never the query.

    `to_lance_filter` promises `None` for anything it cannot lower. Reading a
    node without first establishing its shape would raise `AttributeError` or
    `KeyError` out of that promise instead.
    """
    assert _Lowering(max_in_list=16).predicate(node) == (None, False)


@pytest.mark.parametrize("node", [pytest.param(n, id=name) for name, n in MALFORMED])
def test_a_malformed_node_declines_in_value_position(node: Json) -> None:
    """Value position has no relaxed form, so it declines by raising `_Decline`."""
    with pytest.raises(_Decline):
        _Lowering(max_in_list=16).value(node)


def test_lowering_is_pure_of_dataset_knowledge() -> None:
    """The lowering never touches the dataset; it works from the expression alone."""
    assert to_lance_filter(pl.col("nonexistent") == "x") is not None


# ---------------------------------------------------------------------------
# schema-directed cast elision
# ---------------------------------------------------------------------------


def test_a_float_column_is_not_cast_when_the_schema_says_so() -> None:
    """A redundant `CAST` costs Lance's scalar index, so drop it where we can."""
    predicate = pl.col("val") > 0.75
    assert to_lance_filter(predicate, schema=pl.Schema({"val": pl.Float64})) == (
        LanceFilter(
            sql="((`val` > 0.75) OR `val` < CAST('-inf' AS double))", exact=True
        )
    )


def test_a_float32_column_gets_a_bound_of_its_own_type() -> None:
    """A `double` bound makes Lance cast the column, which costs its index."""
    lowered = to_lance_filter(
        pl.col("val") <= 0.5, schema=pl.Schema({"val": pl.Float32})
    )
    assert lowered == LanceFilter(
        sql="((`val` <= 0.5) AND `val` >= CAST('-inf' AS float))", exact=True
    )


def test_an_integer_column_is_not_guarded_against_nan() -> None:
    schema = pl.Schema({"id": pl.Int64, "val": pl.Float64})
    assert to_lance_filter(pl.col("id") > 1.5, schema=schema) == LanceFilter(
        sql="(CAST(`id` AS double) > 1.5)", exact=True
    )


def test_float_columns_are_compared_with_nan_made_positive() -> None:
    schema = pl.Schema({"f": pl.Float64, "g": pl.Float64})
    lowered = to_lance_filter(pl.col("f") < pl.col("g"), schema=schema)
    nan = "CAST('NaN' AS double)"
    left, right = f"nanvl(`f`, {nan})", f"nanvl(`g`, {nan})"
    assert lowered == LanceFilter(
        sql=(
            f"(({left} < {right}) AND NOT (`f` IN (-0.0, 0.0) AND `g` IN (-0.0, 0.0)))"
        ),
        exact=True,
    )


def test_an_integer_column_is_still_cast_against_a_float_literal() -> None:
    """The promotion is load-bearing there: Lance refuses the mixed comparison."""
    schema = pl.Schema({"id": pl.Int64})
    assert to_lance_filter(pl.col("id") > 0.5, schema=schema) == LanceFilter(
        sql="(CAST(`id` AS double) > 0.5)", exact=True
    )


def test_an_unknown_column_keeps_the_cast() -> None:
    """A schema that does not mention the column must not change the answer."""
    predicate = pl.col("val") > 0.999
    assert to_lance_filter(predicate, schema=pl.Schema({"other": pl.Float64})) == (
        to_lance_filter(predicate)
    )


def test_concatenation_of_two_columns_needs_the_schema() -> None:
    """`+` is concatenation for text and addition otherwise.

    A string literal on either side settles it; two columns need the schema.
    """
    predicate = (pl.col("cat") + pl.col("cat")) == "betabeta"
    assert to_lance_filter(predicate, schema=pl.Schema({"cat": pl.String})) == (
        LanceFilter(sql="((`cat` || `cat`) = 'betabeta')", exact=True)
    )
    assert to_lance_filter(predicate) == LanceFilter(
        sql="((`cat` + `cat`) = 'betabeta')", exact=True
    )


def test_the_optimizer_promotion_cast_is_pushed_when_the_schema_allows_it() -> None:
    """Comparing an int column to a float one is rewritten before we see it.

    Polars' optimizer inserts `cast(Float64, strict=False)`, which the schema
    lets us spell as `TRY_CAST`.
    """
    predicate = pl.col("id").cast(pl.Float64, strict=False) > pl.col("val")
    assert to_lance_filter(predicate) is None
    schema = pl.Schema({"id": pl.Int64, "val": pl.Float64})
    assert to_lance_filter(predicate, schema=schema) == (
        LanceFilter(
            sql=(
                "((TRY_CAST(`id` AS double) > nanvl(`val`, CAST('NaN' AS double)))"
                " AND NOT (TRY_CAST(`id` AS double) IN (-0.0, 0.0)"
                " AND `val` IN (-0.0, 0.0)))"
            ),
            exact=True,
        )
    )


def test_float_arithmetic_lance_computes_differently_declines() -> None:
    schema = pl.Schema({"f64": pl.Float64, "f32": pl.Float32, "i32": pl.Int32})
    for predicate in (
        # `(-inf) ** 0.5` is NaN in Polars and `inf` in Lance.
        (pl.col("f64") ** 0.5) > 0.5,
        # Polars' kernel for a fractional literal divisor disagrees with its own.
        (pl.col("f64") % 0.5) == 0.25,
        # A Float32 is computed as one, which a wider type undoes.
        (pl.col("f32") % pl.col("i32")) == 1,
    ):
        assert to_lance_filter(predicate, schema=schema) is None


def test_floor_division_declines_where_lance_divides_differently() -> None:
    schema = pl.Schema({"i": pl.Int64, "u": pl.UInt64, "f": pl.Float64})
    for predicate in (
        # `min // -1` wraps in Polars and fails the scan in Lance, and a column
        # divisor may be -1.
        (pl.col("i") // -1) == 1,
        (pl.col("i") // pl.col("i")) == 1,
        # Lance divides a UInt64 as a decimal.
        (pl.col("u") // 2) == 1,
        # Lance's `trunc` loses the sign of `-0.0`.
        (pl.col("f") // 2.0) == 1,
    ):
        assert to_lance_filter(predicate, schema=schema) is None


def test_logarithms_decline_where_lance_is_an_ulp_off() -> None:
    schema = pl.Schema({"f32": pl.Float32, "f64": pl.Float64})
    for predicate in (pl.col("f32").log() > 0, pl.col("f64").log10() > 0):
        assert to_lance_filter(predicate, schema=schema) is None


def test_non_strict_casts_without_an_exact_try_cast_spelling_decline() -> None:
    schema = pl.Schema({"id": pl.Int64, "cat": pl.String, "val": pl.Float64})
    for predicate in (
        pl.col("val").cast(pl.String, strict=False) == "0.5",
        pl.col("cat").cast(pl.Boolean, strict=False) == True,  # noqa: E712
        pl.col("cat").cast(pl.Date, strict=False).is_null(),
        pl.col("id").cast(pl.Int32, strict=False) > 1,
    ):
        assert to_lance_filter(predicate, schema=schema) is None


def test_the_schema_does_not_change_which_rows_survive(
    rich_uri: str, rich_frame: pl.DataFrame
) -> None:
    dataset = lance.dataset(rich_uri)
    schema = pl.Schema(rich_frame.schema)
    for predicate in (
        pl.col("val") > 0.5,
        pl.col("val").is_in([0.5, 0.75]),
        (pl.col("val") * 2) > 1.5,
        pl.col("id") > 0.5,
        (pl.col("val") > 0.5) & (pl.col("id") < 100),
        (pl.col("cat") + "!") == "beta!",
    ):
        lowered = to_lance_filter(predicate, schema=schema)
        assert lowered is not None
        pushed = _ids(dataset, lowered.sql)
        kept = set(rich_frame.filter(predicate)["id"].to_list())
        assert kept <= pushed
        if lowered.exact:
            assert pushed == kept


# ---------------------------------------------------------------------------
# Boolean composition: values where SQL and Polars part ways
# ---------------------------------------------------------------------------

NAN = float("nan")
NEG_NAN = math.copysign(math.nan, -1.0)

EDGES = pl.DataFrame(
    {
        "i": [1, 2, 3, None, 5, 0, -7, 7, 4, 6, -1],
        # A NaN with its sign bit set is what `0 / 0` gives on x86. Lance orders
        # it below every number, Polars above.
        "f": [
            1.0,
            NAN,
            None,
            -0.0,
            2.5,
            0.0,
            -1.5,
            1.5,
            NEG_NAN,
            NEG_NAN,
            float("inf"),
        ],
        "g": [0.0, 1.0, 2.0, 0.0, -0.0, -0.0, None, 2.0, NAN, 1.0, NEG_NAN],
        "b": [True, False, None, True, None, False, True, False, None, True, False],
        "l": [[1.0], [-0.0], None, [], [0.0], [1.0, None], [7.0], [2.0], [], [], []],
        # Strings a non-strict cast parses, and ones it nulls.
        "t": [
            "123",
            "+7",
            " 42 ",
            "12.9",
            "abc",
            None,
            "007",
            "NaN",
            "-NaN",
            "inf",
            "9" * 22,
        ],
    }
).with_columns(
    # The same floats one level down, where only the struct's type says so.
    s=pl.struct(x=pl.col("f")),
    fl=pl.concat_list(pl.col("f")),
    # Float32 remainders, which round differently in Float64.
    h=-pl.col("f").cast(pl.Float32) - 0.3,
    k=pl.col("i").cast(pl.Float32),
    r=(-pl.col("f").cast(pl.Float32) - 0.3) % pl.col("i").cast(pl.Float32),
)

EDGE_PREDICATES: list[tuple[str, pl.Expr]] = [
    ("is_in with a null", pl.col("i").is_in(pl.Series([1, None]))),
    ("is_in only null", pl.col("i").is_in(pl.Series([None], dtype=pl.Int64))),
    ("is_in empty", pl.col("i").is_in(pl.Series([], dtype=pl.Int64))),
    ("is_in a zero", pl.col("f").is_in([0.0, 2.5])),
    ("is_in a negative zero", pl.col("f").is_in([-0.0])),
    ("is_in nulls_equal", pl.col("i").is_in([1, None], nulls_equal=True)),
    ("is_in nulls_equal, no null", pl.col("i").is_in([1, 2], nulls_equal=True)),
    ("is_in nulls_equal, only null", pl.col("t").is_in([None], nulls_equal=True)),
    ("is_in nulls_equal, empty", pl.col("t").is_in([], nulls_equal=True)),
    ("is_in nulls_equal, zero", pl.col("f").is_in([0, None], nulls_equal=True)),
    ("float = 0", pl.col("f") == 0.0),
    ("float = -0", pl.col("f") == -0.0),
    ("float != 0", pl.col("f") != 0.0),
    ("float < 0", pl.col("f") < 0.0),
    ("float <= 0", pl.col("f") <= 0.0),
    ("float > -0", pl.col("f") > -0.0),
    ("float >= 0", pl.col("f") >= 0.0),
    ("zero on the left", pl.lit(0.0) < pl.col("f")),
    ("float = float", pl.col("f") == pl.col("g")),
    ("float != float", pl.col("f") != pl.col("g")),
    ("float < float", pl.col("f") < pl.col("g")),
    ("float <= float", pl.col("f") <= pl.col("g")),
    ("float > float", pl.col("f") > pl.col("g")),
    ("float >= float", pl.col("f") >= pl.col("g")),
    ("computed negative zero", (pl.col("f") * -1.0) == 0.0),
    # Not negation: Lance orders a NaN with its sign bit set below everything.
    ("computed float vs float", (pl.col("f") * 2.0) <= pl.col("g").abs()),
    ("is_between zeros", pl.col("f").is_between(-0.0, 0.0)),
    ("list contains a zero", pl.col("l").list.contains(0.0)),
    # An integer literal is compared as a float, as Polars' optimizer would.
    ("float >= int 0", pl.col("f") >= 0),
    ("float < int 1", pl.col("f") < 1),
    ("is_in an int zero", pl.col("f").is_in([0, 5])),
    ("list contains an int zero", pl.col("l").list.contains(0)),
    # Nested floats get their type from the struct or list type in the schema.
    ("struct field = 0", pl.col("s").struct.field("x") == 0),
    ("struct field > 1.0", pl.col("s").struct.field("x") > 1.0),
    ("struct field < float", pl.col("s").struct.field("x") < pl.col("g")),
    ("list element = 0", pl.col("fl").list.get(0, null_on_oob=True) == 0),
    (
        "list element <= float",
        pl.col("fl").list.get(0, null_on_oob=True) <= pl.col("g"),
    ),
    ("null literal", pl.lit(None, dtype=pl.Boolean)),
    ("modulo", (pl.col("i") % 2) == 1),
    ("modulo of a negation", (-pl.col("i") % 2) == 1),
    ("modulo by a negative", (pl.col("i") % -3) == -1),
    ("modulo by zero", (pl.col("i") % 0).is_null()),
    ("floor division", (pl.col("i") // 3) == -3),
    ("floor division by a negative", (pl.col("i") // -2) >= 1),
    ("floor division by zero", (pl.col("i") // 0).is_null()),
    ("modulo by a column", (pl.col("i") % (pl.col("i") - 2)) == 1),
    ("float modulo", (pl.col("f") % pl.col("g")) > 0.5),
    ("float modulo by a literal", (pl.col("f") % -2) < -0.25),
    # A remainder of `-0.0` would turn the quotient negative.
    ("float modulo of a zero", (1.0 / (pl.col("f") % 2.0)) > 0),
    ("float modulo by an integer", (pl.col("f") % pl.col("i")) == 0),
    ("float32 modulo", (pl.col("h") % pl.col("k")) == pl.col("r")),
    # Polars compares a Float32 to a bare literal as a Float32, to a Float64 literal
    # as a Float64.
    ("float32 against a bare literal", pl.col("h").abs() > 0.3),
    ("float32 against a float64 literal", pl.col("h") >= pl.lit(-0.3, pl.Float64)),
    ("nan comparison", pl.col("f") > 1.0),
    ("nan below", pl.col("f") < 1.0),
    ("nan at most", pl.col("f") <= 2.5),
    ("nan at least", pl.col("f") >= -1.5),
    ("nan between", pl.col("f").is_between(-10.0, 10.0)),
    ("nan against a computed value", (pl.col("f") * 2.0) > 1.0),
    ("min_horizontal skips nan", pl.min_horizontal("f", "g") < 1.0),
    ("max_horizontal skips nan", pl.max_horizontal("f", "g", "i") > 1.0),
    ("max_horizontal of nans", pl.max_horizontal("f", "g").is_nan()),
    ("min_horizontal with a null", pl.min_horizontal(pl.col("f"), None) >= 0.0),
    ("sqrt", pl.col("f").sqrt() >= 1.0),
    ("sqrt of an integer", pl.col("i").sqrt() < 2.0),
    ("cbrt", pl.col("f").cbrt() < 1.0),
    ("ln", pl.col("f").log() <= 0.0),
    ("fractional power", (pl.col("i") ** 0.5) > 1.5),
    ("fractional power of a negative", (pl.col("i") ** 0.5).is_nan()),
    ("whole float power", (pl.col("i") ** 2.0) >= 4.0),
    ("kleene or", pl.col("b") | (pl.col("i") > 2)),
    ("try_cast string to int", pl.col("t").cast(pl.Int64, strict=False) == 7),
    ("try_cast string to null", pl.col("t").cast(pl.Int64, strict=False).is_null()),
    ("try_cast string to float", pl.col("t").cast(pl.Float64, strict=False) < 0.0),
    ("try_cast float to int", pl.col("f").cast(pl.Int64, strict=False) == 1),
    ("try_cast int to float", pl.col("i").cast(pl.Float64, strict=False) > 2.5),
    ("try_cast float to bool", pl.col("f").cast(pl.Boolean, strict=False) == False),  # noqa: E712
    ("try_cast bool to int", pl.col("b").cast(pl.Int64, strict=False) == 1),
    ("try_cast int to string", pl.col("i").cast(pl.String, strict=False) == "5"),
]


@pytest.fixture(scope="module")
def edges_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    uri = str(tmp_path_factory.mktemp("edges") / "edges.lance")
    lance.write_dataset(EDGES.with_row_index("row").to_arrow(), uri)
    return uri


def _same(predicate: pl.Expr) -> pl.Expr:
    return predicate


@pytest.mark.parametrize(
    "negate",
    [pytest.param(_same, id="plain"), pytest.param(operator.inv, id="negated")],
)
@pytest.mark.parametrize(
    "predicate", [pytest.param(e, id=name) for name, e in EDGE_PREDICATES]
)
@pytest.mark.parametrize("typed", [True, False], ids=["schema", "no schema"])
def test_exact_lowerings_survive_negation(
    predicate: pl.Expr,
    negate: Callable[[pl.Expr], pl.Expr],
    typed: bool,  # noqa: FBT001 - a pytest parameter
    edges_uri: str,
) -> None:
    """An exact lowering must agree on every row, including under `NOT`.

    Agreeing on which rows a filter keeps is not enough: SQL may say null where
    Polars says false, which only shows once the result is negated. The truth
    is the predicate evaluated per row: `filter` would let the optimizer fold
    `~is_in([])` away first, keeping the null row the evaluation drops.
    """
    predicate = negate(predicate)
    lowered = to_lance_filter(predicate, schema=EDGES.schema if typed else None)
    if lowered is None and not typed:
        # Two untyped columns decline; the schema'd run holds the lowering to it.
        return
    assert lowered is not None, "expected this predicate to lower"
    assert lowered.exact
    table = lance.dataset(edges_uri).to_table(columns=["row"], filter=lowered.sql)
    pushed = set(table["row"].to_pylist())
    evaluated = EDGES.with_row_index("row").with_columns(predicate.alias("keep"))
    kept = set(evaluated.filter(pl.col("keep") == True)["row"].to_list())  # noqa: E712
    assert pushed == kept, lowered.sql
