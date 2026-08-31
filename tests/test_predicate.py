"""What the Polars to Lance SQL lowering does, and what it refuses to do.

A lowering must keep at least the rows the predicate keeps, and an `exact` one
must keep precisely them, since `scan_lance` then stops filtering. Three
families of tests hold it to that: shapes that pin the SQL, refusals, and
differential tests that run both against a real dataset.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import TYPE_CHECKING

import lance
import polars as pl
import pytest

from polars_pylance._predicate import (
    Json,
    LanceFilter,
    _Decline,
    _Lowering,
    to_lance_filter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
    ("is_in empty", pl.col("id").is_in([]), "FALSE"),
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
    ("modulo", (pl.col("id") % 2) == 0, "((`id` % 2) = 0)"),
    ("negate", -pl.col("id") < 0, "((- `id`) < 0)"),
    ("power", (pl.col("id") ** 2) > 9, "(power(`id`, 2) > 9)"),
    ("abs", pl.col("id").abs() > 3, "(abs(`id`) > 3)"),
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
    ("cast", pl.col("id").cast(pl.Float64) > 1, "(CAST(`id` AS double) > 1)"),
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
    # `eq_missing` against a non-null literal cannot be null-sensitive.
    ("eq_missing", pl.col("opt").eq_missing(3), "(`opt` = 3)"),
    ("alias is transparent", (pl.col("id") > 7).alias("x"), "(`id` > 7)"),
    ("quoted identifier", pl.col("odd name") > 1, "(`odd name` > 1)"),
    # Polars promotes to float to compare; Lance refuses the mixed comparison.
    ("int against float", pl.col("id") > 1.5, "(CAST(`id` AS double) > 1.5)"),
    ("true division", pl.col("id") / 2 > 1.5, "((CAST(`id` AS double) / 2) > 1.5)"),
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
]


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [pytest.param(e, s, id=name) for name, e, s in TRANSLATIONS],
)
def test_lowering_shape(predicate: pl.Expr, expected: str) -> None:
    lowered = to_lance_filter(predicate)
    assert lowered is not None
    assert lowered.sql == expected
    assert lowered.exact


DECLINED: list[tuple[str, pl.Expr]] = [
    ("python-level hash", pl.col("id").hash() % 2 == 0),
    ("aggregate in predicate", pl.col("val") > pl.col("val").mean()),
    ("when/then", pl.when(pl.col("id") > 1).then(True).otherwise(False)),
    ("str.slice", pl.col("cat").str.slice(0, 2) == "be"),
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
    # `IN` propagates NULL; `nulls_equal` asks for null to match null.
    ("is_in with nulls_equal", pl.col("opt").is_in([1, None], nulls_equal=True)),
    # Polars raises where `array_element` returns null.
    ("list.get, strict", pl.col("tags").list.get(0) == 3),
    # Polars breaks ties to even, Lance away from zero.
    ("round", pl.col("val").round(2) == 0.5),
    # Outside the domain Polars yields NaN and Lance NULL, which sort apart.
    ("sqrt", pl.col("val").sqrt() > 0.5),
    ("fractional power", (pl.col("val") ** 0.5) > 0.5),
    # `date_part('epoch', ...)` keeps the fraction; `dt.epoch` truncates.
    ("dt.epoch", pl.col("ts").dt.epoch("s") > 0),
    # `date_trunc` has no spelling for a multiple of a unit.
    (
        "dt.truncate, multiple",
        pl.col("ts").dt.truncate("2d") == dt.datetime(2024, 1, 2),
    ),
    ("floor division", (pl.col("id") // 2) == 1),
    # A narrowing integer cast raises in Polars and wraps in Lance.
    ("narrowing cast", pl.col("id").cast(pl.Int32) > 1),
    # `concat_ws` skips nulls, so it cannot spell a null-propagating join.
    (
        "concat_str with a separator, keeping nulls",
        pl.concat_str(pl.col("cat"), pl.col("text"), separator="-") == "x",
    ),
    # A non-strict cast yields null in Polars and fails the scan in Lance --
    # except for the widening one below, which the schema has to confirm.
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
    assert to_lance_filter(pl.col("id").is_in(list(range(10)))) is not None
    assert to_lance_filter(pl.col("id").is_in(list(range(10))), max_in_list=5) is None


# ---------------------------------------------------------------------------
# relaxation: what happens when only part of a predicate lowers
# ---------------------------------------------------------------------------

UNTRANSLATABLE = pl.col("cat").str.slice(0, 2) == "beta"


def test_conjunct_is_dropped() -> None:
    """An AND keeps whatever lowered; the engine filters the rest."""
    lowered = to_lance_filter((pl.col("id") > 5) & UNTRANSLATABLE)
    assert lowered == LanceFilter(sql="(`id` > 5)", exact=False)


def test_deep_conjunct_is_dropped() -> None:
    predicate = (
        (pl.col("id") > 5)
        & (pl.col("val") < 0.9)
        & UNTRANSLATABLE
        & pl.col("text").str.starts_with("row")
    )
    lowered = to_lance_filter(predicate)
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
    lowered = to_lance_filter(predicate)
    assert lowered is not None
    assert not lowered.exact
    assert lowered.sql == "((`id` > 5) OR (CAST(`val` AS double) < 0.1))"


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


@pytest.mark.parametrize(
    "predicate", [pytest.param(p, id=str(p)[:60]) for p in DIFFERENTIAL]
)
def test_lance_keeps_every_row_polars_keeps(
    predicate: pl.Expr, rich_uri: str, rich_frame: pl.DataFrame
) -> None:
    lowered = to_lance_filter(predicate)
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


def test_random_nested_predicates_are_sound(
    rich_uri: str, rich_frame: pl.DataFrame
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
        lowered = to_lance_filter(predicate)
        if lowered is None:
            continue
        lowered_count += 1
        kept = set(rich_frame.filter(predicate)["id"].to_list())
        pushed = _ids(dataset, lowered.sql)
        assert kept <= pushed, f"{predicate}\n  lowered to {lowered.sql}"
        if lowered.exact:
            assert pushed == kept, f"{predicate}\n  lowered to {lowered.sql}"
    assert lowered_count > 60, "the generator stopped producing pushable predicates"


def test_untranslatable_predicate_does_not_raise() -> None:
    """Anything unexpected in the tree is a decline, never an exception."""
    opaque = pl.col("id").map_elements(lambda x: x, return_dtype=pl.Boolean)
    assert to_lance_filter(opaque) is None


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
    assert to_lance_filter(pl.col("nonexistent") > 1) is not None


# ---------------------------------------------------------------------------
# schema-directed cast elision
# ---------------------------------------------------------------------------


def test_a_float_column_is_not_cast_when_the_schema_says_so() -> None:
    """A redundant `CAST` costs Lance's scalar index, so drop it where we can."""
    predicate = pl.col("val") > 0.999
    assert to_lance_filter(predicate) == LanceFilter(
        sql="(CAST(`val` AS double) > 0.999)", exact=True
    )
    assert to_lance_filter(predicate, schema=pl.Schema({"val": pl.Float64})) == (
        LanceFilter(sql="(`val` > 0.999)", exact=True)
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

    Polars' optimizer inserts `cast(Float64, strict=False)`. That is normally
    declined, but widening a number to a float cannot produce the null that
    makes a non-strict cast unsafe.
    """
    predicate = pl.col("id").cast(pl.Float64, strict=False) > pl.col("val")
    assert to_lance_filter(predicate) is None
    assert to_lance_filter(predicate, schema=pl.Schema({"id": pl.Int64})) == (
        LanceFilter(sql="(CAST(`id` AS double) > `val`)", exact=True)
    )
    # A string source can produce null there, so it stays declined.
    assert to_lance_filter(predicate, schema=pl.Schema({"id": pl.String})) is None


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
