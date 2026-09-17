"""Prefilter and postfilter, kept distinct.

Lance can restrict a search two ways, and they answer different questions. A
*prefilter* narrows the rows the search ranks, so it still returns ``k`` of
them. A *postfilter* -- which is what a downstream ``.filter()`` is -- keeps
whatever survives out of the ``k`` already ranked, and so may return fewer.

Silently swapping one for the other changes the answer without changing the
query, so the tests below pin all three of: that each does its own thing, that a
downstream filter is never promoted to a prefilter, and that a prefilter Lance
cannot take fails rather than quietly becoming a postfilter.
"""

from __future__ import annotations

import io
import math
import warnings
from typing import TYPE_CHECKING

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from polars_pylance import scan_lance

if TYPE_CHECKING:
    from conftest import ScannerCall

DIM = 32
ROWS = 2_000
K = 10


def _search(
    uri: str,
    query: list[float],
    *,
    prefilter: str | pl.Expr | None = None,
    **tuning: str | int | bool,
) -> pl.LazyFrame:
    """A k-nearest `scan_lance`, with any search tuning added to `nearest`."""
    nearest = {"column": "vector", "q": query, "k": K, **tuning}
    return scan_lance(uri, nearest=nearest, prefilter=prefilter)


@pytest.fixture(scope="session")
def split_uri(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, list[float]]:
    """A dataset where prefilter and postfilter must disagree.

    Every ``cat='near'`` row sits at the origin and every ``cat='far'`` row sits
    far from it, so the true top-k for a query at the origin is entirely
    ``'near'``. Asking for ``'far'`` then separates the two semantics as widely
    as they can be separated: prefiltering gives k rows, postfiltering gives 0.
    """
    rng = np.random.default_rng(0)
    cats = ["near" if i % 2 == 0 else "far" for i in range(ROWS)]
    vectors = np.array(
        [
            rng.normal(0.0, 0.01, DIM) if i % 2 == 0 else rng.normal(5.0, 0.01, DIM)
            for i in range(ROWS)
        ],
        dtype=np.float32,
    )
    uri = str(tmp_path_factory.mktemp("split") / "vectors.lance")
    lance.write_dataset(
        pa.table(
            {
                "id": pa.array(np.arange(ROWS)),
                "cat": pa.array(cats),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.ravel()), DIM
                ),
            }
        ),
        uri,
    )
    return uri, [0.0] * DIM


@pytest.fixture(scope="session")
def prefiltered_ids(split_uri: tuple[str, list[float]]) -> list[int]:
    """Ground truth, from Lance's own ``prefilter=True``."""
    uri, query = split_uri
    scanner = lance.dataset(uri).scanner(
        nearest={"column": "vector", "q": query, "k": K},
        filter="cat = 'far'",
        prefilter=True,
        columns=["id"],
    )
    ids: list[int] = []
    for value in scanner.to_table()["id"].to_pylist():
        assert isinstance(value, int)
        ids.append(value)
    return ids


# -- the two semantics ------------------------------------------------------


@pytest.mark.parametrize(
    "prefilter",
    [
        pytest.param("cat = 'far'", id="sql"),
        pytest.param(pl.col("cat") == "far", id="expr"),
    ],
)
@pytest.mark.parametrize("variant", ["indexed", "exact search", "serialized"])
def test_prefilter_ranks_only_the_rows_it_admits(
    split_uri: tuple[str, list[float]],
    prefiltered_ids: list[int],
    prefilter: str | pl.Expr,
    variant: str,
) -> None:
    """Exactly k rows come back, all matching, ranked among themselves.

    `use_index=False` is the ground-truth search, and a serialized plan must keep
    the prefilter it was built with.
    """
    uri, query = split_uri
    tuning = {"use_index": False} if variant == "exact search" else {}
    lf = _search(uri, query, prefilter=prefilter, **tuning).select(
        "id", "cat", "_distance"
    )
    if variant == "serialized":
        lf = pl.LazyFrame.deserialize(io.BytesIO(lf.serialize()))
    out = lf.collect(engine="streaming")
    assert set(out["cat"].to_list()) == {"far"}
    assert out["_distance"].is_sorted()
    assert out["id"].to_list() == prefiltered_ids


def test_downstream_filter_is_a_postfilter(
    split_uri: tuple[str, list[float]],
) -> None:
    """It filters the ranked result, so it may return fewer than k -- here none."""
    uri, query = split_uri
    out = (
        _search(uri, query)
        .filter(pl.col("cat") == "far")
        .select("id")
        .collect(engine="streaming")
    )
    assert out.height < K
    assert out.height == 0  # the whole top-k was 'near'


def test_prefilter_composes_with_a_downstream_filter(
    split_uri: tuple[str, list[float]], prefiltered_ids: list[int]
) -> None:
    """Prefilter chooses what is ranked; the later filter trims that ranking."""
    uri, query = split_uri
    cutoff = ROWS // 2
    out = (
        _search(uri, query, prefilter="cat = 'far'")
        .filter(pl.col("id") < cutoff)
        .select("id")
        .collect(engine="streaming")
    )
    assert out["id"].to_list() == [i for i in prefiltered_ids if i < cutoff]


# -- what actually reaches Lance -------------------------------------------


def test_downstream_filter_is_never_pushed_as_a_prefilter(
    split_uri: tuple[str, list[float]], scanner_calls: list[ScannerCall]
) -> None:
    """The guarantee this module exists for, pinned at the Lance call.

    A downstream filter may be pushed down -- but only into the postfilter
    position, which is what leaving ``prefilter`` unset means. If Lance ever
    changed that default, this is where it would show.
    """
    uri, query = split_uri
    (
        _search(uri, query)
        .filter(pl.col("cat") == "far")
        .select("id")
        .collect(engine="streaming")
    )
    assert not any(c.prefilter for c in scanner_calls), (
        f"a downstream filter was pushed as a prefilter: {scanner_calls}"
    )


# -- an unsupported prefilter fails ----------------------------------------


@pytest.mark.parametrize(
    ("predicate", "message"),
    [
        (pl.col("id").hash() % 2 == 0, "does not translate"),
        (pl.col("cat").is_in(["far"]) & (pl.col("id").rank() > 0), "only partly"),
    ],
    ids=["unlowerable", "partial"],
)
def test_prefilter_polars_expression_must_lower_exactly(
    split_uri: tuple[str, list[float]],
    scanner_calls: list[ScannerCall],
    predicate: pl.Expr,
    message: str,
) -> None:
    """A prefilter that cannot be pushed whole is an error, not a postfilter.

    A pushed-down predicate is allowed to be relaxed because Polars finishes it.
    Nothing can finish a prefilter: the ranking has already happened. The refusal
    lands at the call site, before anything is read.
    """
    uri, query = split_uri
    with pytest.raises(ValueError, match=message):
        _search(uri, query, prefilter=predicate)
    assert scanner_calls == []


def test_lance_rejected_prefilter_propagates(
    split_uri: tuple[str, list[float]],
) -> None:
    """No warn-and-rescan for a prefilter: dropping it would rank other rows.

    A pushed-down predicate Lance refuses is retried without it, since Polars
    can still apply it. Doing that to a prefilter would silently search a wider
    set of rows, so the error is left to reach the caller.
    """
    uri, query = split_uri
    lf = _search(uri, query, prefilter="no_such_col = 1").select("id")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Exception, match=r"no_such_col|No field named"):
            lf.collect(engine="streaming")
    assert not [w for w in caught if "scanning without it" in str(w.message)]


# -- the rest of the scan still works --------------------------------------


def test_limit_truncates_the_prefiltered_ranking(
    split_uri: tuple[str, list[float]], prefiltered_ids: list[int]
) -> None:
    """`head()` takes the nearest few, rather than re-running the search."""
    uri, query = split_uri
    out = (
        _search(uri, query, prefilter="cat = 'far'")
        .select("id")
        .head(3)
        .collect(engine="streaming")
    )
    assert out["id"].to_list() == prefiltered_ids[:3]


def test_prefilter_without_a_search_restricts_the_scan(
    split_uri: tuple[str, list[float]],
) -> None:
    """With nothing to rank, a prefilter is simply a filter the caller pushed."""
    uri, _ = split_uri
    out = (
        scan_lance(uri, prefilter="cat = 'far'")
        .select("id")
        .collect(engine="streaming")
    )
    assert out.height == ROWS // 2


def test_search_tuning_survives_a_prefilter(
    split_uri: tuple[str, list[float]], scanner_calls: list[ScannerCall]
) -> None:
    """Search tuning and the prefilter reach Lance side by side, untouched."""
    uri, query = split_uri
    _search(
        uri,
        query,
        metric="cosine",
        nprobes=4,
        use_index=False,
        prefilter="cat = 'far'",
    ).select("id").collect(engine="streaming")

    pushed = [c for c in scanner_calls if c.nearest is not None]
    assert pushed, f"no nearest reached Lance: {scanner_calls}"
    nearest = pushed[-1].nearest
    assert nearest is not None
    assert (nearest["column"], nearest["k"]) == ("vector", K)
    assert (nearest["metric"], nearest["nprobes"], nearest["use_index"]) == (
        "cosine",
        4,
        False,
    )
    assert pushed[-1].filter == "cat = 'far'"
    assert pushed[-1].prefilter is True


# -- translated against the dataset's column types --------------------------


@pytest.fixture(scope="session")
def float_uri(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, pl.DataFrame]:
    """Float columns holding what Lance and Polars order differently.

    A NaN with its sign bit set (what `0 / 0` gives on x86) and `-0.0`. Every
    vector is equal, so a search with `k` covering the table ranks all rows and
    the prefilter alone decides which come back.
    """
    neg_nan = math.copysign(math.nan, -1.0)
    frame = pl.DataFrame(
        {
            "id": list(range(8)),
            "score": [1.0, neg_nan, math.nan, -0.0, 0.0, -2.0, None, 3.0],
            "floor": [0.0, 1.0, neg_nan, 0.0, -0.0, math.nan, 1.0, None],
        }
    )
    table = frame.to_arrow().append_column(
        "vector",
        pa.FixedSizeListArray.from_arrays(pa.array([0.0] * (8 * 2), pa.float32()), 2),
    )
    uri = str(tmp_path_factory.mktemp("floats") / "floats.lance")
    lance.write_dataset(table, uri)
    return uri, frame


@pytest.mark.parametrize(
    "prefilter",
    [
        pytest.param(pl.col("score") >= 0, id="int literal"),
        pytest.param(pl.col("score") > 0.5, id="float literal"),
        pytest.param(pl.col("score") < 1, id="below"),
        pytest.param(pl.col("score") == 0, id="zero"),
        pytest.param(pl.col("score") > pl.col("floor"), id="column against column"),
        pytest.param(~(pl.col("score") <= pl.col("floor")), id="negated columns"),
    ],
)
def test_expression_prefilter_keeps_polars_float_semantics(
    float_uri: tuple[str, pl.DataFrame], prefilter: pl.Expr
) -> None:
    """Without the column types, `score >= 0` would drop the negative NaN and `-0.0`."""
    uri, frame = float_uri
    nearest = {
        "column": "vector",
        "q": [0.0, 0.0],
        "k": frame.height,
        "use_index": False,
    }
    got = (
        scan_lance(uri, nearest=nearest, prefilter=prefilter)
        .select("id")
        .collect(engine="streaming")
    )
    evaluated = frame.with_columns(prefilter.alias("keep"))
    kept = evaluated.filter(pl.col("keep") == True)["id"].to_list()  # noqa: E712
    assert sorted(got["id"].to_list()) == sorted(kept)


def test_expression_prefilter_is_lowered_with_the_schema(
    float_uri: tuple[str, pl.DataFrame], scanner_calls: list[ScannerCall]
) -> None:
    uri, _ = float_uri
    nearest = {"column": "vector", "q": [0.0, 0.0], "k": 3}
    lf = scan_lance(uri, nearest=nearest, prefilter=pl.col("score") > 1)
    assert scanner_calls == [], "scan_lance itself must not read the dataset"

    lf.select("id").collect(engine="streaming")
    # The schema lookup opens a scanner too, without a filter.
    searches = [c for c in scanner_calls if c.filter is not None]
    assert searches
    assert all(c.prefilter for c in searches)
    assert all(
        c.filter == "((`score` > 1.0) OR `score` < CAST('-inf' AS double))"
        for c in searches
    )


def test_prefilter_refused_only_with_the_schema_fails_at_collect(
    float_uri: tuple[str, pl.DataFrame],
) -> None:
    """A float-to-string cast has no exact spelling, which only the types show."""
    uri, _ = float_uri
    nearest = {"column": "vector", "q": [0.0, 0.0], "k": 3}
    prefilter = pl.col("score").cast(pl.String, strict=False) == "1.0"
    lf = scan_lance(uri, nearest=nearest, prefilter=prefilter)
    with pytest.raises(Exception, match="prefilter does not translate"):
        lf.select("id").collect(engine="streaming")
