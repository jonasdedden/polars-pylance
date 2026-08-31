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
import pickle
import warnings
from typing import TYPE_CHECKING

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from polars_pylance import LanceScanSpec, scan_lance

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
    metric: str | None = None,
    nprobes: int | None = None,
    use_index: bool | None = None,
) -> pl.LazyFrame:
    """A k-nearest `scan_lance`, with the `nearest` dict lifted out of the tests.

    The search tuning is spelled out rather than forwarded as `**kwargs`, so a
    misspelled knob is a type error here instead of a key Lance ignores.
    """
    # `scan_lance` takes `nearest` as a `dict[str, Any]`; spelling the values
    # out keeps this side of the call checked.
    nearest: dict[str, str | int | list[float]] = {
        "column": "vector",
        "q": query,
        "k": K,
    }
    if metric is not None:
        nearest["metric"] = metric
    if nprobes is not None:
        nearest["nprobes"] = nprobes
    if use_index is not None:
        nearest["use_index"] = use_index
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


def test_prefilter_ranks_only_the_rows_it_admits(
    split_uri: tuple[str, list[float]], prefiltered_ids: list[int]
) -> None:
    """k rows come back, all of them matching, ranked among themselves."""
    uri, query = split_uri
    out = (
        _search(uri, query, prefilter="cat = 'far'")
        .select("id", "cat", "_distance")
        .collect(engine="streaming")
    )
    assert out.height == K
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


def test_prefilter_and_postfilter_answer_differently(
    split_uri: tuple[str, list[float]],
) -> None:
    """The same predicate, the two positions, two different answers."""
    uri, query = split_uri
    pre = _search(uri, query, prefilter="cat = 'far'").collect(engine="streaming")
    post = (
        _search(uri, query).filter(pl.col("cat") == "far").collect(engine="streaming")
    )
    assert pre.height != post.height


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


def test_prefilter_reaches_lance_as_prefilter(
    split_uri: tuple[str, list[float]], scanner_calls: list[ScannerCall]
) -> None:
    uri, query = split_uri
    _search(uri, query, prefilter="cat = 'far'").select("id").collect(
        engine="streaming"
    )

    pushed = [c for c in scanner_calls if c.filter is not None]
    assert pushed, f"no filter reached Lance: {scanner_calls}"
    assert all(c.filter == "cat = 'far'" for c in pushed)
    assert all(c.prefilter is True for c in pushed)


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
    split_uri: tuple[str, list[float]], predicate: pl.Expr, message: str
) -> None:
    """A prefilter that cannot be pushed whole is an error, not a postfilter.

    A pushed-down predicate is allowed to be relaxed because Polars finishes it.
    Nothing can finish a prefilter: the ranking has already happened.
    """
    uri, query = split_uri
    with pytest.raises(ValueError, match=message):
        _search(uri, query, prefilter=predicate)


def test_prefilter_expression_is_rejected_before_any_read(
    split_uri: tuple[str, list[float]], scanner_calls: list[ScannerCall]
) -> None:
    """The refusal lands at the call site, not deep in a collect."""
    uri, query = split_uri
    with pytest.raises(ValueError, match="does not translate"):
        _search(
            uri,
            query,
            prefilter=pl.col("id").hash() % 2 == 0,
        )
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


def test_exact_expression_prefilter_matches_the_sql_one(
    split_uri: tuple[str, list[float]], prefiltered_ids: list[int]
) -> None:
    uri, query = split_uri
    out = (
        _search(
            uri,
            query,
            prefilter=pl.col("cat") == "far",
        )
        .select("id")
        .collect(engine="streaming")
    )
    assert out["id"].to_list() == prefiltered_ids


def test_exact_search_honours_the_prefilter(
    split_uri: tuple[str, list[float]], prefiltered_ids: list[int]
) -> None:
    """`use_index=False` is the ground-truth mode; the prefilter still applies."""
    uri, query = split_uri
    out = (
        _search(uri, query, use_index=False, prefilter="cat = 'far'")
        .select("id")
        .collect(engine="streaming")
    )
    assert out.height == K
    assert out["id"].to_list() == prefiltered_ids


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
    assert nearest.column == "vector"
    assert nearest.k == K
    assert nearest.metric == "cosine"
    assert nearest.nprobes == 4
    assert nearest.use_index is False
    assert pushed[-1].filter == "cat = 'far'"
    assert pushed[-1].prefilter is True


# -- it still ships ---------------------------------------------------------


def test_prefilter_survives_serialization(
    split_uri: tuple[str, list[float]], prefiltered_ids: list[int]
) -> None:
    uri, query = split_uri
    lf = _search(uri, query, prefilter="cat = 'far'").select("id")
    restored = pl.LazyFrame.deserialize(io.BytesIO(lf.serialize()))
    assert restored.collect(engine="streaming")["id"].to_list() == prefiltered_ids


def test_spec_with_prefilter_is_picklable(split_uri: tuple[str, list[float]]) -> None:
    """A lowered prefilter is a plain string, so the spec still compares equal."""
    uri, _ = split_uri
    spec = LanceScanSpec(uri=uri, prefilter="cat = 'far'")
    assert pickle.loads(pickle.dumps(spec)) == spec
