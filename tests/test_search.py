"""Lance's index-backed search, reached through a scan.

These are the capabilities that motivate pola-rs/polars#12389: vector search and
full-text search are Lance features that no generic Arrow reader can expose,
because they are properties of the scan rather than expressions over the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from polars_pylance import scan_lance

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import ScannerCall

DIM = 16
ROWS = 512


@pytest.fixture
def vector_uri(tmp_path: Path) -> tuple[str, list[float]]:
    """A dataset with a fixed-size-list vector column, plus a query vector."""
    rng = np.random.default_rng(0)
    vectors = rng.random((ROWS, DIM), dtype=np.float32)
    table = pa.table(
        {
            "id": pa.array(np.arange(ROWS)),
            "cat": pa.array(["a", "b", "c", "d"] * (ROWS // 4)),
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), DIM),
        }
    )
    uri = str(tmp_path / "vectors.lance")
    lance.write_dataset(table, uri)
    return uri, vectors[42].tolist()


def test_nearest_returns_k_rows_ranked_by_distance(
    vector_uri: tuple[str, list[float]],
) -> None:
    uri, query = vector_uri
    out = (
        scan_lance(uri, nearest={"column": "vector", "q": query, "k": 5})
        .select("id", "_distance")
        .collect(engine="streaming")
    )
    assert out.height == 5
    # the query vector is row 42 itself, so it must come back first at distance 0
    assert out["id"][0] == 42
    assert out["_distance"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["_distance"].is_sorted()


def test_nearest_composes_with_polars_operations(
    vector_uri: tuple[str, list[float]],
) -> None:
    """The ANN result is an ordinary LazyFrame, so the rest of polars applies."""
    uri, query = vector_uri
    out = (
        scan_lance(uri, nearest={"column": "vector", "q": query, "k": 40})
        .filter(pl.col("cat") == "a")
        .select("id", "cat")
        .collect(engine="streaming")
    )
    assert 0 < out.height <= 40
    assert (out["cat"] == "a").all()


def test_nearest_is_lazy(vector_uri: tuple[str, list[float]]) -> None:
    uri, query = vector_uri
    lf = scan_lance(uri, nearest={"column": "vector", "q": query, "k": 3})
    assert isinstance(lf, pl.LazyFrame)
    assert "vector" in lf.collect_schema().names()


def test_full_text_query_matches_indexed_terms(tmp_path: Path) -> None:
    uri = str(tmp_path / "fts.lance")
    dataset = lance.write_dataset(
        pa.table(
            {
                "id": pa.array([1, 2, 3, 4]),
                "text": pa.array(
                    [
                        "polars is fast",
                        "lance stores vectors",
                        "streaming query engine",
                        "vector search in lance",
                    ]
                ),
            }
        ),
        uri,
    )
    dataset.create_scalar_index("text", "INVERTED")

    out = (
        scan_lance(uri, full_text_query="lance")
        .select("id")
        .collect(engine="streaming")
    )
    assert sorted(out["id"].to_list()) == [2, 4]


def test_limit_is_pushed_into_the_scan(
    tmp_path: Path, scanner_calls: list[ScannerCall]
) -> None:
    """`head()` stops the scan rather than reading to the end.

    This is the other half of pola-rs/polars#12389: a generic Arrow reader has
    nowhere to put the row limit, so it materialises far more than asked for.
    """
    uri = str(tmp_path / "rows.lance")
    lance.write_dataset(
        pa.table({"id": pa.array(np.arange(100_000))}), uri, max_rows_per_file=10_000
    )

    out = scan_lance(uri).head(7).collect(engine="streaming")

    assert out.height == 7
    assert any(call.limit == 7 for call in scanner_calls), (
        f"no scanner call carried the row limit: {scanner_calls}"
    )
