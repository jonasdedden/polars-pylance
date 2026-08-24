"""scan_lance correctness: both implementations must match an eager read."""

from __future__ import annotations

import lance
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pylance import LanceScanOptions, scan_lance, scan_lance_fragments


def test_schema_matches_dataset(lance_uri: str) -> None:
    lf = scan_lance(lance_uri)
    expected = pl.from_arrow(lance.dataset(lance_uri).schema.empty_table())
    assert lf.collect_schema() == expected.schema  # type: ignore[union-attr]


def test_nothing_is_read_until_collect(
    lance_uri: str, frames_yielded: list[int]
) -> None:
    lf = scan_lance(lance_uri).filter(pl.col("val") > 0.5)
    assert frames_yielded[0] == 0
    lf.collect(engine="streaming")
    assert frames_yielded[0] > 0


def test_full_read(lance_uri: str, expected: pl.DataFrame) -> None:
    got = scan_lance(lance_uri).collect(engine="streaming")
    assert_frame_equal(got.sort("id"), expected.sort("id"))


def test_projection_and_filter(lance_uri: str, expected: pl.DataFrame) -> None:
    got = (
        scan_lance(lance_uri)
        .filter(pl.col("cat") == "b", pl.col("val") > 0.9)
        .select("id", "val")
        .collect(engine="streaming")
    )
    want = expected.filter(pl.col("cat") == "b", pl.col("val") > 0.9).select(
        "id", "val"
    )
    assert_frame_equal(got.sort("id"), want.sort("id"))


def test_head_stops_early(lance_uri: str, frames_yielded: list[int]) -> None:
    got = scan_lance(lance_uri).head(5).collect(engine="streaming")
    assert got.height == 5
    # 60k rows across 4 fragments at 25k/batch: a non-streaming reader would pull
    # every batch.
    assert frames_yielded[0] <= 4


def test_limit_after_filter(lance_uri: str, expected: pl.DataFrame) -> None:
    got = (
        scan_lance(lance_uri)
        .filter(pl.col("cat") == "c")
        .sort("id")
        .head(7)
        .collect(engine="streaming")
    )
    want = expected.filter(pl.col("cat") == "c").sort("id").head(7)
    assert_frame_equal(got, want)


def test_in_memory_engine_still_works(lance_uri: str, expected: pl.DataFrame) -> None:
    got = scan_lance(lance_uri).select(pl.col("val").sum()).collect(engine="in-memory")
    assert got.item() == pytest.approx(expected["val"].sum())


def test_predicate_pushdown_disabled_is_still_correct(
    lance_uri: str, expected: pl.DataFrame
) -> None:
    got = (
        scan_lance(lance_uri, predicate_pushdown=False)
        .filter(pl.col("val") > 0.5)
        .select(pl.len())
        .collect(engine="streaming")
    )
    assert got.item() == expected.filter(pl.col("val") > 0.5).height


def test_throughput_options(lance_uri: str, expected: pl.DataFrame) -> None:
    got = (
        scan_lance(lance_uri, options=LanceScanOptions.throughput())
        .select(pl.len())
        .collect(engine="streaming")
    )
    assert got.item() == expected.height


def test_with_row_id(lance_uri: str, expected: pl.DataFrame) -> None:
    lf = scan_lance(lance_uri, with_row_id=True)
    assert "_rowid" in lf.collect_schema()
    got = lf.select("_rowid", "id").collect(engine="streaming")
    assert got.height == expected.height
    assert got["_rowid"].n_unique() == expected.height


def test_projection_of_generated_column_only(lance_uri: str) -> None:
    got = (
        scan_lance(lance_uri, with_row_id=True)
        .select("_rowid")
        .collect(engine="streaming")
    )
    assert got.columns == ["_rowid"]
    assert got.height > 0


def test_single_fragment_subset(lance_uri: str, expected: pl.DataFrame) -> None:
    ids = [f.fragment_id for f in lance.dataset(lance_uri).get_fragments()]
    got = scan_lance(lance_uri, fragments=ids[:1]).collect(engine="streaming")
    assert 0 < got.height < expected.height


def test_unknown_fragment_raises(lance_uri: str) -> None:
    with pytest.raises(ValueError, match="no such fragment"):
        scan_lance(lance_uri, fragments=[9999]).collect(engine="streaming")


def test_fragments_shards_cover_dataset(lance_uri: str, expected: pl.DataFrame) -> None:
    shards = scan_lance_fragments(lance_uri)
    assert len(shards) == len(lance.dataset(lance_uri).get_fragments())
    got = pl.concat(shards).collect(engine="streaming")
    assert_frame_equal(got.sort("id"), expected.sort("id"))


def test_fragments_n_shards(lance_uri: str, expected: pl.DataFrame) -> None:
    shards = scan_lance_fragments(lance_uri, n_shards=2)
    assert len(shards) == 2
    got = pl.concat(shards).select(pl.len()).collect(engine="streaming")
    assert got.item() == expected.height


def test_dataset_object_pins_version(tmp_path, expected: pl.DataFrame) -> None:
    uri = str(tmp_path / "versioned.lance")
    lance.write_dataset(expected.to_arrow(), uri)
    pinned = lance.dataset(uri)

    lance.write_dataset(expected.to_arrow(), uri, mode="append")

    at_v1 = scan_lance(pinned).select(pl.len()).collect(engine="streaming").item()
    latest = scan_lance(uri).select(pl.len()).collect(engine="streaming").item()
    assert at_v1 == expected.height
    assert latest == 2 * expected.height


def test_join_and_group_by(lance_uri: str, expected: pl.DataFrame) -> None:
    weights = pl.LazyFrame({"cat": ["a", "b"], "w": [1.0, 2.0]})
    got = (
        scan_lance(lance_uri)
        .join(weights, on="cat", how="inner")
        .group_by("cat")
        .agg(pl.len().alias("n"))
        .sort("cat")
        .collect(engine="streaming")
    )
    want = (
        expected.lazy()
        .join(weights, on="cat", how="inner")
        .group_by("cat")
        .agg(pl.len().alias("n"))
        .sort("cat")
        .collect()
    )
    assert_frame_equal(got, want)


def test_sink_parquet_from_lance(
    tmp_path, lance_uri: str, expected: pl.DataFrame
) -> None:
    out = tmp_path / "out.parquet"
    scan_lance(lance_uri).select("id", "cat").sink_parquet(out, engine="streaming")
    got = pl.read_parquet(out)
    assert_frame_equal(got.sort("id"), expected.select("id", "cat").sort("id"))
