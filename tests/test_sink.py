"""sink_lance correctness. Memory behaviour is covered by `bench/`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pylance import (
    scan_lance,
    scan_lance_fragments,
    sink_lance,
    write_lance_fragments,
)

if TYPE_CHECKING:
    from pathlib import Path


def _transformed(uri: str) -> pl.LazyFrame:
    return (
        scan_lance(uri)
        .filter(pl.col("val") > 0.5)
        .select("id", "cat", (pl.col("val") * 2).alias("val2"))
    )


def test_create_round_trip(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "out.lance")
    dataset = sink_lance(_transformed(lance_uri), out, max_rows_per_file=5_000)
    assert isinstance(dataset, lance.LanceDataset)

    want = _transformed(lance_uri).collect(engine="streaming")
    got = scan_lance(out).collect(engine="streaming")
    assert_frame_equal(got.sort("id"), want.sort("id"))
    assert len(dataset.get_fragments()) > 1, "expected several fragments"


def test_accepts_eager_dataframe(tmp_path: Path) -> None:
    frame = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    out = str(tmp_path / "eager.lance")

    sink_lance(frame, out)

    assert_frame_equal(scan_lance(out).collect(engine="streaming"), frame)


def test_create_refuses_existing(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "twice.lance")
    sink_lance(_transformed(lance_uri), out)
    with pytest.raises(OSError, match="already exists"):
        sink_lance(_transformed(lance_uri), out, mode="create")


def test_append(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "appended.lance")
    sink_lance(_transformed(lance_uri), out)
    first = scan_lance(out).select(pl.len()).collect(engine="streaming").item()
    sink_lance(_transformed(lance_uri), out, mode="append")
    second = scan_lance(out).select(pl.len()).collect(engine="streaming").item()
    assert second == 2 * first


def test_overwrite(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "overwritten.lance")
    sink_lance(_transformed(lance_uri), out)
    sink_lance(
        _transformed(lance_uri).filter(pl.col("cat") == "a"), out, mode="overwrite"
    )
    got = scan_lance(out).collect(engine="streaming")
    want = (
        _transformed(lance_uri).filter(pl.col("cat") == "a").collect(engine="streaming")
    )
    assert_frame_equal(got.sort("id"), want.sort("id"))


def test_merge_upsert(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "merged.lance")
    sink_lance(_transformed(lance_uri), out)
    before = scan_lance(out).collect(engine="streaming")

    updates = _transformed(lance_uri).head(10).with_columns(pl.col("val2") * 0 - 1.0)
    sink_lance(updates, out, mode="merge", on="id")

    after = scan_lance(out).collect(engine="streaming")
    assert after.height == before.height, "an upsert must not add rows"
    assert after.filter(pl.col("val2") == -1.0).height == 10


def test_merge_requires_on(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "nokey.lance")
    sink_lance(_transformed(lance_uri), out)
    with pytest.raises(ValueError, match="requires `on`"):
        sink_lance(_transformed(lance_uri), out, mode="merge")


def test_on_rejected_for_non_merge(tmp_path: Path, lance_uri: str) -> None:
    with pytest.raises(ValueError, match="only meaningful for mode='merge'"):
        sink_lance(_transformed(lance_uri), str(tmp_path / "x.lance"), on="id")


def test_lazy_sink_defers_until_collect(tmp_path: Path, lance_uri: str) -> None:
    out = tmp_path / "deferred.lance"
    plan = sink_lance(_transformed(lance_uri), str(out), lazy=True)
    assert isinstance(plan, pl.LazyFrame)
    assert not out.exists(), "lazy=True must not write anything yet"

    summary = plan.collect(engine="streaming")
    assert out.exists()
    assert summary.height == 1
    want = _transformed(lance_uri).select(pl.len()).collect(engine="streaming").item()
    assert summary["rows"].item() == want
    assert summary["uri"].item().endswith("deferred.lance")


def test_write_fragments_parallel(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "fragmented.lance")
    shards = [
        shard.filter(pl.col("val") > 0.5).select("id", "cat")
        for shard in scan_lance_fragments(lance_uri)
    ]
    dataset = write_lance_fragments(shards, out, max_rows_per_file=5_000)

    want = (
        scan_lance(lance_uri)
        .filter(pl.col("val") > 0.5)
        .select("id", "cat")
        .collect(engine="streaming")
    )
    got = scan_lance(out).collect(engine="streaming")
    assert_frame_equal(got.sort("id"), want.sort("id"))
    assert dataset.count_rows() == want.height


def test_write_fragments_append(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "frag_append.lance")
    shards = [s.select("id", "cat") for s in scan_lance_fragments(lance_uri)]
    first = write_lance_fragments(shards, out)
    second = write_lance_fragments(shards, out, mode="append")
    assert second.count_rows() == 2 * first.count_rows()


def test_write_fragments_overwrite_existing(tmp_path: Path, lance_uri: str) -> None:
    """Replacing a dataset writes its fragments in 'overwrite' mode.

    `write_fragments(mode='create')` refuses an existing dataset outright.
    """
    out = str(tmp_path / "frag_overwrite.lance")
    shards = [s.select("id", "cat") for s in scan_lance_fragments(lance_uri)]
    write_lance_fragments(shards, out)

    half = [s.filter(pl.col("id") < 10_000) for s in shards]
    dataset = write_lance_fragments(half, out, mode="overwrite")
    assert dataset.count_rows() == sum(
        s.select(pl.len()).collect(engine="streaming").item() for s in half
    )


def test_write_fragments_create_refuses_existing(
    tmp_path: Path, lance_uri: str
) -> None:
    out = str(tmp_path / "frag_twice.lance")
    shards = [s.select("id", "cat") for s in scan_lance_fragments(lance_uri)]
    write_lance_fragments(shards, out)
    with pytest.raises(FileExistsError):
        write_lance_fragments(shards, out)


def test_write_fragments_needs_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one LazyFrame"):
        write_lance_fragments([], str(tmp_path / "empty.lance"))


def test_round_trip_preserves_binary_payload(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "payload.lance")
    sink_lance(scan_lance(lance_uri), out)
    got = scan_lance(out).select("id", "payload").collect(engine="streaming")
    want = scan_lance(lance_uri).select("id", "payload").collect(engine="streaming")
    assert_frame_equal(got.sort("id"), want.sort("id"))
