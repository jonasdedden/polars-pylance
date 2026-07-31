"""Plan serialization: the prerequisite for shipping a scan to Polars Cloud."""

from __future__ import annotations

import io
import pickle

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_lance import LanceScanOptions, LanceScanSpec, scan_lance
from polars_lance.cloud import requirements_txt


def test_spec_is_picklable(lance_uri: str) -> None:
    spec = LanceScanSpec(uri=lance_uri, options=LanceScanOptions(batch_size=100))
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
    assert restored.open().count_rows() > 0


def test_provider_plan_round_trips(lance_uri: str) -> None:
    query = (
        scan_lance(lance_uri, impl="provider")
        .filter(pl.col("cat") == "b")
        .select("id", "val")
    )
    blob = query.serialize()

    restored = pl.LazyFrame.deserialize(io.BytesIO(blob))
    assert_frame_equal(
        restored.collect(engine="streaming").sort("id"),
        query.collect(engine="streaming").sort("id"),
    )


def test_provider_plan_is_small(lance_uri: str) -> None:
    """A plan carries a URI and options, never data or an open dataset handle."""
    blob = scan_lance(lance_uri, impl="provider").select("id").serialize()
    assert len(blob) < 8 * 1024, (
        f"serialized plan unexpectedly large: {len(blob)} bytes"
    )


def test_io_plugin_plan_round_trips(lance_uri: str) -> None:
    query = scan_lance(lance_uri, impl="io_plugin").select("id")
    try:
        blob = query.serialize()
    except Exception as exc:
        # The io_plugin path serializes a closure, which polars delegates to
        # cloudpickle; without it installed there is nothing we can do.
        if "cloudpickle" in str(exc):
            pytest.skip("cloudpickle not installed; io_plugin plans cannot serialize")
        raise
    restored = pl.LazyFrame.deserialize(io.BytesIO(blob))
    assert (
        restored.collect(engine="streaming").height
        == query.collect(engine="streaming").height
    )


def test_requirements_txt_pins_versions() -> None:
    text = requirements_txt(extra=["numpy"])
    assert f"polars=={pl.__version__}" in text
    assert "pylance==" in text
    assert "numpy" in text
