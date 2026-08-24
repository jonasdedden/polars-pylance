"""Plan serialization: the prerequisite for shipping a scan to Polars Cloud."""

from __future__ import annotations

import io
import pickle

import polars as pl
from polars.testing import assert_frame_equal

from polars_pylance import LanceScanOptions, LanceScanSpec, scan_lance
from polars_pylance.cloud import requirements_txt


def test_spec_is_picklable(lance_uri: str) -> None:
    spec = LanceScanSpec(uri=lance_uri, options=LanceScanOptions(batch_size=100))
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec
    assert restored.open().count_rows() > 0


def test_provider_plan_round_trips(lance_uri: str) -> None:
    query = scan_lance(lance_uri).filter(pl.col("cat") == "b").select("id", "val")
    blob = query.serialize()

    restored = pl.LazyFrame.deserialize(io.BytesIO(blob))
    assert_frame_equal(
        restored.collect(engine="streaming").sort("id"),
        query.collect(engine="streaming").sort("id"),
    )


def test_provider_plan_is_small(lance_uri: str) -> None:
    """A plan carries a URI and options, never data or an open dataset handle."""
    blob = scan_lance(lance_uri).select("id").serialize()
    assert len(blob) < 8 * 1024, (
        f"serialized plan unexpectedly large: {len(blob)} bytes"
    )


def test_requirements_txt_pins_versions() -> None:
    text = requirements_txt(extra=["numpy"])
    assert f"polars=={pl.__version__}" in text
    assert "pylance==" in text
    assert "numpy" in text
