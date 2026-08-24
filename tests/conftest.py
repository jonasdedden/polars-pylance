from __future__ import annotations

from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

ROWS = 60_000
PAYLOAD = 64
CATS = np.array(["a", "b", "c", "d"])

SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("cat", pa.string()),
        pa.field("val", pa.float64()),
        pa.field("payload", pa.binary(PAYLOAD)),
    ]
)


def _batches(rows: int, chunk: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for start in range(0, rows, chunk):
        n = min(chunk, rows - start)
        ids = np.arange(start, start + n, dtype=np.int64)
        yield pa.record_batch(
            [
                pa.array(ids),
                pa.array(CATS[ids % 4]),
                pa.array(rng.random(n)),
                pa.FixedSizeBinaryArray.from_buffers(
                    pa.binary(PAYLOAD), n, [None, pa.py_buffer(rng.bytes(n * PAYLOAD))]
                ),
            ],
            schema=SCHEMA,
        )


@pytest.fixture(scope="session")
def lance_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A multi-fragment Lance dataset, written without materialising it."""
    uri = str(tmp_path_factory.mktemp("data") / "src.lance")
    reader = pa.RecordBatchReader.from_batches(SCHEMA, _batches(ROWS, 5_000))
    lance.write_dataset(
        reader,
        uri,
        schema=SCHEMA,
        max_rows_per_file=ROWS // 4,
        max_rows_per_group=2_500,
    )
    return uri


@pytest.fixture(scope="session")
def expected(lance_uri: str) -> pl.DataFrame:
    """Ground truth, computed the eager way we are trying to avoid."""
    return pl.from_arrow(lance.dataset(lance_uri).to_table())  # type: ignore[return-value]


@pytest.fixture
def scanner_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the arguments Lance itself receives, options and all."""
    calls: list[dict[str, Any]] = []
    original = lance.LanceDataset.scanner

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "scanner", spy)
    return calls


@pytest.fixture
def frames_yielded(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the batches actually pulled out of Lance, for early-stop tests."""
    from polars_lance import _scan

    counter = [0]
    original = _scan.LanceScanSpec.iter_frames

    def spy(self: Any, dataset: Any, **kwargs: Any) -> Any:
        for frame in original(self, dataset, **kwargs):
            counter[0] += 1
            yield frame

    monkeypatch.setattr(_scan.LanceScanSpec, "iter_frames", spy)
    return counter
