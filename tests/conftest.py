from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

ROWS = 60_000
PAYLOAD = 64
CATS = np.array(["a", "b", "c", "d"])

_FIELDS: list[pa.Field[Any]] = [
    pa.field("id", pa.int64()),
    pa.field("cat", pa.string()),
    pa.field("val", pa.float64()),
    pa.field("payload", pa.binary(PAYLOAD)),
]
SCHEMA = pa.schema(_FIELDS)


def _batches(rows: int, chunk: int, seed: int = 0) -> Iterator[pa.RecordBatch]:
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
    from polars_pylance import _scan

    counter = [0]
    original = _scan.LanceScanSpec.iter_frames

    def spy(self: Any, dataset: Any, **kwargs: Any) -> Any:
        for frame in original(self, dataset, **kwargs):
            counter[0] += 1
            yield frame

    monkeypatch.setattr(_scan.LanceScanSpec, "iter_frames", spy)
    return counter


# A second dataset, wider in *types* rather than rows: the predicate lowering
# has to deal with strings, timestamps, lists and structs, none of which the
# scan fixtures above exercise.
RICH_ROWS = 2_000
WORDS = np.array(["alpha", "beta", "gamma", "delta"])

_RICH_FIELDS: list[pa.Field[Any]] = [
    pa.field("id", pa.int64()),
    pa.field("cat", pa.string()),
    pa.field("text", pa.string()),
    pa.field("val", pa.float64()),
    pa.field("flag", pa.bool_()),
    pa.field("opt", pa.int64()),
    pa.field("ts", pa.timestamp("us")),
    pa.field("day", pa.date32()),
    pa.field("tags", pa.list_(pa.int64())),
    pa.field("meta", pa.struct([("k", pa.int64()), ("s", pa.string())])),
    pa.field("odd name", pa.int64()),
]
RICH_SCHEMA = pa.schema(_RICH_FIELDS)


@pytest.fixture(scope="session")
def rich_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A dataset covering every dtype the predicate lowering claims to handle."""
    import datetime as dt

    rng = np.random.default_rng(1)
    n = RICH_ROWS
    base = dt.datetime(2024, 1, 1)
    table = pa.table(
        [
            pa.array(np.arange(n, dtype=np.int64)),
            pa.array(WORDS[np.arange(n) % 4]),
            pa.array([f"row-{i:05d}-{WORDS[i % 4]}" for i in range(n)]),
            pa.array(rng.random(n)),
            pa.array(np.arange(n) % 3 == 0),
            pa.array([None if i % 7 == 0 else i for i in range(n)], pa.int64()),
            pa.array([base + dt.timedelta(hours=i) for i in range(n)]),
            pa.array(
                [dt.date(2024, 1, 1) + dt.timedelta(days=i % 365) for i in range(n)]
            ),
            pa.array([[i % 5, i % 3] for i in range(n)], pa.list_(pa.int64())),
            pa.array(
                [{"k": i % 10, "s": WORDS[i % 4]} for i in range(n)],
                pa.struct([("k", pa.int64()), ("s", pa.string())]),
            ),
            pa.array(np.arange(n, dtype=np.int64)),
        ],
        schema=RICH_SCHEMA,
    )
    uri = str(tmp_path_factory.mktemp("rich") / "rich.lance")
    lance.write_dataset(table, uri, max_rows_per_file=500)
    return uri


@pytest.fixture(scope="session")
def rich_frame(rich_uri: str) -> pl.DataFrame:
    """Ground truth for `rich_uri`, materialised eagerly."""
    return pl.from_arrow(lance.dataset(rich_uri).to_table())  # type: ignore[return-value]
