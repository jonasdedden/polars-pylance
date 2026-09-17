from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ROWS = 60_000
PAYLOAD = 64
CATS = np.array(["a", "b", "c", "d"])

_FIELDS: list[pa.Field[pa.DataType]] = [
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
    frame = pl.from_arrow(lance.dataset(lance_uri).to_table())
    assert isinstance(frame, pl.DataFrame)
    return frame


@dataclass(frozen=True)
class ScannerCall:
    """One `LanceDataset.scanner` call, keeping only the arguments the tests read."""

    columns: list[str] | None
    filter: str | None
    limit: int | None
    prefilter: bool | None
    batch_size: int | None
    io_buffer_size: int | None
    nearest: dict[str, object] | None


def spy_on_scanner(
    monkeypatch: pytest.MonkeyPatch, record: Callable[[ScannerCall], None]
) -> None:
    """Send every `LanceDataset.scanner` call to `record` for the rest of a test.

    Exposed alongside the fixtures below because a test that builds an index
    first has to start recording after it, rather than at fixture time.
    """
    original = lance.LanceDataset.scanner

    def scanner(
        self: lance.LanceDataset,
        *args: Any,  # noqa: ANN401 - forwarded as given
        **kwargs: Any,  # noqa: ANN401 - forwarded as given
    ) -> lance.LanceScanner:
        record(ScannerCall(**{f.name: kwargs.get(f.name) for f in fields(ScannerCall)}))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "scanner", scanner)


@pytest.fixture
def scanner_calls(monkeypatch: pytest.MonkeyPatch) -> list[ScannerCall]:
    """Record the arguments Lance itself receives, options and all."""
    calls: list[ScannerCall] = []
    spy_on_scanner(monkeypatch, calls.append)
    return calls


@pytest.fixture
def frames_yielded(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the batches actually pulled out of Lance, for early-stop tests."""
    from polars_pylance import _scan

    counter = [0]
    original = _scan.LanceScanSpec.iter_frames

    def spy(
        self: _scan.LanceScanSpec,
        *args: Any,  # noqa: ANN401 - forwarded as given
        **kwargs: Any,  # noqa: ANN401 - forwarded as given
    ) -> Iterator[pl.DataFrame]:
        for frame in original(self, *args, **kwargs):
            counter[0] += 1
            yield frame

    monkeypatch.setattr(_scan.LanceScanSpec, "iter_frames", spy)
    return counter


# A second dataset, wider in types rather than rows: strings, floats that are
# not numbers, timestamps, dates, lists and structs.
RICH_ROWS = 2_000
WORDS = np.array(["alpha", "beta", "gamma", "delta"])

_RICH_FIELDS: list[pa.Field[pa.DataType]] = [
    pa.field("id", pa.int64()),
    pa.field("cat", pa.string()),
    pa.field("text", pa.string()),
    pa.field("val", pa.float64()),
    # NaN and both infinities, where Polars' ordering and SQL's disagree.
    pa.field("odd", pa.float64()),
    pa.field("flag", pa.bool_()),
    pa.field("opt", pa.int64()),
    pa.field("ts", pa.timestamp("us")),
    pa.field("day", pa.date32()),
    pa.field("tags", pa.list_(pa.int64())),
    pa.field("meta", pa.struct([("k", pa.int64()), ("s", pa.string())])),
    pa.field("odd name", pa.int64()),
]
RICH_SCHEMA = pa.schema(_RICH_FIELDS)


def _odd(i: int) -> float:
    if i % 22 == 0:
        # What `0 / 0` produces on x86: a NaN with its sign bit set, which Lance
        # orders below every number.
        return math.copysign(math.nan, -1.0)
    if i % 11 == 0:
        return float("nan")
    if i % 13 == 0:
        return float("inf") if i % 26 else float("-inf")
    return i / 100.0


@pytest.fixture(scope="session")
def rich_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A dataset covering every dtype the predicate lowering claims to handle."""
    rng = np.random.default_rng(1)
    n = RICH_ROWS
    base = dt.datetime(2024, 1, 1)
    table = pa.table(
        [
            pa.array(np.arange(n, dtype=np.int64)),
            pa.array(WORDS[np.arange(n) % 4]),
            pa.array([f"  row-{i:05d}-{WORDS[i % 4]}  " for i in range(n)]),
            pa.array(rng.random(n)),
            pa.array([_odd(i) for i in range(n)]),
            pa.array(np.arange(n) % 3 == 0),
            pa.array([None if i % 7 == 0 else i for i in range(n)], pa.int64()),
            pa.array([base + dt.timedelta(hours=i) for i in range(n)]),
            pa.array(
                [dt.date(2024, 1, 1) + dt.timedelta(days=i % 365) for i in range(n)]
            ),
            # Some rows are empty, so `list.get` hits its out-of-bounds path.
            pa.array(
                [[i % 5, i % 3] if i % 9 else [] for i in range(n)],
                pa.list_(pa.int64()),
            ),
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
    frame = pl.from_arrow(lance.dataset(rich_uri).to_table())
    assert isinstance(frame, pl.DataFrame)
    return frame


@pytest.fixture
def pushed_filters(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every SQL filter string that actually reached Lance's scanner."""
    filters: list[str] = []

    def record(call: ScannerCall) -> None:
        if call.filter is not None:
            filters.append(call.filter)

    spy_on_scanner(monkeypatch, record)
    return filters
