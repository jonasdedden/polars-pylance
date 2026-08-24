"""Generate Lance datasets for the benchmarks.

The payload column is random bytes on purpose: a repeating pattern compresses to
almost nothing, which makes a 1 M-row dataset occupy 15 MB on disk and renders
every memory measurement meaningless.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from typing import Any

import lance
import numpy as np
import pyarrow as pa

PAYLOAD = 512
CATS = np.array(["a", "b", "c", "d"])

_FIELDS: list[pa.Field[Any]] = [
    pa.field("id", pa.int64()),
    pa.field("cat", pa.string()),
    pa.field("val", pa.float64()),
    pa.field("payload", pa.binary(PAYLOAD)),
]
SCHEMA = pa.schema(_FIELDS)


def batches(rows: int, chunk: int = 50_000, seed: int = 0) -> Iterator[pa.RecordBatch]:
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


def ensure_dataset(
    uri: str, rows: int, *, overwrite: bool = False
) -> lance.LanceDataset:
    """Create `uri` with `rows` rows if it is not already there."""
    if os.path.exists(uri):
        if not overwrite:
            existing = lance.dataset(uri)
            if existing.count_rows() == rows:
                return existing
        shutil.rmtree(uri)

    reader = pa.RecordBatchReader.from_batches(SCHEMA, batches(rows))
    return lance.write_dataset(
        reader, uri, schema=SCHEMA, max_rows_per_file=200_000, max_rows_per_group=10_000
    )


def on_disk_mb(uri: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(dirpath, name))
        for dirpath, _, names in os.walk(uri)
        for name in names
    )
    return total / 1e6
