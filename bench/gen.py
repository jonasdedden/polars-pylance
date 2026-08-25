"""Generate the Lance size ladder.

Writes to ``$BENCH_ROOT/data`` (default ``/mnt/nvme``). Point BENCH_ROOT at a
local NVMe filesystem -- network storage measures the network, not the format.

Payload bytes are random on purpose: a repeating pattern compresses to nothing
and would make every memory and throughput number meaningless.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
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
ROOT = Path(os.environ.get("BENCH_ROOT", "/mnt/nvme")) / "data"


def batches(rows: int, chunk: int = 100_000, seed: int = 0) -> Iterator[pa.RecordBatch]:
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


def on_disk_gib(uri: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(uri) for f in fs
    )
    return total / 1024**3


ROOT.mkdir(parents=True, exist_ok=True)
for rows in [int(x) for x in sys.argv[1:]]:
    uri = str(ROOT / f"{rows // 1_000_000}m.lance")
    if os.path.exists(uri):
        try:
            if lance.dataset(uri).count_rows() == rows:
                have = on_disk_gib(uri)
                m = rows // 1_000_000
                print(f"{m:>4}M rows: already present ({have:.1f} GiB)", flush=True)
                continue
        except Exception:
            pass
        shutil.rmtree(uri)
    t0 = time.perf_counter()
    reader = pa.RecordBatchReader.from_batches(SCHEMA, batches(rows))
    lance.write_dataset(
        reader,
        uri,
        schema=SCHEMA,
        max_rows_per_file=5_000_000,
        max_rows_per_group=10_000,
    )
    dt = time.perf_counter() - t0
    gib = on_disk_gib(uri)
    m = rows // 1_000_000
    print(
        f"{m:>4}M rows -> {gib:7.1f} GiB in {dt:6.1f}s ({gib / dt:.2f} GiB/s)",
        flush=True,
    )
