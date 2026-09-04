"""Build scalar indices on the ladder, so the indexed pass has something to use.

An index is only reachable by a reader that pushes the predicate down. That is
the point of the indexed pass: the same query, the same data, and one of the two
implementations can act on the index while the other reads the column.

Run after the unindexed passes, since this changes the datasets in place.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lance

ROOT = Path(os.environ.get("BENCH_ROOT", "/mnt/nvme")) / "data"

# One index per predicate shape the indexed pass measures. `ts` is deliberately
# absent: `dt.hour()` cannot use an index on the timestamp, and that case is
# there to show it.
INDICES = [
    ("id", "BTREE"),  # id.is_in(...)
    ("val", "BTREE"),  # val > 0.999
    ("cat", "BITMAP"),  # cat == 'a', four distinct values
    ("text", "NGRAM"),  # text.str.contains(...)
]


def indices_gib(uri: str) -> float:
    path = Path(uri) / "_indices"
    if not path.is_dir():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1024**3


for rows in [int(x) for x in sys.argv[1:]]:
    m = rows // 1_000_000
    uri = str(ROOT / f"{m}m.lance")
    dataset = lance.dataset(uri)
    have = {f for i in dataset.list_indices() for f in i["fields"]}
    for column, kind in INDICES:
        if column in have:
            print(f"{m:>4}M {column:<5} {kind:<7} already present", flush=True)
            continue
        t0 = time.perf_counter()
        dataset.create_scalar_index(column, index_type=kind)
        print(
            f"{m:>4}M {column:<5} {kind:<7} {time.perf_counter() - t0:7.1f}s",
            flush=True,
        )
    print(f"{m:>4}M indices on disk: {indices_gib(uri):.2f} GiB", flush=True)
