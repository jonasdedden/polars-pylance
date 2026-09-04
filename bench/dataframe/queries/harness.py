"""Test and measurement scaffolding around the queries.

Worker warmup, scheduler-traffic byte counts, the tiny datasets the smoke
tests generate, and the smoke test itself -- everything the backends need
besides the queries, kept here so the query modules stay readable.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .cases import CaseName

if TYPE_CHECKING:
    import polars as pl
    from lance.fragment import FragmentMetadata

    from bench.dataframe.backends import Backend


def import_dependencies() -> str:
    """Import what a worker needs, and report what it got.

    Sent to every worker before the measured region starts. A worker process
    that has not yet imported Polars and Lance would do it inside the query
    and charge several seconds of import to it, which is startup wearing the
    query's clothes.
    """
    import lance
    import polars as pl

    return f"polars {pl.__version__}, lance {lance.__version__}"


def plan_bytes(shards: list[pl.LazyFrame]) -> int:
    """Count what the scheduler ships per run: the pickled lazy queries."""
    return sum(len(pickle.dumps(s)) for s in shards)


def metadata_bytes(fragments: list[FragmentMetadata]) -> int:
    """Count what the workers hand back: the pickled fragment metadata."""
    return sum(len(pickle.dumps(f)) for f in fragments)


def make_smoke_source(
    directory: str, *, rows: int = 20_000, payload: int = 64, seed: int = 0
) -> str:
    """Write a tiny bench-shaped dataset for smoke tests. Return its URI."""
    import lance
    import numpy as np

    fields: list[pa.Field[Any]] = [
        pa.field("id", pa.int64()),
        pa.field("cat", pa.string()),
        pa.field("val", pa.float64()),
        pa.field("payload", pa.binary(payload)),
    ]
    schema = pa.schema(fields)
    rng = np.random.default_rng(seed)
    batch = pa.record_batch(
        [
            pa.array(np.arange(rows)),
            pa.array(np.array(["a", "b", "c", "d"])[np.arange(rows) % 4]),
            pa.array(rng.random(rows)),
            pa.array(
                [rng.bytes(payload) for _ in range(rows)], type=pa.binary(payload)
            ),
        ],
        schema=schema,
    )
    src = str(Path(directory) / "src.lance")
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, [batch]), src, schema=schema
    )
    return src


def smoke(name: str, run: Backend) -> None:
    """Run one backend end to end on a tiny generated dataset.

    Every backend module calls this under `if __name__ == "__main__"`, so
    `python -m bench.dataframe.backends.<backend>` checks that backend without a
    ladder, a driver, or a cluster to point at. Both paths are exercised:
    the `w_filter` write (rows land in a dataset) and the `r_agg` reduction
    (only scalars come back), since they share nothing but the scheduler.
    """
    import lance

    tmp = Path(tempfile.mkdtemp(prefix=f"dist-{name}-"))
    try:
        src = make_smoke_source(str(tmp))
        write_dst = str(tmp / "out.lance")
        setup = time.perf_counter()
        # The case literals resolve through `Backend`'s overloads to a single
        # record type each, so no union to narrow and no cast.
        written = run(
            src, write_dst, case=CaseName.W_FILTER, n_shards=4, chunk_size=5_000
        )
        reduced = run(src, None, case=CaseName.R_AGG, n_shards=4, chunk_size=5_000)
        overall = time.perf_counter() - setup
        want = lance.dataset(src).count_rows(filter="val > 0.9")
        got = written["result_rows"]
        assert got == want, f"{name}: wrote {got} rows, expected {want}"
        assert reduced["result_count"] == lance.dataset(src).count_rows(
            # Independent oracle: states the expected answer rather than
            # asking the code under test.
            filter="val > 0.5"
        ), f"{name}: reduction counted {reduced['result_count']}"
        # The queries, then what the calls cost in total: the difference is
        # the cluster startup the measurement excludes.
        queries = written["seconds"] + reduced["seconds"]
        print(
            f"{name} smoke: {got} rows written, "
            f"{reduced['result_count']} counted, queries {queries:.2f}s, "
            f"{overall - queries:.2f}s of setup around them"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
