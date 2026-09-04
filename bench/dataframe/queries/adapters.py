"""Native frame in, native frame out, per engine.

The queries in `cases` are engine-agnostic; these are the thin wrappers
that run one on a concrete frame and hand the native result back, plus the
Daft engine setup the backends need before they can do that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from .cases import CASES

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa
    from daft import DataFrame as DaftFrame

    from bench.dataframe.queries import CaseName


def apply_polars(shard: pl.LazyFrame, case: CaseName) -> pl.LazyFrame:
    """Run one case's query on a Polars shard, returning a native plan.

    The Narwhals wrapper lowers to the same Polars IR nodes a hand-written
    query would -- same plan, same predicate pushdown into Lance -- so what
    ships to workers is an ordinary pickled LazyFrame.
    """
    # `to_native` widens to `Any` once the query is typed as `Frame -> Frame`;
    # the annotation re-attaches the engine type without a cast.
    out: pl.LazyFrame = nw.to_native(CASES[case].query(nw.from_native(shard)))
    return out


def apply_daft(frame: DaftFrame, case: CaseName) -> DaftFrame:
    """Run one case's query on a Daft frame, returning a native Daft plan."""
    out: DaftFrame = nw.to_native(CASES[case].query(nw.from_native(frame)))
    return out


def apply_arrow(batch: pa.Table, case: CaseName) -> pa.Table:
    """Run one case's query on one Arrow batch, returning an Arrow table.

    The Ray Data `map_batches` spelling: one top-level function taking the
    case by name, so `functools.partial` over it stays picklable.
    """
    out: pa.Table = nw.to_native(
        CASES[case].query(nw.from_native(batch, eager_only=True))
    )
    return out


def daft_query_frame(src: str, n_shards: int, case: CaseName) -> DaftFrame:
    """Read `src` into Daft and apply one case's query, still lazy."""
    return apply_daft(daft_read(src, n_shards), case)


def daft_totals(frame: DaftFrame) -> dict[str, int]:
    """Collect a one-row Daft reduction to a plain dict."""
    row = frame.collect().to_pydict()
    # Daft yields None for a sum over zero rows; the reduction's zero is 0.
    return {"n": row["n"][0] or 0, "s_id": row["s_id"][0] or 0}


_daft_configured: tuple[str, int, int] | None = None


def configure_daft(mode: str, budget_cpus: int, chunk_size: int) -> None:
    """Set Daft's runner once per process; Daft refuses to reconfigure it.

    One process runs one budget (the smoke test runs two cases, the matrix
    one case per process), so a repeat call with the same key is a no-op
    and anything else is a programming error, not something to retry.
    """
    import daft

    global _daft_configured
    key = (mode, budget_cpus, chunk_size)
    if _daft_configured == key:
        return
    if _daft_configured is not None:
        msg = f"Daft runner already set to {_daft_configured}; cannot switch to {key}"
        raise SystemExit(msg)
    if mode == "native":
        daft.set_runner_native(num_threads=budget_cpus)
    else:
        daft.set_runner_ray(noop_if_initialized=True)
    # `chunk_size` means the same thing here as the batch size handed
    # to a writer elsewhere: the rows an operator works on at once.
    daft.set_execution_config(default_morsel_size=chunk_size)
    _daft_configured = key


def daft_read(src: str, n_shards: int) -> DaftFrame:
    """Read a Lance dataset into Daft, cut into about `n_shards` scan tasks.

    Daft groups fragments into scan tasks, and by default groups more of
    them than there are cores here, which leaves threads idle. Grouping to
    the shard count instead gives Daft the same granularity the other
    backends get from `scan_lance_fragments`.
    """
    import daft
    import lance

    fragments = len(lance.dataset(src).get_fragments())
    frame: DaftFrame = daft.read_lance(
        src, fragment_group_size=max(1, fragments // max(1, n_shards))
    )
    return frame
