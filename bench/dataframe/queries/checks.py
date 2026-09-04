"""Checksums: how backends prove they computed the same answer.

`agree` compares two checksum fields, tolerating the float wobble that
batching order introduces; `checksum` reduces a whole dataset to one row
of count plus id and value sums, so verification stays affordable at the
top of the ladder.
"""

from __future__ import annotations

import polars as pl

from .cases import FLOAT_TOL


def agree(left: float, right: float) -> bool:
    """Whether two checksums say the same thing.

    Exact for integers -- counts and id sums are order-independent -- with
    a relative tolerance for floats, whose sums depend on batching order.
    """
    if isinstance(left, float) or isinstance(right, float):
        return abs(left - right) <= FLOAT_TOL * max(abs(left), abs(right), 1.0)
    return bool(left == right)


def checksum(frame: pl.DataFrame) -> dict[str, float | int]:
    """Row count plus id and value checksums, streamed to one row."""
    row = frame.select(
        pl.len().alias("n"),
        pl.col("id").sum().alias("s_id"),
        pl.col("val").sum().alias("s_val"),
    ).to_dicts()[0]
    return {"n": row["n"] or 0, "s_id": row["s_id"] or 0, "s_val": row["s_val"] or 0.0}
