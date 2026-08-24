"""Lazy, streaming Lance <-> Polars integration.

Reading gives a real :class:`~polars.LazyFrame`: the optimizer pushes column
projections, filters and row limits into Lance, and batches are pulled only as
the streaming engine consumes them. Writing streams a query into Lance batch by
batch, so neither direction holds the dataset -- or a whole fragment -- in RAM.

>>> import polars as pl
>>> import polars_lance as pll
>>> lf = pll.scan_lance("data.lance")  # doctest: +SKIP
>>> pll.sink_lance(
...     lf.filter(pl.col("score") > 0.9), "filtered.lance"
... )  # doctest: +SKIP
"""

from __future__ import annotations

from ._options import LanceScanOptions
from ._predicate import to_lance_filter
from ._scan import (
    LanceDatasetProvider,
    LanceScanSpec,
    scan_lance,
    scan_lance_fragments,
)
from ._sink import (
    commit_lance_fragments,
    sink_lance,
    write_lance_fragments,
)

__version__ = "0.1.0"

__all__ = [
    "LanceDatasetProvider",
    "LanceScanOptions",
    "LanceScanSpec",
    "commit_lance_fragments",
    "scan_lance",
    "scan_lance_fragments",
    "sink_lance",
    "to_lance_filter",
    "write_lance_fragments",
]
