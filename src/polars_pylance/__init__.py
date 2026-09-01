"""Lazy, streaming Lance <-> Polars integration.

Reading gives a real `LazyFrame`: the optimizer pushes column
projections, filters and row limits into Lance, and batches are pulled only as
the streaming engine consumes them. Writing streams a query into Lance batch by
batch, so neither direction holds the dataset (or a whole fragment) in RAM.

>>> import polars as pl
>>> import polars_pylance as pll
>>> lf = pll.scan_lance("data.lance")  # doctest: +SKIP
>>> pll.sink_lance(
...     lf.filter(pl.col("score") > 0.9), "filtered.lance"
... )  # doctest: +SKIP
"""

from __future__ import annotations

from ._options import LanceScanOptions
from ._predicate import LanceFilter, to_lance_filter
from ._scan import (
    LanceScanSpec,
    scan_lance,
    scan_lance_fragments,
)
from ._sink import (
    commit_lance_fragments,
    sink_lance,
    write_lance_fragments,
)

# `_version.py` is written at build time from the git tag; it is not in the
# repository. A checkout that has never been built has no such file.
try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source tree that was never built
    __version__ = "0.0.0+unknown"

__all__ = [
    "LanceFilter",
    "LanceScanOptions",
    "LanceScanSpec",
    "commit_lance_fragments",
    "scan_lance",
    "scan_lance_fragments",
    "sink_lance",
    "to_lance_filter",
    "write_lance_fragments",
]
