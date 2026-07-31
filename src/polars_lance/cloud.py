"""Helpers for running Lance scans on Polars Cloud.

What works and what does not, as of polars-cloud 0.9:

Reading
    A ``scan_lance`` plan serializes to ~1-2 kB and can be shipped with
    ``LazyFrame.remote()``, but the remote workers must be able to ``import
    lance`` and to reach the dataset's storage. Install the dependency with
    ``ComputeContext(requirements=...)``; see :func:`requirements_txt`.

Writing
    Not possible remotely. Polars Cloud's only sink destinations are Parquet,
    CSV, IPC and Iceberg. Sink the remote query to Parquet on object storage and
    convert it in a second step -- see :func:`convert_parquet_to_lance` -- or
    ``.collect()`` and stream the result into Lance client-side with
    :func:`~polars_lance.sink_lance`.

Untested
    Whether the *distributed* planner accepts a Python scan node at all. If it
    refuses, shard the read yourself with
    :func:`~polars_lance.scan_lance_fragments` and concatenate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lance
import polars as pl

if TYPE_CHECKING:
    from polars_lance._sink import WriteMode


def requirements_txt(extra: list[str] | None = None) -> str:
    """Render a requirements file pinning the versions a cloud worker needs.

    Polars Cloud rejects a compute context whose polars version differs from the
    client's, so both pins are exact.

    Examples
    --------
    >>> import polars_cloud as pc  # doctest: +SKIP
    >>> ctx = pc.ComputeContext(
    ...     cpus=8, memory=32, requirements=requirements_txt().encode()
    ... )  # doctest: +SKIP
    """
    lines = [
        f"polars=={pl.__version__}",
        f"pylance=={lance.__version__}",
        "polars-lance",
    ]
    lines.extend(extra or [])
    return "\n".join(lines) + "\n"


def convert_parquet_to_lance(
    parquet_source: str | Path | list[str],
    target: str | Path,
    *,
    mode: WriteMode = "create",
    chunk_size: int = 25_000,
    storage_options: dict[str, str] | None = None,
    **lance_write_kwargs: Any,
) -> lance.LanceDataset:
    """Stream Parquet output from a remote query into a Lance dataset.

    The documented way to land Polars Cloud results in Lance: the remote query
    sinks Parquet to object storage, then this converts it without materialising
    the data.

    Examples
    --------
    >>> query.remote(ctx).distributed().sink_parquet(staging)  # doctest: +SKIP
    >>> convert_parquet_to_lance(staging, "s3://bucket/out.lance")  # doctest: +SKIP
    """
    from polars_lance._sink import sink_lance

    lf = pl.scan_parquet(parquet_source, storage_options=storage_options)
    dataset = sink_lance(
        lf,
        target,
        mode=mode,
        chunk_size=chunk_size,
        **lance_write_kwargs,
    )
    assert isinstance(dataset, lance.LanceDataset)
    return dataset
