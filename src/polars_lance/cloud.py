"""Helpers for running Lance scans and writes on Polars Cloud.

What works and what does not, as of polars-cloud 0.10:

Reading
    A ``scan_lance`` plan serializes to ~2-6 kB and can be shipped with
    ``LazyFrame.remote()``, but the remote workers must be able to ``import
    lance`` and to reach the dataset's storage. Install the dependency with
    ``ComputeContext(requirements=...)``; see :func:`requirements_txt`.

    Both scan implementations survive ``prepare_cloud_plan`` -- ``provider`` and
    ``io_plugin``, on their own and under ``pl.concat`` of
    :func:`~polars_lance.scan_lance_fragments` shards. 0.9 added distributed
    unions of Python scans, so the sharded form is the sanctioned way to fan a
    read out across workers rather than a fallback.

Writing
    Possible remotely since 0.10, via :func:`sink_lance_remote`. Polars Cloud's
    native sink destinations are still Parquet, CSV, IPC and Iceberg, but
    ``sink_batches`` hands each result batch to a Python callable that is
    cloudpickled into the query plan and therefore runs *on the workers* -- so
    the workers write Lance data files directly, and a single client-side commit
    publishes them. :mod:`polars_lance._remote` documents the arrangement.

    The Parquet-staging route remains as the conservative fallback: sink the
    remote query to Parquet on object storage and convert it with
    :func:`convert_parquet_to_lance`. polars-cloud 0.10's
    ``DirectQuery.delete_result()`` makes cleaning up the intermediate a single
    call, in direct mode with anonymous storage configured for ``allow_delete``.

The polars pin
    polars-cloud 0.10 requires ``polars==1.43.2``, up from 1.42.1 in 0.9. The
    ``collect_batches`` Arrow C stream deadlock that :mod:`polars_lance._sink`
    works around is still present in 1.43.2 for the default ``provider`` scan,
    so the workaround stays.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lance
import polars as pl

from polars_lance._remote import (
    StagedLanceSink,
    sink_lance_remote,
    stage_lance_sink,
)

if TYPE_CHECKING:
    from polars_lance._sink import WriteMode

__all__ = [
    "StagedLanceSink",
    "convert_parquet_to_lance",
    "requirements_txt",
    "sink_lance_remote",
    "stage_lance_sink",
]


def requirements_txt(extra: list[str] | None = None) -> str:
    """Render a requirements file pinning the versions a cloud worker needs.

    Polars Cloud rejects a compute context whose polars version differs from the
    client's, so both pins are exact. ``polars-lance`` itself is on the list
    because a :func:`sink_lance_remote` callback is pickled by reference: the
    worker imports it rather than receiving its code.

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
