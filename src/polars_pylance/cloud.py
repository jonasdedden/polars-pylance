"""Helpers for running Lance scans and writes on Polars Cloud.

.. warning::
    **Not installable today.** polars-cloud pins polars with ``==`` -- 0.10.0
    pins ``polars==1.43.2`` -- which is below this package's ``polars>=1.44.1``
    floor, so there is no ``cloud`` extra and the two cannot be resolved
    together. Everything here is written and kept working against the 0.10 API;
    it becomes usable as soon as polars-cloud ships a release tracking 1.44.
    See "The polars pin" below.

What works and what does not, as of polars-cloud 0.10:

Reading
    A ``scan_lance`` plan serializes to ~2-6 kB and can be shipped with
    ``LazyFrame.remote()``, but the remote workers must be able to ``import
    lance`` and to reach the dataset's storage. Install the dependency with
    ``ComputeContext(requirements=...)``; see :func:`requirements_txt`.

    The scan survives ``prepare_cloud_plan``, on its own and under
    ``pl.concat`` of :func:`~polars_pylance.scan_lance_fragments` shards. 0.9
    added distributed unions of Python scans, so the sharded form is the
    sanctioned way to fan a read out across workers rather than a fallback.

Writing
    Possible remotely since 0.10, via :func:`sink_lance_remote`. Polars Cloud's
    native sink destinations are still Parquet, CSV, IPC and Iceberg, but
    ``sink_batches`` hands each result batch to a Python callable that is
    cloudpickled into the query plan and therefore runs *on the workers* -- so
    the workers write Lance data files directly, and a single client-side commit
    publishes them. :mod:`polars_pylance._remote` documents the arrangement.

    The Parquet-staging route remains as the conservative fallback: sink the
    remote query to Parquet on object storage and convert it with
    :func:`convert_parquet_to_lance`. polars-cloud 0.10's
    ``DirectQuery.delete_result()`` makes cleaning up the intermediate a single
    call, in direct mode with anonymous storage configured for ``allow_delete``.

The polars pin
    polars-cloud 0.10 requires ``polars==1.43.2``, up from 1.42.1 in 0.9, and
    this package now requires ``polars>=1.44.1``. Those are mutually exclusive,
    which is why the ``cloud`` extra was dropped rather than left declared: an
    extra pinning below the floor makes even ``uv lock`` unresolvable, not just
    ``pip install polars-pylance[cloud]``.

    The floor is there because 1.43.2 is the last release in which a
    ``sort().head()`` pushes an unevaluable ``dynamic_pred`` node into an IO
    plugin's predicate, which is exactly what ``scan_lance`` is. 1.44.0 fixed
    that but was yanked, so 1.44.1 is the first usable release. Installing
    polars-cloud alongside polars-pylance downgrades polars into that range and
    reintroduces it, so it is not a supported workaround. Wait for the
    polars-cloud release that tracks 1.44.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lance
import polars as pl

from ._remote import (
    StagedLanceSink,
    sink_lance_remote,
    stage_lance_sink,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._sink import WriteMode

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
    client's, so both pins are exact. ``polars-pylance`` itself is on the list
    because a :func:`sink_lance_remote` callback is pickled by reference: the
    worker imports it rather than receiving its code.

    Parameters
    ----------
    extra
        Further requirement lines to append, for whatever else the query needs
        on the worker.

    Returns
    -------
    str
        The file contents, newline-terminated.

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
        "polars-pylance",
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
    **lance_write_kwargs: Any,  # noqa: ANN401 - passed through to Lance as given
) -> lance.LanceDataset:
    """Stream Parquet output from a remote query into a Lance dataset.

    The documented way to land Polars Cloud results in Lance: the remote query
    sinks Parquet to object storage, then this converts it without materialising
    the data.

    Parameters
    ----------
    parquet_source
        The staged Parquet: a path, a URI, a glob, or a list of them.
    target
        Destination Lance URI or path.
    mode
        ``"create"`` (fail if it exists), ``"append"``, ``"overwrite"`` (new
        version), or ``"merge"`` for an upsert, whose join key goes through
        ``lance_write_kwargs`` as ``on``.
    chunk_size
        Rows buffered per batch handed to Lance.
    storage_options
        Object-store credentials and settings for reading the Parquet. The
        Lance write takes its own; pass those in ``lance_write_kwargs``.
    **lance_write_kwargs
        Passed through to :func:`~polars_pylance.sink_lance`, e.g.
        ``max_rows_per_file``.

    Returns
    -------
    lance.LanceDataset
        The written dataset.

    Examples
    --------
    >>> query.remote(ctx).distributed().sink_parquet(staging)  # doctest: +SKIP
    >>> convert_parquet_to_lance(staging, "s3://bucket/out.lance")  # doctest: +SKIP

    """
    from ._sink import sink_lance

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
