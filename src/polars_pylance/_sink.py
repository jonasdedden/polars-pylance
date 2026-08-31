"""Streaming Lance writer for Polars LazyFrames.

Lance is not a native Polars sink, so the data has to cross the Python boundary.
:meth:`polars.LazyFrame.collect_batches` streams the query out batch by batch and
exposes those batches as an Arrow C stream, which PyArrow adopts as a
``RecordBatchReader`` and Lance's writer consumes natively. The streaming engine
on one side and the writer on the other then pull through at their own pace, and
neither holds the whole result.

Measured cost of the boundary on a 527 MB source: a fixed ~340 MB of extra
resident memory versus an engine-internal sink such as ``sink_parquet``. It does
*not* grow with the size of the input, and does not grow when the writer is
slower than the producer. ``collect_batches`` is marked unstable by Polars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import lance
import polars as pl
import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from lance.fragment import FragmentMetadata

WriteMode = Literal["create", "append", "overwrite", "merge"]
# Mirrors polars' own engine literal, so the value passes through to
# `collect_batches` without a cast.
EngineType = Literal["auto", "in-memory", "streaming", "gpu"]

DEFAULT_CHUNK_SIZE = 25_000


def _target_uri(target: str | Path | lance.LanceDataset) -> str:
    return target.uri if isinstance(target, lance.LanceDataset) else str(target)


def _reader_from_lazyframe(
    lf: pl.LazyFrame,
    *,
    chunk_size: int,
    engine: EngineType,
) -> pa.RecordBatchReader:
    """Adopt a LazyFrame's streaming output as an Arrow record-batch reader.

    ``collect_batches`` returns an object exposing ``__arrow_c_stream__``, so
    PyArrow can adopt the query's output directly: no copy, no Python generator
    between the engine and Lance's writer, and the reader's schema is the one
    Polars resolves for the plan.
    """
    batches = lf.collect_batches(chunk_size=chunk_size, engine=engine)
    # polars annotates this as `Iterator[DataFrame]`, but the object it returns
    # also implements `__arrow_c_stream__` -- which is the entire point here.
    return pa.RecordBatchReader.from_stream(batches)  # type: ignore[arg-type]


def sink_lance(
    lf: pl.LazyFrame,
    target: str | Path | lance.LanceDataset,
    *,
    mode: WriteMode = "create",
    on: str | list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    engine: EngineType = "streaming",
    lazy: bool = False,
    **lance_write_kwargs: Any,  # noqa: ANN401 - passed through to Lance as given
) -> lance.LanceDataset | pl.LazyFrame:
    """Stream a LazyFrame into a Lance dataset.

    The query is executed in batches and handed to Lance as it goes; the result
    is never materialised in full.

    Parameters
    ----------
    lf
        The query to write.
    target
        Destination URI, path, or an existing :class:`lance.LanceDataset`.
    mode
        ``"create"`` (fail if it exists), ``"append"``, ``"overwrite"`` (new
        version), or ``"merge"`` for an upsert, which requires `on`.
    on
        Join key(s) for ``mode="merge"``.
    chunk_size
        Rows buffered per batch handed to Lance.
    engine
        Polars engine. Leave at ``"streaming"``; ``"in-memory"`` defeats the
        purpose by materialising the result first.
    lazy
        Return a LazyFrame that performs the write when collected, instead of
        writing immediately, so the Lance write can sit inside a larger deferred
        plan. Collecting it yields a one-row summary of what was written.
    **lance_write_kwargs
        Passed through to :func:`lance.write_dataset`, e.g. ``max_rows_per_file``,
        ``storage_options``, ``data_storage_version``.

    Returns
    -------
    lance.LanceDataset
        The written dataset, unless `lazy` is set.
    polars.LazyFrame
        When `lazy` is set.

    Examples
    --------
    >>> query = lf.filter(pl.col("ok"))  # doctest: +SKIP
    >>> sink_lance(query, "out.lance", mode="overwrite")  # doctest: +SKIP

    """
    uri = _target_uri(target)

    if mode == "merge":
        if on is None:
            msg = "mode='merge' requires `on` (the join key column(s))"
            raise ValueError(msg)
        if lazy:
            msg = "mode='merge' does not support lazy=True"
            raise NotImplementedError(msg)
        if lance_write_kwargs:
            msg = (
                "mode='merge' does not accept write_dataset arguments: "
                f"{sorted(lance_write_kwargs)}"
            )
            raise TypeError(msg)
        dataset = (
            target
            if isinstance(target, lance.LanceDataset)
            else lance.dataset(uri, storage_options=None)
        )
        reader = _reader_from_lazyframe(lf, chunk_size=chunk_size, engine=engine)
        dataset.merge_insert(
            on
        ).when_matched_update_all().when_not_matched_insert_all().execute(reader)
        return dataset

    if on is not None:
        msg = f"`on` is only meaningful for mode='merge', not {mode!r}"
        raise ValueError(msg)

    if not lazy:
        reader = _reader_from_lazyframe(lf, chunk_size=chunk_size, engine=engine)
        return lance.write_dataset(reader, uri, mode=mode, **lance_write_kwargs)

    return _lazy_sink(
        lf,
        uri,
        mode=mode,
        chunk_size=chunk_size,
        engine=engine,
        lance_write_kwargs=lance_write_kwargs,
    )


SINK_SUMMARY_SCHEMA = {"uri": pl.String, "version": pl.UInt64, "rows": pl.UInt32}


def _lazy_sink(
    lf: pl.LazyFrame,
    uri: str,
    *,
    mode: WriteMode,
    chunk_size: int,
    engine: EngineType,
    lance_write_kwargs: dict[str, Any],
) -> pl.LazyFrame:
    """Defer the write until the returned LazyFrame is collected.

    Polars gives ``sink_batches`` callbacks no end-of-stream signal, so a
    queue-and-writer-thread arrangement cannot tell when to close the Lance
    writer. Deferring the whole streaming write instead composes cleanly and
    keeps the same bounded-memory behaviour: collecting the result yields a
    one-row summary of what was written.
    """

    def run() -> pl.DataFrame:
        reader = _reader_from_lazyframe(lf, chunk_size=chunk_size, engine=engine)
        dataset = lance.write_dataset(reader, uri, mode=mode, **lance_write_kwargs)
        return pl.DataFrame(
            {
                "uri": [dataset.uri],
                "version": [dataset.version],
                "rows": [dataset.count_rows()],
            },
            schema=SINK_SUMMARY_SCHEMA,
        )

    return pl.defer(run, schema=SINK_SUMMARY_SCHEMA, validate_schema=False)


def write_lance_fragments(
    lazyframes: Iterable[pl.LazyFrame],
    target: str | Path,
    *,
    mode: Literal["create", "overwrite", "append"] = "create",
    max_workers: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    engine: EngineType = "streaming",
    arrow_schema: pa.Schema | None = None,
    **lance_write_kwargs: Any,  # noqa: ANN401 - passed through to Lance as given
) -> lance.LanceDataset:
    """Write several LazyFrames as Lance fragments in parallel, then commit once.

    This is the distributed write shape: every shard streams into its own
    fragment files independently, and a single commit at the end makes them one
    dataset version. Pair with :func:`~polars_pylance.scan_lance_fragments` to get
    the shards.

    Parameters
    ----------
    lazyframes
        One query per shard. All must produce the same schema.
    target
        Destination URI or path.
    mode
        ``"create"``/``"overwrite"`` replace the dataset contents with the shards;
        ``"append"`` adds them to an existing dataset.
    max_workers
        Threads used to write shards. Defaults to one per shard.
    chunk_size
        Rows buffered per batch handed to Lance.
    engine
        Polars engine. Leave at ``"streaming"``; ``"in-memory"`` defeats the
        purpose by materialising each shard first.
    arrow_schema
        Schema to write. Inferred from the first shard when omitted.
    **lance_write_kwargs
        Passed to :func:`lance.fragment.write_fragments`.

    Examples
    --------
    >>> shards = scan_lance_fragments("in.lance")  # doctest: +SKIP
    >>> write_lance_fragments(
    ...     [s.filter(pl.col("ok")) for s in shards], "out.lance"
    ... )  # doctest: +SKIP

    """
    from concurrent.futures import ThreadPoolExecutor

    shards = list(lazyframes)
    if not shards:
        msg = "write_lance_fragments() needs at least one LazyFrame"
        raise ValueError(msg)

    uri = str(target)
    schema = arrow_schema or shards[0].collect_schema().to_arrow()
    fragment_mode = fragment_write_mode(mode)

    def write_shard(shard: pl.LazyFrame) -> list[FragmentMetadata]:
        reader = _reader_from_lazyframe(shard, chunk_size=chunk_size, engine=engine)
        # `return_transaction=False` is the default; naming it picks the
        # overload that returns fragments rather than a transaction, which
        # `**lance_write_kwargs` would otherwise leave unresolved.
        return lance.fragment.write_fragments(
            reader,
            uri,
            schema=schema,
            mode=fragment_mode,
            return_transaction=False,
            **lance_write_kwargs,
        )

    with ThreadPoolExecutor(max_workers=max_workers or len(shards)) as pool:
        written = list(pool.map(write_shard, shards))

    fragments = [f for shard in written for f in shard]
    storage_options = lance_write_kwargs.get("storage_options")
    return commit_lance_fragments(
        uri, fragments, schema=schema, mode=mode, storage_options=storage_options
    )


def fragment_write_mode(mode: Literal["create", "overwrite", "append"]) -> str:
    """The ``write_fragments`` mode that matches a dataset write mode.

    ``"append"`` reuses the existing dataset's field ids, which is what adding
    to it needs. Everything else installs a fresh schema, and that is spelled
    ``"overwrite"`` rather than ``"create"``: ``write_fragments(mode="create")``
    refuses outright if the dataset already exists, before we ever reach the
    commit that would have decided what to do about it.
    """
    return "append" if mode == "append" else "overwrite"


def commit_lance_fragments(
    uri: str,
    fragments: list[Any],
    *,
    schema: pa.Schema,
    mode: Literal["create", "overwrite", "append"] = "create",
    storage_options: dict[str, str] | None = None,
) -> lance.LanceDataset:
    """Publish already-written fragments as one dataset version.

    The second half of a distributed write: the fragments' data files are on
    storage but no manifest references them, so nothing has been published yet.
    This is the single commit that makes them a version -- whether they were
    written by threads (:func:`write_lance_fragments`) or by Polars Cloud
    workers (:func:`~polars_pylance.cloud.sink_lance_remote`).
    """
    operation: lance.LanceOperation.BaseOperation
    if mode == "append":
        operation = lance.LanceOperation.Append(fragments)
        read_version = lance.dataset(uri, storage_options=storage_options).version
        return lance.LanceDataset.commit(
            uri, operation, read_version=read_version, storage_options=storage_options
        )

    if mode == "create" and _dataset_exists(uri, storage_options):
        msg = (
            f"dataset already exists at {uri!r}; use mode='overwrite' to replace "
            "its contents or mode='append' to add to it"
        )
        raise FileExistsError(msg)

    operation = lance.LanceOperation.Overwrite(schema, fragments)
    return lance.LanceDataset.commit(uri, operation, storage_options=storage_options)


def _dataset_exists(uri: str, storage_options: dict[str, str] | None) -> bool:
    """Advisory: whether there is already a dataset here.

    Lance reports "not found" and "could not reach the store" as the same
    ValueError, so an unreachable store reads as absent. That only ever costs a
    clearer error message -- the commit that follows fails on its own -- and
    ``Overwrite`` adds a version rather than destroying the old one.
    """
    try:
        lance.dataset(uri, storage_options=storage_options)
    except (ValueError, OSError):
        return False
    return True
