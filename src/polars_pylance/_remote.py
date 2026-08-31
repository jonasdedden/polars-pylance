"""Distributed Lance writes on Polars Cloud, built on ``sink_batches``.

polars-cloud 0.10 added ``lf.remote(ctx).sink_batches(fn)``: the query runs on
the cluster and a Python callable is invoked with each result batch. That
callable is **cloudpickled into the serialized query plan** -- verifiable with
``polars._utils.cloud.prepare_cloud_plan`` -- so it runs on the workers, not on
the client. Lance therefore no longer has to be a sink format Polars Cloud knows
about: the workers write the data files themselves, in parallel, and this is a
genuinely distributed write rather than a streaming client-side one.

What that costs is the return path. A callback shipped to a worker cannot hand
anything back -- its return value is a stop signal, and mutations to captured
state die with the worker process. Lance's write/commit split is what makes the
arrangement work anyway:

1. Each worker writes **data files only** with ``lance.fragment.write_fragments``
   and commits nothing, so no worker can publish a partial dataset.
2. It then writes the resulting fragment metadata as JSON to a staging prefix
   next to the dataset -- the side channel that replaces the return value.
3. When the query finishes, the client lists that prefix and makes every staged
   fragment one dataset version with a single commit.

Idempotency is the design constraint, not an afterthought: polars-cloud
documents that the callback "might be called multiple times from different
workers", and appending a fragment is not idempotent. Each staging object is
named after a deterministic key derived from the batch contents, so a re-run of
the same batch overwrites its own metadata rather than adding a second copy.
Re-runs do leave their earlier data files behind unreferenced; see
:meth:`StagedLanceSink.commit` for how to reclaim them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import posixpath
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import lance
import polars as pl
import pyarrow as pa
import pyarrow.fs as pafs

from ._sink import (
    _dataset_exists,
    commit_lance_fragments,
    fragment_write_mode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

RemoteWriteMode = Literal["create", "overwrite", "append"]

#: Appended to the dataset URI to get the default staging prefix. A sibling of
#: the dataset rather than a directory inside it, so nothing that inspects a
#: Lance dataset's layout ever sees it.
STAGING_SUFFIX = ".pll-staging"


# ---------------------------------------------------------------------------
# staging object store
# ---------------------------------------------------------------------------


# Lance and PyArrow spell the same S3 settings differently, and only Lance gets
# handed `storage_options`. These are the keys worth translating so that the
# staging channel reaches the same bucket as the data files; anything else falls
# through to PyArrow's own credential resolution.
_S3_OPTION_ALIASES = {
    "access_key_id": "access_key",
    "aws_access_key_id": "access_key",
    "secret_access_key": "secret_key",
    "aws_secret_access_key": "secret_key",
    "session_token": "session_token",
    "aws_session_token": "session_token",
    "token": "session_token",
    "region": "region",
    "aws_region": "region",
    "endpoint": "endpoint_override",
    "endpoint_url": "endpoint_override",
    "aws_endpoint": "endpoint_override",
    "aws_endpoint_url": "endpoint_override",
}
_S3_TRUTHY = {"true", "1", "yes", "on"}


def _normalise_uri(uri: str) -> str:
    """Absolutise a bare filesystem path; leave anything with a scheme alone."""
    if "://" in uri:
        return uri
    return Path(uri).expanduser().resolve().as_uri()


def _fs_path(uri: str) -> str:
    """The path a PyArrow filesystem expects, i.e. the URI minus its scheme."""
    if "://" not in uri:
        return str(Path(uri).expanduser().resolve())
    return uri.split("://", 1)[1]


def _s3_filesystem(storage_options: dict[str, str]) -> pafs.S3FileSystem | None:
    """Translate Lance S3 `storage_options` into an ``S3FileSystem``.

    Returns None when nothing in `storage_options` is recognised, so the caller
    can fall back to PyArrow's own resolution rather than build a filesystem
    that silently ignores the user's credentials.
    """
    kwargs: dict[str, Any] = {}
    for key, value in storage_options.items():
        target = _S3_OPTION_ALIASES.get(key.lower())
        if target is not None:
            kwargs[target] = value
    allow_http = storage_options.get(
        "allow_http", storage_options.get("aws_allow_http")
    )
    if str(allow_http).lower() in _S3_TRUTHY:
        kwargs["scheme"] = "http"
    return pafs.S3FileSystem(**kwargs) if kwargs else None


def _resolve_filesystem(
    uri: str,
    *,
    filesystem: pafs.FileSystem | None,
    storage_options: dict[str, str] | None,
) -> tuple[pafs.FileSystem, str]:
    if filesystem is not None:
        return filesystem, _fs_path(uri)
    if storage_options and uri.startswith("s3://"):
        fs = _s3_filesystem(storage_options)
        if fs is not None:
            return fs, _fs_path(uri)
    return pafs.FileSystem.from_uri(_normalise_uri(uri))


# ---------------------------------------------------------------------------
# the worker-side callback
# ---------------------------------------------------------------------------


def _content_key(df: pl.DataFrame) -> str:
    """A key that is the same for two invocations with the same batch.

    Polars' row hashes over the batch, folded together with its height and
    schema. Deterministic for a given polars version, which is exactly the
    scope that matters: polars-cloud pins client and workers to one version.

    The tradeoff is that two *distinct* batches with byte-identical contents
    collapse onto one key, and only one of them would be committed. That needs
    the query to emit the same rows, in the same order, in two whole chunks --
    pass ``fragment_key=`` to :func:`stage_lance_sink` if your data can do that.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(df.schema).encode())
    digest.update(df.height.to_bytes(8, "big"))
    hashes = df.hash_rows(seed=0).to_arrow()
    if isinstance(hashes, pa.ChunkedArray):
        hashes = hashes.combine_chunks()
    # buffers() is [validity, values]; the validity buffer is None here because
    # hash_rows never produces nulls, but the values buffer is always present.
    values_buffer = hashes.buffers()[1]
    assert values_buffer is not None
    values = memoryview(values_buffer)
    digest.update(values[hashes.offset * 8 : (hashes.offset + len(hashes)) * 8])
    return digest.hexdigest()


@dataclass
class _FragmentWriter:
    """The ``sink_batches`` callback. Writes data files, commits nothing.

    Everything it holds is plain data, so it survives being cloudpickled into
    the query plan. The Arrow schema travels as IPC bytes rather than as a
    pickled :class:`pyarrow.Schema`, which keeps it readable by any pyarrow the
    workers happen to have.
    """

    uri: str
    schema_ipc: bytes
    staging_uri: str
    fragment_mode: str
    storage_options: dict[str, str] | None
    staging_storage_options: dict[str, str] | None
    write_kwargs: dict[str, Any]
    fragment_key: Callable[[pl.DataFrame], str] | None

    def schema(self) -> pa.Schema:
        return pa.ipc.read_schema(pa.py_buffer(self.schema_ipc))

    def __call__(self, df: pl.DataFrame) -> None:
        if df.height == 0:
            return

        schema = self.schema()
        table = df.to_arrow()
        if table.schema != schema:
            table = table.cast(schema)

        key = (self.fragment_key or _content_key)(df)

        fragments = lance.fragment.write_fragments(
            table,
            self.uri,
            schema=schema,
            mode=self.fragment_mode,
            storage_options=self.storage_options,
            **self.write_kwargs,
        )

        # Written after the data files, so a staged entry always refers to
        # files that exist. The reverse order could commit a dangling fragment.
        payload = json.dumps(
            {"key": key, "fragments": [f.to_json() for f in fragments]}
        ).encode()
        # Resolved per call rather than cached on the instance: a live
        # filesystem handle is not picklable, and this object has to be.
        fs, prefix = _resolve_filesystem(
            self.staging_uri,
            filesystem=None,
            storage_options=self.staging_storage_options,
        )
        fs.create_dir(prefix, recursive=True)
        with fs.open_output_stream(posixpath.join(prefix, f"{key}.json")) as sink:
            sink.write(payload)
        return


# ---------------------------------------------------------------------------
# client-side handle
# ---------------------------------------------------------------------------


@dataclass
class StagedLanceSink:
    """A remote Lance write in progress: a callback to ship, and a commit to run.

    Returned by :func:`stage_lance_sink`. Hand :attr:`callback` to
    ``sink_batches``, wait for the query, then call :meth:`commit`.
    """

    uri: str
    staging_uri: str
    mode: RemoteWriteMode
    schema: pa.Schema
    callback: _FragmentWriter
    storage_options: dict[str, str] | None = None
    staging_filesystem: pafs.FileSystem | None = None
    staging_storage_options: dict[str, str] | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def _fs(self) -> tuple[pafs.FileSystem, str]:
        return _resolve_filesystem(
            self.staging_uri,
            filesystem=self.staging_filesystem,
            storage_options=self.staging_storage_options,
        )

    def staged_fragments(self) -> list[Any]:
        """Every fragment the workers staged, in a deterministic order.

        Ordered by staging key so that two commits of the same staged output
        produce the same fragment layout.
        """
        fs, prefix = self._fs()
        selector = pafs.FileSelector(prefix, recursive=False, allow_not_found=True)
        entries: dict[str, list[Any]] = {}
        for info in fs.get_file_info(selector):
            if info.type != pafs.FileType.File or not info.path.endswith(".json"):
                continue
            with fs.open_input_stream(info.path) as stream:
                payload = json.loads(stream.readall())
            # Keyed by content, and the key is the object name, so a retried
            # batch has already overwritten itself. Re-keying here is belt and
            # braces against a caller-supplied `fragment_key` colliding.
            entries[payload["key"]] = payload["fragments"]

        return [
            lance.fragment.FragmentMetadata.from_json(json.dumps(fragment))
            for key in sorted(entries)
            for fragment in entries[key]
        ]

    def commit(self, *, cleanup: bool = True) -> lance.LanceDataset:
        """Make every staged fragment one dataset version, in a single commit.

        Parameters
        ----------
        cleanup
            Remove the staging prefix once the commit lands. The *data* files a
            retried batch orphaned are not touched -- they live inside the
            dataset and are unreferenced by any manifest, so reclaim them with
            ``dataset.cleanup_old_versions(..., delete_unverified=True)``.

        Raises
        ------
        ValueError
            If nothing was staged. An empty commit would replace the dataset
            with nothing, which is never what a failed remote query meant.
        """
        fragments = self.staged_fragments()
        if not fragments:
            msg = (
                f"nothing staged under {self.staging_uri!r}: the remote query "
                "wrote no batches, or the workers could not reach the staging "
                "prefix (pass `staging_uri` / `staging_storage_options`)"
            )
            raise ValueError(msg)

        dataset = commit_lance_fragments(
            self.uri,
            fragments,
            schema=self.schema,
            mode=self.mode,
            storage_options=self.storage_options,
        )
        if cleanup:
            self.cleanup()
        return dataset

    def cleanup(self) -> None:
        """Delete the staging prefix. Safe to call when it was never created."""
        fs, prefix = self._fs()
        with contextlib.suppress(FileNotFoundError):
            fs.delete_dir(prefix)


def stage_lance_sink(
    target: str | Path | lance.LanceDataset,
    schema: pa.Schema | pl.Schema | pl.LazyFrame,
    *,
    mode: RemoteWriteMode = "create",
    storage_options: dict[str, str] | None = None,
    staging_uri: str | None = None,
    staging_filesystem: pafs.FileSystem | None = None,
    staging_storage_options: dict[str, str] | None = None,
    fragment_key: Callable[[pl.DataFrame], str] | None = None,
    **lance_write_kwargs: Any,
) -> StagedLanceSink:
    """Build a worker-side Lance writer for ``sink_batches``.

    Use this when you want to drive the remote query yourself -- to pick a
    planner, set ``maintain_order``, or inspect the query handle.
    :func:`sink_lance_remote` is the same thing with the query submitted and
    awaited for you.

    Parameters
    ----------
    target
        Destination URI, path, or an existing :class:`lance.LanceDataset`.
    schema
        The query's output schema. A LazyFrame is accepted and resolved with
        ``collect_schema()``, which needs the *client* to be able to reach the
        sources -- as it already must be to build a ``scan_lance`` plan.
    mode
        ``"create"`` (fail if the dataset exists), ``"overwrite"`` (replace its
        contents with a new version), or ``"append"``.
    storage_options
        Passed to Lance for the data files, and -- for ``s3://`` targets --
        translated into a PyArrow filesystem for the staging prefix.
    staging_uri
        Where fragment metadata is staged. Defaults to the dataset URI plus
        ``.pll-staging``, with a per-run subdirectory so concurrent writes to
        one dataset do not read each other's fragments.
    staging_filesystem
        An explicit :class:`pyarrow.fs.FileSystem` for the staging prefix, for
        stores whose credentials do not translate. Client-side only: the
        callback resolves the staging filesystem itself on the worker, from
        `staging_storage_options` or the worker's ambient credentials.
    staging_storage_options
        Staging credentials, when they differ from `storage_options`.
    fragment_key
        ``(batch) -> str``, replacing the default content digest. Must be
        deterministic: two invocations for the same batch must agree, and two
        distinct batches must not. Worth supplying when the query carries a
        natural key, such as a partition column.
    **lance_write_kwargs
        Passed to :func:`lance.fragment.write_fragments`, e.g.
        ``max_rows_per_file``, ``data_storage_version``.

    Examples
    --------
    >>> staged = stage_lance_sink(
    ...     "s3://bucket/out.lance", lf, mode="overwrite"
    ... )  # doctest: +SKIP
    >>> query = (
    ...     lf.remote(ctx).distributed().sink_batches(staged.callback)
    ... )  # doctest: +SKIP
    >>> query.await_result()  # doctest: +SKIP
    >>> staged.commit()  # doctest: +SKIP
    """
    uri = target.uri if isinstance(target, lance.LanceDataset) else str(target)
    arrow_schema = _as_arrow_schema(schema)
    run_id = uuid.uuid4().hex
    base = staging_uri if staging_uri is not None else uri.rstrip("/") + STAGING_SUFFIX
    run_staging = f"{base.rstrip('/')}/{run_id}"

    if staging_storage_options is None and storage_options is not None:
        staging_storage_options = storage_options

    # Checked again at commit time, which is authoritative. Doing it here too
    # means `mode="create"` over an existing dataset costs nothing rather than a
    # whole cluster run that is discarded.
    if mode == "create" and _dataset_exists(uri, storage_options):
        msg = (
            f"dataset already exists at {uri!r}; use mode='overwrite' to replace "
            "its contents or mode='append' to add to it"
        )
        raise FileExistsError(msg)

    fragment_mode = fragment_write_mode(mode)

    callback = _FragmentWriter(
        uri=uri,
        schema_ipc=arrow_schema.serialize().to_pybytes(),
        staging_uri=run_staging,
        fragment_mode=fragment_mode,
        storage_options=storage_options,
        staging_storage_options=staging_storage_options,
        write_kwargs=dict(lance_write_kwargs),
        fragment_key=fragment_key,
    )

    return StagedLanceSink(
        uri=uri,
        staging_uri=run_staging,
        mode=mode,
        schema=arrow_schema,
        callback=callback,
        storage_options=storage_options,
        staging_filesystem=staging_filesystem,
        staging_storage_options=staging_storage_options,
        run_id=run_id,
    )


def _as_arrow_schema(schema: pa.Schema | pl.Schema | pl.LazyFrame) -> pa.Schema:
    if isinstance(schema, pa.Schema):
        return schema
    if isinstance(schema, pl.LazyFrame):
        return schema.collect_schema().to_arrow()
    if isinstance(schema, pl.Schema):
        return schema.to_arrow()
    msg = (
        "schema must be a pyarrow.Schema, polars.Schema or LazyFrame, "
        f"got {type(schema).__name__}"
    )
    raise TypeError(msg)


def sink_lance_remote(
    remote: Any,
    target: str | Path | lance.LanceDataset,
    *,
    schema: pa.Schema | pl.Schema | None = None,
    mode: RemoteWriteMode = "create",
    chunk_size: int | None = None,
    maintain_order: bool = False,
    storage_options: dict[str, str] | None = None,
    staging_uri: str | None = None,
    staging_filesystem: pafs.FileSystem | None = None,
    staging_storage_options: dict[str, str] | None = None,
    fragment_key: Callable[[pl.DataFrame], str] | None = None,
    cleanup: bool = True,
    **lance_write_kwargs: Any,
) -> lance.LanceDataset:
    """Run a Polars Cloud query and write its output to Lance from the workers.

    Submits `remote` with a worker-side fragment writer, waits for it, and
    commits every staged fragment as one dataset version.

    Parameters
    ----------
    remote
        A ``polars_cloud.LazyFrameRemote``, i.e. the result of
        ``lf.remote(ctx)`` and any of ``.distributed()`` / ``.single_node()``.
    target
        Destination URI, path, or an existing :class:`lance.LanceDataset`.
    schema
        The query's output schema. Resolved from the LazyFrame behind `remote`
        when omitted, which requires the client to be able to reach the sources.
    chunk_size
        Rows buffered before the callback runs; the remote counterpart of
        ``max_rows_per_file``. Left to Polars Cloud when omitted.
    maintain_order
        Call the writer serially rather than in parallel across workers. Costs
        the parallelism that makes this a distributed write; the fragment order
        at commit is deterministic either way.
    cleanup
        Remove the staging prefix after a successful commit. On failure it is
        always left in place, so a re-run can be diagnosed or the fragments
        committed by hand.

    Other parameters are as :func:`stage_lance_sink`.

    Returns
    -------
    lance.LanceDataset
        The committed dataset.

    Examples
    --------
    >>> ctx = pc.ComputeContext(
    ...     cpus=8, memory=32, requirements=requirements_txt().encode()
    ... )  # doctest: +SKIP
    >>> lf = scan_lance("s3://bucket/in.lance").filter(
    ...     pl.col("score") > 0.9
    ... )  # doctest: +SKIP
    >>> sink_lance_remote(
    ...     lf.remote(ctx).distributed(), "s3://bucket/out.lance", mode="overwrite"
    ... )  # doctest: +SKIP
    """
    if schema is None:
        lf = getattr(remote, "lf", None)
        if lf is None:
            msg = (
                "could not read the LazyFrame behind `remote` to infer the "
                "output schema; pass `schema=`"
            )
            raise TypeError(msg)
        # `remote` is untyped, so `lf` is `Any` and whatever it returns here
        # is unchecked. `_as_arrow_schema` would reject a non-schema further in;
        # this says so where the value enters, and narrows away the `None` the
        # parameter still declares.
        collected = lf.collect_schema()
        if not isinstance(collected, pl.Schema):
            msg = (
                "`remote.lf.collect_schema()` did not return a Polars schema; "
                "pass `schema=`"
            )
            raise TypeError(msg)
        schema = collected

    staged = stage_lance_sink(
        target,
        schema,
        mode=mode,
        storage_options=storage_options,
        staging_uri=staging_uri,
        staging_filesystem=staging_filesystem,
        staging_storage_options=staging_storage_options,
        fragment_key=fragment_key,
        **lance_write_kwargs,
    )

    query = remote.sink_batches(
        staged.callback, chunk_size=chunk_size, maintain_order=maintain_order
    )
    query.await_result()

    return staged.commit(cleanup=cleanup)
