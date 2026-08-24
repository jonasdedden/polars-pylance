"""Lazy, streaming Lance reader for Polars.

Two implementations of the same scan, sharing one batch-producing core:

``provider``
    Registers a dataset object through ``PyLazyFrame.new_from_dataset_object``,
    the hook behind :func:`polars.scan_delta` and :func:`polars.scan_iceberg`.
    Polars resolves the scan at IR-resolution time and hands over the projection,
    the row limit, the columns a filter touches, and the filter itself already
    translated to a PyArrow expression -- which is exactly what Lance accepts.
    It also passes back the version key of the previous resolution so an
    unchanged dataset need not be re-planned. This is the default.

``io_plugin``
    Uses the public :func:`polars.io.plugins.register_io_source`. The predicate
    arrives as a Polars expression instead, so it is applied per batch and, where
    possible, additionally translated to a Lance SQL string for page skipping.

The hook used by ``provider`` is private and unstable, hence the fallback. Both
satisfy the same test suite.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import lance
import polars as pl
import pyarrow as pa

from polars_lance._options import LanceScanOptions
from polars_lance._predicate import to_lance_filter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

ScanImpl = Literal["auto", "provider", "io_plugin"]

# Columns Lance synthesises rather than reads. They are appended to the output by
# the scanner itself and must be kept out of the `columns=` projection.
VIRTUAL_COLUMNS = frozenset(
    {"_rowid", "_rowaddr", "_distance", "_score", "query_index"}
)


@dataclass
class LanceScanSpec:
    """Everything needed to reproduce a scan, and nothing that cannot be pickled.

    Holding a URI rather than an open :class:`lance.LanceDataset` is what lets a
    scan be serialized into a query plan and executed elsewhere, which is the
    prerequisite for Polars Cloud.
    """

    uri: str
    version: int | str | None = None
    storage_options: dict[str, str] | None = None
    options: LanceScanOptions = field(default_factory=LanceScanOptions)
    nearest: dict[str, Any] | None = None
    full_text_query: str | dict[str, Any] | None = None
    with_row_id: bool = False
    with_row_address: bool = False
    fragment_ids: list[int] | None = None
    predicate_pushdown: bool = True

    # -- dataset access ----------------------------------------------------

    def open(self) -> lance.LanceDataset:
        return lance.dataset(
            self.uri, version=self.version, storage_options=self.storage_options
        )

    def scanner(
        self,
        dataset: lance.LanceDataset,
        *,
        columns: list[str] | None = None,
        filter: pa.compute.Expression | str | None = None,
        limit: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            **self.options.to_scan_kwargs(),
            "columns": columns,
            "filter": filter,
            "limit": limit,
        }
        if self.nearest is not None:
            kwargs["nearest"] = self.nearest
        if self.full_text_query is not None:
            kwargs["full_text_query"] = self.full_text_query
        if self.with_row_id:
            kwargs["with_row_id"] = True
        if self.with_row_address:
            kwargs["with_row_address"] = True
        if self.fragment_ids is not None:
            by_id = {f.fragment_id: f for f in dataset.get_fragments()}
            missing = [i for i in self.fragment_ids if i not in by_id]
            if missing:
                msg = f"no such fragment(s) in {self.uri}: {missing}"
                raise ValueError(msg)
            kwargs["fragments"] = [by_id[i] for i in self.fragment_ids]
        return dataset.scanner(**kwargs)

    def arrow_schema(self, dataset: lance.LanceDataset | None = None) -> pa.Schema:
        """Full output schema, including any Lance-generated columns."""
        dataset = dataset if dataset is not None else self.open()
        return self.scanner(dataset).projected_schema

    def polars_schema(self, dataset: lance.LanceDataset | None = None) -> pl.Schema:
        arrow = self.arrow_schema(dataset)
        return pl.from_arrow(arrow.empty_table()).schema  # type: ignore[union-attr]

    # -- batch production --------------------------------------------------

    def iter_frames(
        self,
        dataset: lance.LanceDataset,
        *,
        projection: Sequence[str] | None = None,
        filter: pa.compute.Expression | str | None = None,
        limit: int | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Stream `projection` out of Lance as Polars frames.

        Lazy by construction: nothing is read until the consumer pulls, and
        dropping the generator early stops the scan.
        """
        columns = None if projection is None else self._physical_columns(projection)
        scanner = self.scanner(dataset, columns=columns, filter=filter, limit=limit)

        remaining = limit
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            if remaining is not None:
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                remaining -= batch.num_rows

            if batch.num_columns == 0:
                # A column-less batch still carries a row count (this is what a
                # bare `pl.len()` projects to), but Arrow -> Polars conversion
                # would lose it.
                yield pl.DataFrame(height=batch.num_rows)
                continue

            frame: pl.DataFrame = pl.from_arrow(batch)  # type: ignore[assignment]
            if projection is not None and frame.columns != list(projection):
                # Lance appends generated columns after the requested ones; the
                # engine expects exactly the projection, in order.
                frame = frame.select(projection)
            yield frame

    def _physical_columns(self, projection: Sequence[str]) -> list[str]:
        # Generated columns are added by the scanner itself and must not appear
        # in `columns=`; an empty result is legal and means "generated only".
        return [c for c in projection if c not in VIRTUAL_COLUMNS]


# ---------------------------------------------------------------------------
# provider implementation (private polars hook, used by scan_delta/scan_iceberg)
# ---------------------------------------------------------------------------


def _pyarrow_eval_namespace() -> dict[str, Any]:
    """The namespace Polars' generated PyArrow predicate strings expect.

    Mirrors ``polars.io.delta._dataset``; kept in one place so a change upstream
    surfaces here rather than in the middle of a scan.
    """
    from polars._utils.convert import (
        to_py_date,
        to_py_datetime,
        to_py_time,
        to_py_timedelta,
    )
    from polars.datatypes import Date, Datetime, Duration

    return {
        "pa": pa,
        "Date": Date,
        "Datetime": Datetime,
        "Duration": Duration,
        "to_py_date": to_py_date,
        "to_py_datetime": to_py_datetime,
        "to_py_time": to_py_time,
        "to_py_timedelta": to_py_timedelta,
    }


class LanceDatasetProvider:
    """Implements Polars' (private) dataset-provider interface for Lance."""

    def __init__(self, spec: LanceScanSpec) -> None:
        self.spec = spec

    # Interface: schema()
    def schema(self) -> pa.Schema:
        return self.spec.arrow_schema()

    # Interface: to_dataset_scan(). Polars omits keys that do not apply, so
    # every parameter needs a default.
    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> tuple[pl.LazyFrame, str] | None:
        dataset = self.spec.open()
        version_key = str(dataset.version)
        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == version_key
        ):
            return None

        pa_filter = None
        if pyarrow_predicate is not None and self.spec.predicate_pushdown:
            try:
                pa_filter = eval(pyarrow_predicate, _pyarrow_eval_namespace())
            except Exception as exc:
                warnings.warn(
                    "polars-lance: could not evaluate the predicate Polars "
                    f"generated ({exc!r}); filtering falls back to the engine",
                    RuntimeWarning,
                    stacklevel=2,
                )

        predicate_applied = pa_filter is not None
        # The limit may only be pushed into Lance when no rows will be removed
        # downstream of the scan; otherwise it would truncate before filtering.
        # Polars is not observed to send both, but be explicit about it.
        pushed_limit = (
            limit if (pyarrow_predicate is None or predicate_applied) else None
        )

        spec = self.spec

        def impl(*_args: Any, **_kwargs: Any) -> tuple[Iterator[pl.DataFrame], bool]:
            # Called with no arguments: all pushdown state is captured here.
            frames = spec.iter_frames(
                dataset,
                projection=projection,
                filter=pa_filter,
                limit=pushed_limit,
            )
            return frames, predicate_applied

        lf = pl.LazyFrame._scan_python_function(
            self.spec.arrow_schema(dataset), impl, pyarrow=True, is_pure=True
        )
        return lf, version_key

    def __repr__(self) -> str:
        return f"LanceDatasetProvider({self.spec.uri!r}, version={self.spec.version!r})"


def _provider_lazyframe(spec: LanceScanSpec) -> pl.LazyFrame:
    from polars._plr import PyLazyFrame
    from polars._utils.wrap import wrap_ldf

    return wrap_ldf(PyLazyFrame.new_from_dataset_object(LanceDatasetProvider(spec)))


def _provider_available() -> bool:
    try:
        from polars._plr import PyLazyFrame
    except ImportError:
        return False
    return hasattr(PyLazyFrame, "new_from_dataset_object")


# ---------------------------------------------------------------------------
# io_plugin implementation (public polars API)
# ---------------------------------------------------------------------------


def _io_plugin_lazyframe(spec: LanceScanSpec) -> pl.LazyFrame:
    schema = spec.polars_schema()

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        dataset = spec.open()
        # Polars includes the predicate's columns in `with_columns`, so the
        # predicate can always be evaluated on what we are about to yield.
        sql_filter = (
            to_lance_filter(predicate)
            if predicate is not None and spec.predicate_pushdown
            else None
        )
        scan_spec = spec if batch_size is None else _with_batch_size(spec, batch_size)

        remaining = n_rows
        for frame in scan_spec.iter_frames(
            dataset, projection=with_columns, filter=sql_filter
        ):
            # The SQL filter is a page-skipping optimisation only; correctness
            # comes from re-applying the original predicate here.
            out = frame if predicate is None else frame.filter(predicate)
            if remaining is not None:
                if out.height > remaining:
                    out = out.head(remaining)
                remaining -= out.height
            if out.height:
                yield out
            if remaining is not None and remaining <= 0:
                return

    return pl.io.plugins.register_io_source(
        source, schema=schema, validate_schema=False, is_pure=True
    )


def _with_batch_size(spec: LanceScanSpec, batch_size: int) -> LanceScanSpec:
    """Honour the engine's batch-size hint without mutating the shared spec."""
    return dataclasses.replace(
        spec, options=spec.options.replace(batch_size=batch_size)
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def scan_lance(
    source: str | Path | lance.LanceDataset,
    *,
    version: int | str | None = None,
    storage_options: dict[str, str] | None = None,
    options: LanceScanOptions | None = None,
    nearest: dict[str, Any] | None = None,
    full_text_query: str | dict[str, Any] | None = None,
    with_row_id: bool = False,
    with_row_address: bool = False,
    fragments: Sequence[int] | None = None,
    predicate_pushdown: bool = True,
    impl: ScanImpl = "auto",
) -> pl.LazyFrame:
    """Lazily read a Lance dataset as a Polars :class:`~polars.LazyFrame`.

    Nothing is read when this returns. Column projection, filters and (when no
    filter is present) row limits are pushed into Lance; batches are pulled only
    as the query consumes them, so ``.head()`` stops the scan early. Use
    ``engine="streaming"`` when collecting: the in-memory engine materialises the
    whole result and gives up the memory advantage.

    Parameters
    ----------
    source
        Dataset URI, path, or an open :class:`lance.LanceDataset`. Passing a
        dataset object pins the scan to that dataset's version; passing a URI
        reads whatever version is current when the query runs.
    version
        Read a specific version (or tag) instead of the latest.
    storage_options
        Object-store credentials and settings, passed to Lance.
    options
        Reader tuning; see :class:`~polars_lance.LanceScanOptions`. The defaults
        favour bounded memory over raw IO parallelism.
    nearest
        Lance vector-search specification, e.g.
        ``{"column": "vector", "q": query, "k": 10}``. Adds a ``_distance``
        column. Not expressible as a Polars predicate, hence a scan argument.
    full_text_query
        Lance full-text search query. Adds a ``_score`` column.
    with_row_id, with_row_address
        Include Lance's ``_rowid`` / ``_rowaddr`` columns.
    fragments
        Restrict the scan to these fragment ids. See
        :func:`~polars_lance.scan_lance_fragments` for the sharded form.
    predicate_pushdown
        Set to False to keep filtering entirely in Polars. Worth trying if you
        depend on Polars' null comparison semantics, which differ from SQL's.
    impl
        Which Polars hook to scan through. ``"auto"`` prefers ``"provider"``
        (better pushdown, smaller serialized plans) and falls back to
        ``"io_plugin"`` if the private hook is unavailable.

    Examples
    --------
    >>> lf = scan_lance("s3://bucket/data.lance")  # doctest: +SKIP
    >>> lf.filter(pl.col("label") == 3).select("id", "score").collect(
    ...     engine="streaming"
    ... )  # doctest: +SKIP
    """
    if isinstance(source, lance.LanceDataset):
        uri = source.uri
        version = version if version is not None else source.version
    else:
        uri = str(source)

    spec = LanceScanSpec(
        uri=uri,
        version=version,
        storage_options=storage_options,
        options=options if options is not None else LanceScanOptions(),
        nearest=nearest,
        full_text_query=full_text_query,
        with_row_id=with_row_id,
        with_row_address=with_row_address,
        fragment_ids=list(fragments) if fragments is not None else None,
        predicate_pushdown=predicate_pushdown,
    )

    if impl == "auto":
        impl = "provider" if _provider_available() else "io_plugin"
    if impl == "provider":
        if not _provider_available():
            msg = (
                "impl='provider' needs polars._plr.PyLazyFrame."
                "new_from_dataset_object, which this Polars build does not have; "
                "use impl='io_plugin'"
            )
            raise RuntimeError(msg)
        return _provider_lazyframe(spec)
    if impl == "io_plugin":
        return _io_plugin_lazyframe(spec)

    msg = f"unknown impl: {impl!r}"
    raise ValueError(msg)


def scan_lance_fragments(
    source: str | Path | lance.LanceDataset,
    *,
    n_shards: int | None = None,
    **kwargs: Any,
) -> list[pl.LazyFrame]:
    """Return one LazyFrame per fragment (or per shard) of a Lance dataset.

    Lance fragments are the natural unit of parallelism: each is a self-contained
    set of files. Use this to fan a scan out over threads, processes or workers,
    then recombine with :func:`polars.concat`. Also the manual parallelisation
    route if a distributed planner refuses a Python scan node.

    Examples
    --------
    >>> shards = scan_lance_fragments("data.lance", n_shards=4)  # doctest: +SKIP
    >>> pl.concat(shards).collect(engine="streaming")  # doctest: +SKIP
    """
    dataset = (
        source
        if isinstance(source, lance.LanceDataset)
        else lance.dataset(
            str(source),
            version=kwargs.get("version"),
            storage_options=kwargs.get("storage_options"),
        )
    )
    ids = [f.fragment_id for f in dataset.get_fragments()]
    if not ids:
        return [scan_lance(source, fragments=[], **kwargs)]

    if n_shards is None:
        groups = [[i] for i in ids]
    else:
        if n_shards < 1:
            msg = f"n_shards must be >= 1, got {n_shards}"
            raise ValueError(msg)
        groups = [chunk for s in range(n_shards) if (chunk := ids[s::n_shards])]

    return [scan_lance(source, fragments=g, **kwargs) for g in groups]
