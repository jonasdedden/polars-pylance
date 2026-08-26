"""Lazy, streaming Lance reader for Polars.

Polars offers two ways to plug a foreign source into a query, and they differ in
*what the source is told about the filter*:

``provider`` (default)
    Registers a dataset object through ``PyLazyFrame.new_from_dataset_object``,
    the private hook behind :func:`polars.scan_delta` and
    :func:`polars.scan_iceberg`. Polars resolves the scan while building the IR
    and hands over the projection, the row limit, the columns a filter touches,
    and the filter *already translated to a PyArrow expression* -- which is
    exactly what Lance accepts. It also passes back the version key of the
    previous resolution, so an unchanged dataset need not be re-planned. The
    catch is that Polars' PyArrow lowering is narrow: no ``is_in``, no string
    functions, no arithmetic, no temporal parts, no list or struct access. What
    it cannot lower, it drops, and the scan then reads rows the filter would
    have skipped.

``io_plugin``
    The public :func:`polars.io.plugins.register_io_source`. It hands over the
    *whole* predicate as a :class:`polars.Expr`, which
    :mod:`polars_pylance._predicate` lowers into Lance SQL -- a much wider
    language. Predicates the provider path leaves entirely to the engine reach
    Lance here, which is worth a great deal when Lance can answer them from a
    scalar index or skip pages of a wide column.

Both hooks are unstable API (one private, one marked unstable), and both are
used deliberately. ``docs/PREDICATE_PUSHDOWN.md`` has the measurements behind
the default.

The split is not fundamental. A Polars that also passed the provider its
*serialized* predicate would let the provider path lower filters as widely as
the IO-plugin path; :meth:`LanceDatasetProvider.to_dataset_scan` already accepts
that argument, and uses it when a Polars offers it.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lance
import polars as pl
import pyarrow as pa

from ._options import LanceScanOptions
from ._predicate import VIRTUAL_COLUMNS, to_lance_filter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Literal

    # `pyarrow` does not re-export `compute` from its top level, so `pa.compute`
    # only resolves once something else has imported the submodule.
    import pyarrow.compute as pc

# `VIRTUAL_COLUMNS` -- the columns Lance synthesises rather than reads -- is
# imported above rather than defined here: the predicate lowering has to refuse
# them, and one list beats two.


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
        filter: pc.Expression | str | None = None,
        limit: int | None = None,
    ) -> lance.LanceScanner:
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
        return _polars_schema(self.arrow_schema(dataset))

    # -- batch production --------------------------------------------------

    def iter_frames(
        self,
        dataset: lance.LanceDataset,
        *,
        projection: Sequence[str] | None = None,
        filter: pc.Expression | str | None = None,
        limit: int | None = None,
        filter_is_optional: bool = False,
    ) -> Iterator[pl.DataFrame]:
        """Stream `projection` out of Lance as Polars frames.

        Lazy by construction: nothing is read until the consumer pulls, and
        dropping the generator early stops the scan.

        `filter_is_optional` says the caller re-applies the predicate itself, so
        a filter Lance refuses to plan may be dropped rather than raised. Lance
        validates a filter while building the plan, before any batch is
        produced, so dropping it can never duplicate rows already yielded.
        """
        columns = None if projection is None else self._physical_columns(projection)

        batches = self._batches(
            dataset,
            columns=columns,
            filter=filter,
            limit=limit,
            filter_is_optional=filter_is_optional,
        )

        remaining = limit
        for batch in batches:
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

            frame = cast("pl.DataFrame", pl.from_arrow(batch))
            if projection is not None and frame.columns != list(projection):
                # Lance appends generated columns after the requested ones; the
                # engine expects exactly the projection, in order.
                frame = frame.select(projection)
            yield frame

    def _batches(
        self,
        dataset: lance.LanceDataset,
        *,
        columns: list[str] | None,
        filter: pc.Expression | str | None,
        limit: int | None,
        filter_is_optional: bool,
    ) -> Iterator[pa.RecordBatch]:
        """`scanner.to_batches()`, retried without the filter if Lance says no.

        Lance rejects a filter it cannot plan -- an out-of-range literal for the
        column's type, say -- and it does so on the first pull, before any
        batch exists. When the caller is filtering anyway, a scan without the
        hint is still correct and much better than a failed query.
        """
        scanner = self.scanner(dataset, columns=columns, filter=filter, limit=limit)
        try:
            iterator = iter(scanner.to_batches())
            first = next(iterator, None)
        except Exception as exc:
            if filter is None or not filter_is_optional:
                raise
            warnings.warn(
                f"polars-pylance: Lance rejected the pushed-down filter ({exc}); "
                "scanning without it",
                RuntimeWarning,
                stacklevel=2,
            )
            yield from self.scanner(
                dataset, columns=columns, filter=None, limit=limit
            ).to_batches()
            return
        if first is None:
            return
        yield first
        yield from iterator

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


def _polars_schema(schema: pa.Schema) -> pl.Schema:
    """The Polars view of an Arrow schema, without reading a row."""
    return cast("pl.DataFrame", pl.from_arrow(schema.empty_table())).schema


def _unsafe_for_lance(pyarrow_predicate: str, schema: pa.Schema) -> bool:
    """Whether Lance would answer this PyArrow expression with the wrong rows.

    Lance takes a PyArrow filter through Substrait, and as of pylance 10.0.0 a
    comparison against a **timezone-naive timestamp** column comes back with
    every row or none of them, silently -- so a scan that pushes it returns a
    wrong answer rather than a slow one::

        ts = pa.array([...], pa.timestamp("us"))                # 100 rows
        ds.scanner(filter=pc.field("ts") > pa.scalar(cut)).to_table()   # 0 rows
        ds.scanner(filter="ts > timestamp '...'").to_table()            # correct

    Every other type this package can push is either right (`date32`, numbers,
    strings) or *loudly* wrong -- `date64`, `time64`, `duration` and a
    timestamp with a time zone all raise, which `iter_frames` catches, warns
    about and scans without. Only the naive timestamp is silent, so only it is
    refused here. Lance's own SQL dialect is unaffected, which is what the
    predicate visitor emits, so a Polars carrying
    `#28995 <https://github.com/pola-rs/polars/pull/28995>`_ pushes this
    predicate correctly instead of not at all.

    Matching on the generated source is crude but sound in the direction that
    matters: Polars spells a column as ``pa.compute.field('name')``, and a
    false positive costs pushdown rather than rows.
    """
    return any(
        pa.types.is_timestamp(field.type)
        and field.type.tz is None
        and f"'{field.name}'" in pyarrow_predicate
        for field in schema
    )


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
        serialized_predicate: bytes | None = None,
    ) -> tuple[pl.LazyFrame, str] | None:
        dataset = self.spec.open()
        version_key = str(dataset.version)
        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == version_key
        ):
            return None

        # Resolved once and reused: the returned LazyFrame declares it, and the
        # lowering below reads it to decide whether a cast is redundant.
        arrow_schema = self.spec.arrow_schema(dataset)

        pa_filter: pc.Expression | str | None = None
        if self.spec.predicate_pushdown:
            pa_filter = self._filter(
                arrow_schema, pyarrow_predicate, serialized_predicate
            )

        # Polars keeps the filter above a provider-resolved scan and re-applies
        # it whatever this flag says (measured on both engines in 1.44), and it
        # has to: when only part of a conjunction lowers to PyArrow, only that
        # part arrives here. Reporting False is the reading that stays correct
        # if a future Polars starts honouring the flag.
        predicate_applied = False
        # The limit may only be pushed into Lance when no rows will be removed
        # downstream of the scan; otherwise it would truncate before filtering.
        # Polars is not observed to send both, but be explicit about it.
        pushed_limit = limit if pyarrow_predicate is None else None

        spec = self.spec

        def impl(*_args: Any, **_kwargs: Any) -> tuple[Iterator[pl.DataFrame], bool]:
            # Called with no arguments: all pushdown state is captured here.
            frames = spec.iter_frames(
                dataset,
                projection=projection,
                filter=pa_filter,
                limit=pushed_limit,
                # Reporting the predicate as unapplied is what makes the filter
                # droppable: the engine evaluates it above the scan either way.
                filter_is_optional=not predicate_applied,
            )
            return frames, predicate_applied

        lf = pl.LazyFrame._scan_python_function(
            arrow_schema, impl, pyarrow=True, is_pure=True
        )
        return lf, version_key

    def _filter(
        self,
        schema: pa.Schema,
        pyarrow_predicate: str | None,
        serialized_predicate: bytes | None,
    ) -> pc.Expression | str | None:
        """The best filter available from what Polars passed.

        `serialized_predicate` is the whole predicate and only arrives from a
        Polars that offers it (see docs/PREDICATE_PUSHDOWN.md); `pyarrow_predicate`
        is the part Polars could lower itself. Lowering the whole one covers far
        more of the expression language, but it is also allowed to give up on a
        conjunct, so a relaxed lowering defers to Polars' own where there is one.
        """
        lowered = None
        if serialized_predicate is not None:
            try:
                lowered = to_lance_filter(
                    pl.Expr.deserialize(serialized_predicate),
                    # Only ever drops a cast the schema shows is redundant, and
                    # a redundant cast costs Lance's scalar index.
                    schema=_polars_schema(schema),
                )
            except Exception as exc:
                warnings.warn(
                    "polars-pylance: could not read the predicate Polars passed "
                    f"({exc!r}); falling back to its PyArrow lowering",
                    RuntimeWarning,
                    stacklevel=3,
                )

        if lowered is not None and (lowered.exact or pyarrow_predicate is None):
            return lowered.sql

        if pyarrow_predicate is None or _unsafe_for_lance(pyarrow_predicate, schema):
            return lowered.sql if lowered is not None else None

        try:
            return cast(
                "pc.Expression", eval(pyarrow_predicate, _pyarrow_eval_namespace())
            )
        except Exception as exc:
            warnings.warn(
                "polars-pylance: could not evaluate the predicate Polars "
                f"generated ({exc!r}); filtering falls back to the engine",
                RuntimeWarning,
                stacklevel=3,
            )
            return lowered.sql if lowered is not None else None

    def __repr__(self) -> str:
        return f"LanceDatasetProvider({self.spec.uri!r}, version={self.spec.version!r})"


def _provider_lazyframe(spec: LanceScanSpec) -> pl.LazyFrame:
    from polars._plr import PyLazyFrame
    from polars._utils.wrap import wrap_ldf

    return wrap_ldf(PyLazyFrame.new_from_dataset_object(LanceDatasetProvider(spec)))


# ---------------------------------------------------------------------------
# io_plugin implementation (public polars API, sees the whole predicate)
# ---------------------------------------------------------------------------


def _io_plugin_lazyframe(spec: LanceScanSpec) -> pl.LazyFrame:
    """An IO-plugin scan that leaves the second evaluation to the engine.

    :func:`polars.io.plugins.register_io_source` is the obvious way to build
    this, but it hardcodes "the source applied the predicate". A source pushing a
    *relaxed* filter -- which this one does, whenever a conjunct has no Lance
    spelling -- then has to re-apply the predicate itself, and it can only do
    that per batch, from Python, once per morsel. Reporting the predicate as
    unapplied instead hands that job to the streaming engine, which evaluates it
    over the same batch inside the scan node: same rows, same pushed filter, one
    less crossing of the FFI boundary. Measured at 0.104 s against 0.039 s on a
    4M-row filtered scan; see docs/PATCHED_POLARS_PUSHDOWN.md.

    The price is `LazyFrame._scan_python_function` rather than the public
    wrapper. That is the same trade the provider path already makes, and this
    reproduces what `register_io_source` does around it -- deserialize the
    predicate, tolerate a predicate that will not deserialize -- rather than
    relying on any of it.
    """
    schema = spec.polars_schema()

    def wrap(
        with_columns: list[str] | None,
        predicate: bytes | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> tuple[Iterator[pl.DataFrame], bool]:
        parsed: pl.Expr | None = None
        if predicate:
            try:
                parsed = pl.Expr.deserialize(predicate)
            except Exception as exc:
                warnings.warn(
                    "polars-pylance: could not read the predicate Polars passed "
                    f"({exc!r}); filtering falls back to the engine",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # False: the engine evaluates `parsed` above whatever is yielded, so the
        # SQL filter is only ever an IO hint. That is what makes a relaxed
        # lowering sound, and what lets a filter Lance refuses be dropped.
        return _io_plugin_frames(
            spec, schema, with_columns, parsed, n_rows, batch_size
        ), False

    return pl.LazyFrame._scan_python_function(
        schema, wrap, pyarrow=False, validate_schema=False, is_pure=True
    )


def _io_plugin_frames(
    spec: LanceScanSpec,
    schema: pl.Schema,
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
    batch_size: int | None,
) -> Iterator[pl.DataFrame]:
    dataset = spec.open()
    lowered = (
        to_lance_filter(predicate, schema=schema)
        if predicate is not None and spec.predicate_pushdown
        else None
    )
    # A limit is only pushed when there is no predicate at all: applied to rows
    # that still have to be filtered it would truncate the wrong ones. Polars
    # does not offer a limit together with a predicate anyway.
    limit = n_rows if predicate is None else None
    # The engine's batch-size hint is sized in rows with no idea how wide they
    # are, and on a 256-byte payload it is 4x the option's default. It is used
    # only where the option asked for Lance's own choice.
    scan_spec = (
        _with_batch_size(spec, batch_size)
        if batch_size is not None and spec.options.batch_size is None
        else spec
    )

    remaining = n_rows if predicate is None else None
    for frame in scan_spec.iter_frames(
        dataset,
        projection=with_columns,
        filter=lowered.sql if lowered is not None else None,
        limit=limit,
        filter_is_optional=True,
    ):
        out = frame
        if remaining is not None:
            if out.height > remaining:
                out = out.head(remaining)
            remaining -= out.height
        if out.height:
            yield out
        if remaining is not None and remaining <= 0:
            return


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
    impl: Literal["provider", "io_plugin"] = "provider",
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
        Reader tuning; see :class:`~polars_pylance.LanceScanOptions`. The defaults
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
        :func:`~polars_pylance.scan_lance_fragments` for the sharded form.
    predicate_pushdown
        Set to False to keep filtering entirely in Polars. Both scan paths
        re-apply the predicate after Lance, so this only ever costs speed.
    impl
        Which Polars hook to scan through, and with it how much of a filter
        reaches Lance. ``"provider"`` uses the private dataset hook, whose
        filter comes pre-lowered by Polars into a PyArrow expression --
        comparisons, boolean structure and null checks only. ``"io_plugin"``
        uses the public IO-plugin API, which hands over the whole predicate for
        :mod:`polars_pylance._predicate` to lower into Lance SQL: string
        matching, ``is_in``, arithmetic, temporal parts, list and struct access
        all reach the scanner. Use it when a filter Polars cannot lower is what
        makes the query expensive; see ``docs/PREDICATE_PUSHDOWN.md``.

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

    if impl == "provider":
        return _provider_lazyframe(spec)
    if impl == "io_plugin":
        return _io_plugin_lazyframe(spec)
    msg = f"unknown scan impl {impl!r}; expected 'provider' or 'io_plugin'"
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
