"""Lazy, streaming Lance reader for Polars.

The scan is a Polars IO plugin. :func:`polars.io.plugins.register_io_source`
hands the source the projection, the row limit, a batch-size hint, and the whole
predicate as a :class:`polars.Expr`; :mod:`polars_pylance._predicate` translates
that into a Lance SQL filter, which is the widest filter language Lance accepts.

That is why this is a plugin rather than the private
``PyLazyFrame.new_from_dataset_object`` hook behind ``scan_delta``. The hook
offers a predicate Polars has already lowered for PyArrow, and drops everything
that language cannot say, which is most of it.

Polars considers the predicate handled once a plugin has been given it, so this
module makes that true: an exact lowering is left to Lance, and a relaxed one is
finished per batch here.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lance
import polars as pl
import pyarrow as pa

from ._options import LanceScanOptions
from ._predicate import VIRTUAL_COLUMNS, to_lance_filter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# `VIRTUAL_COLUMNS` is imported rather than defined here because the predicate
# lowering has to refuse the same columns.


class _FilterRejected(RuntimeError):
    """Lance refused to plan a scan with the filter we pushed.

    Raised only from the first pull, where Lance validates a filter and before
    any batch exists, so the caller can restart the scan without it.
    """


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
    # Already lowered to Lance SQL by `scan_lance`, so the spec stays a plain
    # picklable record and an unsupported prefilter fails at the call site.
    prefilter: str | None = None
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
        filter: str | None = None,
        limit: int | None = None,
        prefilter: bool = False,
    ) -> lance.LanceScanner:
        kwargs: dict[str, Any] = {
            **self.options.to_scan_kwargs(),
            "columns": columns,
            "filter": filter,
            "limit": limit,
        }
        if prefilter:
            # Lance has one filter slot; this says it restricts the rows the
            # search runs over instead of filtering the search's result.
            kwargs["prefilter"] = True
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
        empty = pl.from_arrow(arrow.empty_table())
        assert isinstance(empty, pl.DataFrame)
        return empty.schema

    # -- batch production --------------------------------------------------

    def iter_frames(
        self,
        dataset: lance.LanceDataset,
        *,
        projection: Sequence[str] | None = None,
        filter: str | None = None,
        limit: int | None = None,
        prefilter: bool = False,
    ) -> Iterator[pl.DataFrame]:
        """Stream `projection` out of Lance as Polars frames.

        Lazy by construction: nothing is read until the consumer pulls, and
        dropping the generator early stops the scan.
        """
        columns = None if projection is None else self._physical_columns(projection)
        scanner = self.scanner(
            dataset, columns=columns, filter=filter, limit=limit, prefilter=prefilter
        )

        remaining = limit
        # A pushed-down predicate may be dropped and the scan retried without
        # it, because Polars can finish it. A prefilter has no such second
        # chance: dropping it would silently widen what the search ranked.
        for batch in self._batches(
            scanner, fallback=filter is not None and not prefilter
        ):
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

            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            if projection is not None and frame.columns != list(projection):
                # Lance appends generated columns after the requested ones; the
                # engine expects exactly the projection, in order.
                frame = frame.select(projection)
            yield frame

    @staticmethod
    def _batches(
        scanner: lance.LanceScanner, *, fallback: bool
    ) -> Iterator[pa.RecordBatch]:
        """`scanner.to_batches()`, reporting a rejected filter as such.

        Lance validates a filter while planning, on the first pull, so telling
        that failure from any other lets the caller retry without it. Only when
        `fallback` says a retry would still give the same rows; otherwise the
        error is Lance's own and is left to reach the caller.
        """
        iterator = iter(scanner.to_batches())
        try:
            first = next(iterator, None)
        except Exception as exc:
            if not fallback:
                raise
            raise _FilterRejected(str(exc)) from exc
        if first is None:
            return
        yield first
        yield from iterator

    def _physical_columns(self, projection: Sequence[str]) -> list[str]:
        # Generated columns are added by the scanner itself and must not appear
        # in `columns=`; an empty result is legal and means "generated only".
        return [c for c in projection if c not in VIRTUAL_COLUMNS]


# ---------------------------------------------------------------------------
# the IO plugin
# ---------------------------------------------------------------------------


def _io_plugin_lazyframe(spec: LanceScanSpec) -> pl.LazyFrame:
    # Imported by path: `polars` does not re-export its `io` subpackage.
    from polars.io.plugins import register_io_source

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        return _frames(spec, with_columns, predicate, n_rows, batch_size)

    # `schema` as a callable keeps `scan_lance()` from touching the dataset:
    # Polars asks for it when the query is resolved, not when it is built.
    return register_io_source(
        source, schema=spec.polars_schema, validate_schema=False, is_pure=True
    )


def _frames(
    spec: LanceScanSpec,
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
    batch_size: int | None,
) -> Iterator[pl.DataFrame]:
    """Produce the batches for one resolved scan.

    Restarts without the pushed filter if Lance refuses to plan it.
    """
    dataset = spec.open()
    # The engine's batch-size hint is sized in rows with no idea how wide they
    # are; it is used only where the option asked for Lance's own choice.
    if batch_size is not None and spec.options.batch_size is None:
        spec = dataclasses.replace(
            spec, options=spec.options.replace(batch_size=batch_size)
        )

    sql: str | None
    if spec.prefilter is not None:
        # Lance's one filter slot is spoken for. The prefilter decides which
        # rows the search ranks; the query's own `.filter()` therefore stays in
        # Polars, where it is a postfilter over that ranking. `predicate_pushdown`
        # governs the automatic lowering, not this explicit argument.
        sql, residual, prefilter = spec.prefilter, predicate, True
    else:
        lowered = (
            to_lance_filter(predicate, schema=spec.polars_schema(dataset))
            if predicate is not None and spec.predicate_pushdown
            else None
        )
        # An exact lowering keeps the same rows, so Lance can be left to it. A
        # relaxed one keeps more, and nothing downstream will filter again.
        residual = None if lowered is not None and lowered.exact else predicate
        sql = lowered.sql if lowered is not None else None
        prefilter = False

    batches = _apply(
        spec, dataset, with_columns, sql, residual, n_rows, prefilter=prefilter
    )
    try:
        first = next(batches)
    except StopIteration:
        return
    except _FilterRejected as exc:
        # Only reachable for a lowered predicate: `iter_frames` lets a rejected
        # prefilter raise Lance's own error instead.
        warnings.warn(
            f"polars-pylance: Lance rejected the pushed-down filter ({exc}); "
            "scanning without it",
            RuntimeWarning,
            stacklevel=2,
        )
        yield from _apply(spec, dataset, with_columns, None, predicate, n_rows)
        return
    yield first
    yield from batches


def _apply(
    spec: LanceScanSpec,
    dataset: lance.LanceDataset,
    projection: list[str] | None,
    sql: str | None,
    residual: pl.Expr | None,
    n_rows: int | None,
    *,
    prefilter: bool = False,
) -> Iterator[pl.DataFrame]:
    """Stream the scan, evaluating `residual` on each batch if there is one."""
    if residual is None:
        # Nothing is left to filter afterwards, so the row limit can go down
        # too: it truncates the ranked result rather than narrowing the search.
        yield from spec.iter_frames(
            dataset,
            projection=projection,
            filter=sql,
            limit=n_rows,
            prefilter=prefilter,
        )
        return

    # The limit is counted off on rows that survived `residual`, never pushed
    # past a filter that has not run yet.
    columns = _with_predicate_columns(projection, residual)
    remaining = n_rows
    for frame in spec.iter_frames(
        dataset, projection=columns, filter=sql, prefilter=prefilter
    ):
        out = frame.filter(residual)
        if projection is not None and out.columns != projection:
            out = out.select(projection)
        if remaining is not None:
            if out.height > remaining:
                out = out.head(remaining)
            remaining -= out.height
        if out.height:
            yield out
        if remaining is not None and remaining <= 0:
            return


def _with_predicate_columns(
    projection: list[str] | None, predicate: pl.Expr
) -> list[str] | None:
    """`projection`, widened to what the predicate still has to read.

    Polars already projects the filter's columns, but only because it knows the
    filter runs at the scan.
    """
    if projection is None:
        return None
    extra = [c for c in predicate.meta.root_names() if c not in projection]
    return projection + extra if extra else projection


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _prefilter_sql(prefilter: str | pl.Expr) -> str:
    """Lower an explicit prefilter, refusing whatever Lance cannot decide alone.

    A pushed-down predicate is allowed to be relaxed, because Polars still
    evaluates it afterwards. A prefilter chooses which rows the search ranks at
    all, and nothing downstream can repair that choice -- so a partial lowering
    is an error here rather than a silent demotion to a postfilter.
    """
    if isinstance(prefilter, str):
        return prefilter
    lowered = to_lance_filter(prefilter)
    if lowered is None:
        msg = (
            f"prefilter does not translate to a Lance filter: {prefilter}. "
            "It is not applied as a postfilter, because filtering the ranked "
            "result is a different question; rewrite it, or pass the Lance SQL "
            "as a string."
        )
        raise ValueError(msg)
    if not lowered.exact:
        msg = (
            f"prefilter only partly translates to a Lance filter: {prefilter} "
            f"lowers to {lowered.sql!r}, which keeps a superset of its rows. A "
            "prefilter decides what the search ranks, so pushing the wider one "
            "would silently change the result; pass the Lance SQL as a string "
            "to say exactly what should be pushed."
        )
        raise ValueError(msg)
    return lowered.sql


def scan_lance(
    source: str | Path | lance.LanceDataset,
    *,
    version: int | str | None = None,
    storage_options: dict[str, str] | None = None,
    options: LanceScanOptions | None = None,
    nearest: dict[str, Any] | None = None,
    full_text_query: str | dict[str, Any] | None = None,
    prefilter: str | pl.Expr | None = None,
    with_row_id: bool = False,
    with_row_address: bool = False,
    fragments: Sequence[int] | None = None,
    predicate_pushdown: bool = True,
) -> pl.LazyFrame:
    """Lazily read a Lance dataset as a Polars :class:`~polars.LazyFrame`.

    Nothing is read when this returns. Column projections, filters and row
    limits are pushed into Lance; batches are pulled only as the query consumes
    them, so ``.head()`` stops the scan early. Use ``engine="streaming"`` when
    collecting: the in-memory engine materialises the whole result and gives up
    the memory advantage.

    The filter becomes a Lance SQL filter, so ``is_in``, string functions,
    arithmetic, temporal parts and list or struct access reach the scanner, none
    of which a PyArrow expression can carry. A predicate that only partly
    translates is pushed as far as it goes and finished in Polars.
    :func:`~polars_pylance.to_lance_filter` shows what a given one lowers to.

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
    prefilter
        Restrict which rows ``nearest`` or ``full_text_query`` may return before
        the search runs, as Lance SQL or a Polars expression. This is not the
        same question as ``.filter()`` on the result, which ranks first and
        filters after and so may return fewer than ``k`` rows; with a prefilter
        the search picks its ``k`` from the surviving rows only. A Polars
        expression that does not translate exactly is an error rather than a
        postfilter, since nothing downstream can repair a candidate set the
        search has already used.
    with_row_id, with_row_address
        Include Lance's ``_rowid`` / ``_rowaddr`` columns.
    fragments
        Restrict the scan to these fragment ids. See
        :func:`~polars_pylance.scan_lance_fragments` for the sharded form.
    predicate_pushdown
        Set to False to keep filtering entirely in Polars. Worth trying if you
        depend on Polars' null comparison semantics, which differ from SQL's.

    Examples
    --------
    >>> lf = scan_lance("s3://bucket/data.lance")  # doctest: +SKIP
    >>> lf.filter(pl.col("label").is_in([3, 7])).select("id", "score").collect(
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
        prefilter=None if prefilter is None else _prefilter_sql(prefilter),
        with_row_id=with_row_id,
        with_row_address=with_row_address,
        fragment_ids=list(fragments) if fragments is not None else None,
        predicate_pushdown=predicate_pushdown,
    )

    return _io_plugin_lazyframe(spec)


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
