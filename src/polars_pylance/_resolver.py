"""Opt-in adapter for Polars' unstable LazyFrame resolver protocol.

Imported only when requested, so supported stable Polars releases do not need
this API. The TYPE_CHECKING declarations mirror the API added by Polars #29003;
all runtime classes come from Polars itself.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

import polars as pl

from ._predicate import to_lance_filter
from ._scan import (
    LanceScanSpec,
    _execute_scan,  # pyright: ignore[reportPrivateUsage]
    _ScanPlan,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    @dataclass
    class FilterExpr:
        expr: pl.Expr

    @dataclass(kw_only=True)
    class ResolvedLazyFrameProps:
        version_key: str | None = None
        applied_filters: set[int] = dataclasses.field(default_factory=set)

    class LazyFrameResolver:
        def lazy(self) -> pl.LazyFrame: ...

else:
    try:
        _api = import_module("polars.lazyframe.resolver")
    except ModuleNotFoundError as exc:
        if exc.name != "polars.lazyframe.resolver":
            raise
        _api = None
    LazyFrameResolver = object if _api is None else _api.LazyFrameResolver
    if _api is not None:
        FilterExpr = _api.FilterExpr
        ResolvedLazyFrameProps = _api.ResolvedLazyFrameProps


def _conjunction(expressions: list[pl.Expr]) -> pl.Expr | None:
    if not expressions:
        return None
    result = expressions[0]
    for expr in expressions[1:]:
        result = result & expr
    return result


class LanceResolver(LazyFrameResolver):
    """Resolve latest-version scans to reproducible, serializable readers."""

    def __init__(self, spec: LanceScanSpec) -> None:
        if not TYPE_CHECKING and _api is None:
            msg = (
                "backend='resolver' requires a Polars build with LazyFrameResolver "
                "(pola-rs/polars#29003); use backend='io' on stable Polars"
            )
            raise ImportError(msg)
        self.spec = spec
        self._schema: pl.Schema | None = None

    def schema(self) -> pl.Schema:
        """Remember the query's declared schema without retaining a dataset."""
        if self._schema is None:
            self._schema = self.spec.polars_schema()
        return self._schema

    def cse_eq(self, other: object) -> bool:
        """Do not share evaluations of independently resolved snapshots."""
        del other
        return False

    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],
        filter_columns: list[str],
        filter_drop_columns_idx: int | None,
        existing_resolved_version_key: str | None,
    ) -> tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        """Pin a snapshot and acknowledge only filters guaranteed by the reader."""
        del filter_columns
        dataset = self.spec.open()
        schema = self.spec.polars_schema(dataset)
        if schema != self.schema():
            msg = "Lance schema changed; construct a new scan_lance query"
            raise pl.exceptions.SchemaError(msg)
        version_key = str(dataset.version)
        if existing_resolved_version_key == version_key:
            return None, ResolvedLazyFrameProps(version_key=version_key)

        spec = dataclasses.replace(self.spec, version=dataset.version)
        plan, applied = _resolve_filters(spec, [item.expr for item in filters], schema)
        lf = _planned_source(plan, schema)
        if projection is not None:
            columns = list(projection[:filter_drop_columns_idx])
            for i, item in enumerate(filters):
                if i not in applied:
                    for name in item.expr.meta.root_names():
                        if name not in columns:
                            columns.append(name)
            lf = lf.select(columns) if columns else lf.drop("*")
        # The protocol exposes an upper bound, never a slice offset. Polars
        # reapplies the original slice. Never move a limit ahead of a residual.
        if limit is not None and not filters:
            lf = lf.head(limit)
        return lf, ResolvedLazyFrameProps(
            version_key=version_key, applied_filters=applied
        )


def _resolve_filters(
    spec: LanceScanSpec, filters: list[pl.Expr], schema: pl.Schema
) -> tuple[_ScanPlan, set[int]]:
    sql_parts: list[str] = []
    applied: set[int] = set()
    exact: list[pl.Expr] = []
    # Search filters remain postfilters. Explicit prefilters keep their
    # dedicated slot and are never weakened by automatic translation.
    can_push = (
        spec.predicate_pushdown
        and spec.prefilter is None
        and spec.nearest is None
        and spec.full_text_query is None
    )
    for i, item in enumerate(filters):
        lowered = to_lance_filter(item, schema=schema) if can_push else None
        if lowered is not None:
            sql_parts.append(f"({lowered.sql})")
            if lowered.exact:
                applied.add(i)
                exact.append(item)

    plan = _ScanPlan(
        spec=spec,
        sql=(
            spec.prefilter
            if spec.prefilter is not None
            else (" AND ".join(sql_parts) or None)
        ),
        residual=None,
        predicate=_conjunction(exact),
        prefilter=spec.prefilter is not None,
    )
    return plan, applied


def _planned_source(plan: _ScanPlan, schema: pl.Schema) -> pl.LazyFrame:
    from polars.io.plugins import register_io_source

    def source(
        projection: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # Reoptimization can push residual filters into this child IO plugin.
        # Fulfil its contract locally, including on a rejected SQL retry.
        fallback = _conjunction(
            [expr for expr in (plan.predicate, predicate) if expr is not None]
        )
        spec = plan.spec
        if batch_size is not None and spec.options.batch_size is None:
            spec = dataclasses.replace(
                spec, options=spec.options.replace(batch_size=batch_size)
            )
        execution = dataclasses.replace(
            plan, spec=spec, residual=predicate, predicate=fallback
        )
        yield from _execute_scan(execution, spec.open(), projection, n_rows)

    return register_io_source(source, schema=schema, validate_schema=True, is_pure=True)
