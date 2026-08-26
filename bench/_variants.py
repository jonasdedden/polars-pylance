"""Scan-path variants, for isolating one mechanism at a time.

``bench/pushdown.py`` compares scan paths; this module is where the paths that
are *not* shipped live. Each variant changes exactly one thing about
``polars_pylance``, so the difference between two columns of the matrix names a
mechanism rather than a bundle of them.

Everything here monkey-patches the installed package, which is safe because
``pushdown.py`` runs every measurement in its own process.

The variants split into two families.

**Which lowering the provider path uses.** On a Polars carrying
`pola-rs/polars#28995 <https://github.com/pola-rs/polars/pull/28995>`_ the
provider hook is handed two descriptions of the same predicate: Polars'
``pyarrow_predicate`` (a PyArrow expression, widened by #28994 and #28996) and
``serialized_predicate`` (the whole thing, which this package lowers to Lance
SQL). ``provider`` picks between them; ``provider_pa`` and ``provider_sql`` pin
one, which is how the two upstream changes are told apart.

**Where the predicate is re-applied after Lance has answered.** Polars evaluates
it a second time whatever the scan says, and the hooks differ in where: the
streaming engine, or Python, once per batch. ``io_plugin_rust`` moves it from
the second to the first. (There is no provider counterpart, because a
provider-resolved scan's ``predicate_applied`` flag is ignored outright --
``bench/hooks.py`` probes for that and reports it.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow.compute as pc

    from polars_pylance._predicate import LanceFilter
    from polars_pylance._scan import LanceScanSpec

VARIANTS = (
    "provider",
    "provider_pa",
    "provider_sql",
    "io_plugin",
    "io_plugin_rust",
    "io_plugin_hint",
    "engine",
)


def build(impl: str, uri: str) -> pl.LazyFrame:
    """A LazyFrame over `uri` scanning through the path named by `impl`."""
    from polars_pylance import LanceScanOptions, scan_lance

    if impl in ("provider", "io_plugin"):
        return scan_lance(uri, impl=impl)
    if impl == "engine":
        # Same hook as `io_plugin`, lowering nothing: the difference to
        # `io_plugin` is the visitor's contribution, and the difference to
        # `provider` is the hook's.
        return scan_lance(uri, impl="io_plugin", predicate_pushdown=False)
    if impl == "io_plugin_hint":
        # Only the IO plugin is handed the engine's batch-size hint (100k rows),
        # and it takes it only when the scan options do not pin one.
        return scan_lance(
            uri, impl="io_plugin", options=LanceScanOptions(batch_size=None)
        )
    if impl in ("provider_pa", "provider_sql"):
        _pin_provider_lowering(impl)
        return scan_lance(uri, impl="provider")
    if impl == "io_plugin_rust":
        return _io_plugin_engine_filter(uri)
    msg = f"unknown impl {impl!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# which lowering the provider path uses
# ---------------------------------------------------------------------------


def _pin_provider_lowering(impl: str) -> None:
    """Make the provider path use only one of the two predicates Polars passes.

    ``LanceDatasetProvider._filter`` normally prefers an exact Lance SQL
    lowering and falls back to Polars' PyArrow one. Pinning it isolates what
    each upstream change contributes: ``provider_pa`` is what #28994 and #28996
    buy on their own, ``provider_sql`` is what #28995 buys on its own.
    """
    from polars_pylance import _scan

    original = _scan.LanceDatasetProvider._filter

    def only_pyarrow(
        self: Any,
        dataset: Any,
        pyarrow_predicate: str | None,
        serialized_predicate: bytes | None,
    ) -> pc.Expression | str | None:
        return original(self, dataset, pyarrow_predicate, None)

    def only_sql(
        self: Any,
        dataset: Any,
        pyarrow_predicate: str | None,
        serialized_predicate: bytes | None,
    ) -> pc.Expression | str | None:
        return original(self, dataset, None, serialized_predicate)

    _scan.LanceDatasetProvider._filter = (  # type: ignore[method-assign]
        only_pyarrow if impl == "provider_pa" else only_sql
    )


# ---------------------------------------------------------------------------
# an IO plugin that lets the engine re-apply the predicate
# ---------------------------------------------------------------------------


def _io_plugin_engine_filter(uri: str) -> pl.LazyFrame:
    """``impl="io_plugin"``, but with the second evaluation moved into Rust.

    ``polars.io.plugins.register_io_source`` hardcodes "the source applied the
    predicate", so a source that pushes a *relaxed* filter has to re-apply the
    predicate itself -- which it can only do per batch, in Python, once per
    morsel. Calling ``_scan_python_function`` directly and reporting ``False``
    instead hands that job to the streaming engine, which evaluates it over the
    same batch inside the scan node. Same rows, same pushed filter; the only
    difference is which side of the FFI boundary the second evaluation happens
    on.
    """
    from polars_pylance._predicate import to_lance_filter
    from polars_pylance._scan import LanceScanSpec

    spec = LanceScanSpec(uri=uri)
    schema = spec.polars_schema()

    def wrap(
        with_columns: list[str] | None,
        predicate: bytes | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> tuple[Iterator[pl.DataFrame], bool]:
        parsed = pl.Expr.deserialize(predicate) if predicate else None
        lowered = to_lance_filter(parsed) if parsed is not None else None
        frames = _frames(
            spec, with_columns, lowered, n_rows if parsed is None else None
        )
        # False: the engine re-applies `parsed` above the batches we yield.
        return frames, False

    return pl.LazyFrame._scan_python_function(
        schema, wrap, pyarrow=False, validate_schema=False, is_pure=True
    )


def _frames(
    spec: LanceScanSpec,
    with_columns: list[str] | None,
    lowered: LanceFilter | None,
    limit: int | None,
) -> Iterator[pl.DataFrame]:
    yield from spec.iter_frames(
        spec.open(),
        projection=with_columns,
        filter=None if lowered is None else lowered.sql,
        limit=limit,
        filter_is_optional=True,
    )
