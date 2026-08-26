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
it a second time whatever the scan says, and the two places differ: the
streaming engine, or Python, once per batch. ``io_plugin_register`` is the
``register_io_source`` form the shipped path used to have, kept as a column so
the difference stays measurable. (There is no provider counterpart, because a
provider-resolved scan's ``predicate_applied`` flag is ignored outright --
``bench/hooks.py`` probes for that and reports it.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow.compute as pc


VARIANTS = (
    "provider",
    "provider_pa",
    "provider_sql",
    "io_plugin",
    "io_plugin_register",
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
    if impl == "io_plugin_register":
        return _io_plugin_register_source(uri)
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


def _io_plugin_register_source(uri: str) -> pl.LazyFrame:
    """The IO-plugin path as `register_io_source` builds it.

    That wrapper hardcodes "the source applied the predicate", so a source
    pushing a *relaxed* filter has to re-apply the predicate itself, per batch,
    from Python. The shipped path no longer does this -- it reports the
    predicate as unapplied and lets the streaming engine evaluate it -- and this
    column is what that costs, over the identical pushed filter.
    """
    from polars.io.plugins import register_io_source

    from polars_pylance._predicate import to_lance_filter
    from polars_pylance._scan import LanceScanSpec

    spec = LanceScanSpec(uri=uri)
    schema = spec.polars_schema()

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        lowered = (
            to_lance_filter(predicate, schema=schema) if predicate is not None else None
        )
        frames = spec.iter_frames(
            spec.open(),
            projection=with_columns,
            filter=None if lowered is None else lowered.sql,
            limit=n_rows if predicate is None else None,
            filter_is_optional=True,
        )
        for frame in frames:
            out = frame if predicate is None else frame.filter(predicate)
            if out.height:
                yield out

    return register_io_source(
        source, schema=schema, validate_schema=False, is_pure=True
    )
