"""Scan tuning knobs.

Lance's own read-ahead defaults, not Polars, dominate the memory footprint of a
streaming scan: `io_buffer_size` alone defaults to 2 GiB. Measured on a 527 MB
single-column scan, the defaults below cut peak RSS from 697 MB to 426 MB at no
measurable cost in wall time, so they are what [`scan_lance`][polars_pylance.scan_lance]
uses unless asked otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

MIB = 1024 * 1024


@dataclass(frozen=True)
class LanceScanOptions:
    """Per-scan Lance reader tuning. Immutable and picklable.

    Every field maps to the identically named argument of
    `lance.LanceDataset.scanner`. `None` means "leave it to Lance".

    Args:
        batch_size: Rows per record batch handed to Polars.
        batch_readahead: Batches decoded ahead of the consumer, per fragment.
        fragment_readahead: Fragments read concurrently.
        io_buffer_size: Size of the Lance IO buffer, in bytes. This is the single
            biggest lever on peak memory; Lance's own default is 2 GiB.
        scan_in_order: Yield fragments in order. Out-of-order scanning is faster but
            lets more data accumulate in flight.
        use_scalar_index: Whether pushed-down predicates may use scalar indices.
        late_materialization: Defer loading of large columns until after filtering.
            Either a bool for all columns or a list of column names.
    """

    batch_size: int | None = 25_000
    batch_readahead: int | None = 1
    fragment_readahead: int | None = 1
    io_buffer_size: int | None = 32 * MIB
    scan_in_order: bool | None = True
    use_scalar_index: bool | None = None
    late_materialization: bool | list[str] | None = None

    @classmethod
    def throughput(cls, **overrides: Any) -> LanceScanOptions:  # noqa: ANN401
        """Restore Lance's own aggressive read-ahead, for when RAM is plentiful.

        Roughly 1.6x the peak memory of the defaults on a large-payload scan, in
        exchange for more IO parallelism.
        """
        return cls(
            batch_size=None,
            batch_readahead=None,
            fragment_readahead=None,
            io_buffer_size=None,
            scan_in_order=None,
            **overrides,
        )

    def replace(self, **overrides: Any) -> LanceScanOptions:  # noqa: ANN401
        """Return a copy with `overrides` applied."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        unknown = set(overrides) - set(current)
        if unknown:
            msg = f"unknown LanceScanOptions fields: {sorted(unknown)}"
            raise TypeError(msg)
        return type(self)(**{**current, **overrides})

    def to_scan_kwargs(self) -> dict[str, Any]:
        """Render as `scanner()` keyword arguments, omitting unset fields."""
        return {
            f.name: value
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }
