"""The five benchmark queries and the registry that names them.

Each case is a filter plus a projection over a Narwhals frame, so the same
definition runs on Polars, Daft and Arrow. `CASES` maps the short names the
driver takes on the command line to their query plus the metadata the backends
need around it. The pushed Lance SQL is translated from that same filter --
never spelled twice -- so the two cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import narwhals as nw
import polars as pl

from polars_pylance import to_lance_filter

if TYPE_CHECKING:
    from collections.abc import Callable

    import narwhals.typing as nwt

#: Rows per batch handed to a writer, unless the driver overrides it.
DEFAULT_CHUNK_SIZE = 25_000
#: Relative tolerance for float checksums. The engines may differ in the
#: last ULP of a transcendental, and those differences accumulate as a
#: random walk over millions of rows -- still orders of magnitude inside
#: this, as is the reordering from different batch sizes on one engine.
FLOAT_TOL = 1e-9


def _extract_polars(predicate: nw.Expr) -> pl.Expr:
    """Lower a Narwhals predicate to Polars through narwhals' own machinery.

    This is the same step `LazyFrame.filter` performs on every predicate: the
    expression nodes evaluated against the Polars namespace, whose compliant
    expression wraps the native `pl.Expr`. Translating that -- what the Polars
    engine actually executes -- rather than a parallel spelling keeps the pushed
    filter faithful by construction, whatever shape a future predicate takes.

    It reaches through narwhals' private compliant layer, which has no stable
    equivalent: the public translators convert frames and series only. The
    import is deferred so a future narwhals that moves these names breaks the
    pushdown path with a clear error, not the whole package at import. The two
    markers below name that deliberate use: SLF001 is exempted per file (the
    whole file is the query registry, so the exemption stays close), and the
    `type: ignore` expires on its own if narwhals ever types the accessor.
    """
    try:
        from narwhals._polars.namespace import PolarsNamespace
        from narwhals._utils import Version
    except ImportError as exc:
        msg = (
            "narwhals moved its compliant layer; _extract_polars needs the "
            "new location of the Polars namespace and Version"
        )
        raise SystemExit(msg) from exc
    ns = PolarsNamespace(version=Version.MAIN)
    lowered = predicate._to_compliant_expr(ns)  # pyright: ignore[reportPrivateUsage]
    native: pl.Expr = lowered.native  # type: ignore[attr-defined]
    return native


#: Keeps half the rows; shared by the reduction and the compute-heavy write.
_HALF: nw.Expr = nw.col("val") > 0.5
#: Keeps a tenth of the rows.
_TENTH: nw.Expr = nw.col("val") > 0.9
#: Keeps a hundredth of the rows.
_HUNDREDTH: nw.Expr = nw.col("val") > 0.99


def _heavy(value: nw.Expr) -> nw.Expr:
    """A transform expensive enough to separate compute from IO scaling."""
    return (
        (value * 6.283185307179586).sin()
        + (value * 12.566370614359172).cos()
        + (value + 1.0).sqrt()
    )


def _r_agg(frame: nwt.Frame) -> nwt.Frame:
    """One row per shard: the surviving count and the exact id checksum."""
    return frame.select(nw.len().alias("n"), nw.col("id").sum().alias("s_id"))


def _w_filter(frame: nwt.Frame) -> nwt.Frame:
    return frame.select(
        "id", "cat", "val", "payload", (nw.col("val") * 2).alias("val2")
    )


def _w_compute(frame: nwt.Frame) -> nwt.Frame:
    return frame.select("id", "cat", "val", "payload", _heavy(nw.col("val")).alias("w"))


def _w_full(frame: nwt.Frame) -> nwt.Frame:
    return frame.select("id", "cat", "val", "text", "ts", "payload")


def _w_commit(frame: nwt.Frame) -> nwt.Frame:
    return frame.select("id", "val")


CaseKind = Literal["read", "write"]


class CaseName(str, Enum):
    """Every benchmark query, by the short name the driver takes."""

    R_AGG = "r_agg"
    W_FILTER = "w_filter"
    W_COMPUTE = "w_compute"
    W_FULL = "w_full"
    W_COMMIT = "w_commit"


#: A read case returns a reduction; used to overload backends precisely.
ReadCaseName = Literal[CaseName.R_AGG]
#: Any write case returns row/fragment counts; one overload covers all four.
WriteCaseName = Literal[
    CaseName.W_FILTER, CaseName.W_COMPUTE, CaseName.W_FULL, CaseName.W_COMMIT
]


#: The ladder schema, mirroring `bench/polars_lance/gen.py` SCHEMA. The pushed
#: filter is translated against it so the SQL matches what `scan_lance` emits
#: for the same predicate -- and stays eligible for scalar indices, which a
#: `CAST` around the column would cost.
_LADDER_FIELDS: dict[str, pl.DataType | type[pl.DataType]] = {
    "id": pl.Int64,
    "cat": pl.String,
    "val": pl.Float64,
    "text": pl.String,
    "ts": pl.Datetime("us"),
    "payload": pl.Binary,
}
_LADDER_SCHEMA: pl.Schema = pl.Schema(_LADDER_FIELDS)


def _pushdown_sql(predicate: nw.Expr | None) -> str:
    """The Lance SQL Ray Data pushes for `predicate`, or "" without one.

        Translated by the package's own pushdown from the lowered Polars spelling,
    never spelled by hand, so the pushed filter says exactly what the query's
    filter says. Anything less than an exact translation would silently change
    what the benchmark measures, so it fails fast instead.
    """
    if predicate is None:
        return ""
    translated = to_lance_filter(_extract_polars(predicate), schema=_LADDER_SCHEMA)
    if translated is None or not translated.exact:
        msg = f"cannot push case filter {predicate!r} exactly to Lance"
        raise SystemExit(msg)
    return translated.sql


@dataclass(frozen=True)
class Case:
    """One benchmark query, written once, run on every engine.

    A case is an optional filter followed by a projection: `predicate` states
    the filter once, `select` maps a filtered frame to its result -- the rows
    to write for a write case, the one-row reduction for a read case. `query`
    joins the two, so the filter cannot say one thing in the query and another
    in the pushdown. Each backend applies it to its own native frame
    (`apply_polars`, `apply_daft`, `apply_arrow`) and unwraps the result.
    `columns` is the scan projection Ray Data reads -- input columns, not the
    query's output, and the two differ whenever the query computes (`val2`,
    `w`) or aggregates (`n`, `s_id`). Read cases carry no output schema
    because they write no dataset.
    """

    kind: CaseKind
    blurb: str
    predicate: nw.Expr | None
    select: Callable[[nwt.Frame], nwt.Frame]
    columns: list[str] = field(default_factory=list)

    @property
    def query(self) -> Callable[[nwt.Frame], nwt.Frame]:
        """The full query: the predicate filter, then the projection."""
        predicate = self.predicate
        select = self.select

        def run(frame: nwt.Frame) -> nwt.Frame:
            if predicate is not None:
                frame = frame.filter(predicate)
            return select(frame)

        return run

    @property
    def lance_filter(self) -> str:
        """The SQL Ray Data pushes into its scanner, translated from `predicate`."""
        return _pushdown_sql(self.predicate)


CASES: dict[CaseName, Case] = {
    CaseName.R_AGG: Case(
        kind="read",
        blurb="scan-only reduction: filter, then count and id checksum per shard",
        predicate=_HALF,
        select=_r_agg,
        columns=["id"],
    ),
    CaseName.W_FILTER: Case(
        kind="write",
        blurb="selective pushed-down predicate plus the wide payload column",
        predicate=_TENTH,
        select=_w_filter,
        columns=["id", "cat", "val", "payload"],
    ),
    CaseName.W_COMPUTE: Case(
        kind="write",
        blurb="compute-heavy transform: trig chain separating compute from IO",
        predicate=_HALF,
        select=_w_compute,
        columns=["id", "cat", "val", "payload"],
    ),
    CaseName.W_FULL: Case(
        kind="write",
        blurb="full copy: every column, no predicate, maximum IO pressure",
        predicate=None,
        select=_w_full,
        columns=["id", "cat", "val", "text", "ts", "payload"],
    ),
    CaseName.W_COMMIT: Case(
        kind="write",
        blurb="commit stress: narrow 1% output, fragment count is the variable",
        predicate=_HUNDREDTH,
        select=_w_commit,
        columns=["id", "val"],
    ),
}

#: The commit-stress sweep. Shard counts, not file sizes: each shard writes
#: at least one fragment whatever `max_rows_per_file` says, so at small
#: outputs the file-size knob does nothing and the shard count is what
#: moves the fragment count. `scan_lance_fragments` caps at the dataset's
#: own fragment count, so an oversized entry degrades gracefully.
COMMIT_SHARDS = (4, 16, 64)
