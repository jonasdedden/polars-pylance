"""Helpers for running Lance scans and writes on Polars Cloud.

Install the client with the `cloud` extra (`pip install polars-pylance[cloud]`),
which brings `polars-cloud>=0.11` tracking `polars==1.44.2`. See "The polars pin"
below for why the extra exists rather than a hard dependency.

What works and what does not, as of polars-cloud 0.11:

### Reading

A `scan_lance` plan serializes to ~2-6 kB and can be shipped with
`LazyFrame.remote()`, but the remote workers must be able to `import lance` and to
reach the dataset's storage. Install the dependency with
`ComputeContext(requirements=...)`; see
[`requirements_txt`][polars_pylance.cloud.requirements_txt].

The scan survives `prepare_cloud_plan`, on its own and under `pl.concat` of
[`scan_lance_fragments`][polars_pylance.scan_lance_fragments] shards. 0.9 added
distributed unions of Python scans, so the sharded form is the sanctioned way to fan
a read out across workers rather than a fallback.

### Writing

Possible remotely since 0.10, via
[`sink_lance_remote`][polars_pylance.cloud.sink_lance_remote]. Polars Cloud's native
sink destinations are still Parquet, CSV, IPC and Iceberg, but `sink_batches` hands
each result batch to a Python callable that is cloudpickled into the query plan and
therefore runs *on the workers*, so the workers write Lance data files directly, and
a single client-side commit publishes them. `polars_pylance._remote` documents the
arrangement.

### The polars pin

polars-cloud pins polars with `==` (0.11 tracks `polars==1.44.2`, 0.10 required
`polars==1.43.2`), and this package requires `polars>=1.44.1`. The `cloud` extra
is an extra rather than a hard dependency so a plain `pip install polars-pylance`
stays usable without a Cloud workspace; installing the extra resolves both to a
1.44 line that supports the IO-plugin hook `scan_lance` is built on.

The floor is there because 1.43.2 is the last release in which a `sort().head()`
pushes an unevaluable `dynamic_pred` node into an IO plugin's predicate, which is
exactly what `scan_lance` is. 1.44.0 fixed that but was yanked, so 1.44.1 is the
first usable release.
"""

from __future__ import annotations

import lance
import polars as pl

from ._remote import (
    StagedLanceSink,
    sink_lance_remote,
    stage_lance_sink,
)

__all__ = [
    "StagedLanceSink",
    "requirements_txt",
    "sink_lance_remote",
    "stage_lance_sink",
]


def requirements_txt(extra: list[str] | None = None) -> str:
    """Render a requirements file pinning the versions a cloud worker needs.

    Polars Cloud rejects a compute context whose polars version differs from the
    client's, so both pins are exact. `polars-pylance` itself is on the list because a
    [`sink_lance_remote`][polars_pylance.cloud.sink_lance_remote] callback is pickled by
    reference: the worker imports it rather than receiving its code.

    Args:
        extra: Further requirement lines to append, for whatever else the query needs on
            the worker.

    Returns:
        str: The file contents, newline-terminated.

    Examples:
        >>> import polars_cloud as pc  # doctest: +SKIP
        >>> ctx = pc.ComputeContext(
        ...     cpus=8, memory=32, requirements=requirements_txt().encode()
        ... )  # doctest: +SKIP
    """
    lines = [
        f"polars=={pl.__version__}",
        f"pylance=={lance.__version__}",
        "polars-pylance",
    ]
    lines.extend(extra or [])
    return "\n".join(lines) + "\n"
