"""One Narwhals spelling per case, run on every engine.

- `cases`: the five queries and the registry that names them.
- `adapters`: native frame in, native frame out, per engine.
- `checks`: checksums proving the backends agree.
- `harness`: worker warmup, byte counts, smoke fixtures and the smoke test.
"""

from .adapters import (
    apply_arrow,
    apply_daft,
    apply_polars,
    configure_daft,
    daft_query_frame,
    daft_read,
    daft_totals,
)
from .cases import (
    CASES,
    COMMIT_SHARDS,
    DEFAULT_CHUNK_SIZE,
    Case,
    CaseKind,
    CaseName,
    ReadCaseName,
    WriteCaseName,
)
from .checks import agree, checksum
from .harness import (
    import_dependencies,
    make_smoke_source,
    metadata_bytes,
    plan_bytes,
    smoke,
)

__all__ = [
    "CASES",
    "COMMIT_SHARDS",
    "DEFAULT_CHUNK_SIZE",
    "Case",
    "CaseKind",
    "CaseName",
    "ReadCaseName",
    "WriteCaseName",
    "agree",
    "apply_arrow",
    "apply_daft",
    "apply_polars",
    "checksum",
    "configure_daft",
    "daft_query_frame",
    "daft_read",
    "daft_totals",
    "import_dependencies",
    "make_smoke_source",
    "metadata_bytes",
    "plan_bytes",
    "smoke",
]
