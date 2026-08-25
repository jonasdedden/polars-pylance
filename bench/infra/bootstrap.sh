#!/bin/bash
# Build the benchmark environment on the instance-store NVMe.
# Expects the polars-pylance wheel already uploaded to $BENCH_ROOT.
set -eu
BENCH_ROOT=${BENCH_ROOT:-/mnt/nvme}
export HOME=/root UV_CACHE_DIR="$BENCH_ROOT/uv-cache"
cd "$BENCH_ROOT"

uv venv -p 3.12 "$BENCH_ROOT/venv"
uv pip install -q --python "$BENCH_ROOT/venv/bin/python" \
    "polars==1.44.0" "pylance==10.0.0" pyarrow numpy \
    polars-lance "$BENCH_ROOT"/polars_pylance-*.whl

"$BENCH_ROOT/venv/bin/python" - <<'PY'
import polars, lance, importlib.metadata as md
print("polars", polars.__version__, "| pylance", lance.__version__,
      "| polars-lance", md.version("polars-lance"),
      "| polars-pylance", md.version("polars-pylance"))
PY
nproc; free -g | head -2; df -h "$BENCH_ROOT" | tail -1
