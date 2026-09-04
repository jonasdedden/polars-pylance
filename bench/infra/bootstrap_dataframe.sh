#!/bin/bash
# Build the dataframe-benchmark environment on the instance-store NVMe.
# Shares $BENCH_ROOT/venv with bootstrap.sh (creates it if absent); the base
# pins match it so plan-shipping sees the same Polars IR either way, plus the
# dataframe group both local backends need (Ray, Daft, Dask, lance-ray,
# Narwhals and the trig-exprs plugin branch).
set -eu
BENCH_ROOT=${BENCH_ROOT:-/mnt/nvme}
export HOME=/root UV_CACHE_DIR="$BENCH_ROOT/uv-cache"
cd "$BENCH_ROOT"

if [ ! -x "$BENCH_ROOT/venv/bin/python" ]; then
    uv venv -p 3.12 "$BENCH_ROOT/venv"
fi
uv pip install -q --python "$BENCH_ROOT/venv/bin/python" \
    "polars==1.44.1" "pylance==10.0.0" pyarrow numpy \
    "$BENCH_ROOT"/polars_pylance-*.whl \
    "ray>=2.40" "lance-ray>=0.4" "dask[distributed]>=2024.8" "daft[lance]>=0.6" \
    "narwhals>=2.10" \
    "narwhals-daft @ git+https://github.com/jonasdedden/narwhals-daft@trig-exprs"

"$BENCH_ROOT/venv/bin/python" - <<'PY'
import polars, lance, ray, daft, dask, narwhals
print("polars", polars.__version__, "| pylance", lance.__version__,
      "| ray", ray.__version__, "| daft", daft.__version__,
      "| dask", dask.__version__, "| narwhals", narwhals.__version__)
PY
nproc; free -g | head -2; df -h "$BENCH_ROOT" | tail -1
