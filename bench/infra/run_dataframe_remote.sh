#!/bin/bash
# Local-tier (1-100 GiB) dataframe benchmarks on the EC2 NVMe instance.
#
#   cd bench/infra
#   ./run_dataframe_remote.sh
#
# Same flow as run_remote.sh -- SSM wait, chunked upload, bootstrap, generate,
# background matrix, poll, fetch -- but the payload is the bench/dataframe
# package (plus gen.py for the ladder) and no cluster is needed: tiers at or
# below 200M rows run threads/daft/ray-data only, ray-data starting its own
# local Ray. This is the single-node dataframe tier, not distributed
# benchmarking: a ladder reaching 200M rows or more refuses without
# DIST_CLUSTER, and multi-node provisioning is still TODO. Assumes AWS credentials, pulumi, uv, and a stack that has been
# `pulumi up`'d (or pass --deploy to do it here). Do NOT run `pulumi up`
# unless asked: the instance bills while it lives.
# Results land in ../dataframe-results.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO=$(cd ../.. && pwd)   # repo root: bench/infra -> bench -> here
# ~1, 4, 16, 49 and 98 GiB at the measured 0.5072 GiB/Mrow.
LADDER=${LADDER:-2000000,8000000,32000000,97000000,194000000}
CAP_GIB=${CAP_GIB:-55}    # generous cap; the tier tops out near 100 GiB
DIST_SHARDS=${DIST_SHARDS:-16}

if [[ "${1:-}" == "--deploy" ]]; then
  pulumi up --yes
fi

IID=$(pulumi stack output instance_id)
export BENCH_INSTANCE_ID="$IID"
echo "instance: $IID"

echo "== waiting for SSM =="
until [[ "$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$IID" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)" == "Online" ]]; do
  sleep 10
done

echo "== uploading payload =="
# Fresh: `wheel/` keeps one wheel per commit, and copying them all would ship a
# stale build as well as inflating the upload.
rm -rf "$HERE/wheel"
uv build --wheel --out-dir "$HERE/wheel" "$REPO" >/dev/null
TMP=$(mktemp -d)
# driver.matrix runs `python -m bench.dataframe.driver.cases` with cwd at the
# repo root, so the payload keeps that layout: the bench package, gen.py for
# the ladder, the dataframe bootstrap, and the wheel.
mkdir -p "$TMP/payload/bench/polars_lance"
cp "$REPO/bench/__init__.py" "$TMP/payload/bench/"
cp -r "$REPO/bench/dataframe" "$TMP/payload/bench/"
cp "$REPO/bench/polars_lance/gen.py" "$TMP/payload/bench/polars_lance/"
cp bootstrap_dataframe.sh "$TMP/payload/"
cp "$HERE"/wheel/polars_pylance-*.whl "$TMP/payload/"
tar czf "$TMP/payload.tgz" --exclude='__pycache__' -C "$TMP/payload" .

# SSM takes its parameters through argv, and Linux caps a single argument at
# 128 KiB whatever ARG_MAX says, so the archive goes up in chunks and is
# reassembled on the far side. CHUNK is sized so a chunk plus its own base64
# wrapper stays well under both that and the SSM parameter limit.
CHUNK=40000
base64 -w0 < "$TMP/payload.tgz" > "$TMP/payload.b64"
split -b "$CHUNK" "$TMP/payload.b64" "$TMP/chunk."
printf 'set -eu\nmkdir -p /mnt/nvme\nrm -f /mnt/nvme/payload.b64\n' > "$TMP/start.sh"
./ssm.sh "$TMP/start.sh" 120 > /dev/null
N=$(ls "$TMP"/chunk.* | wc -l)
i=0
for c in "$TMP"/chunk.*; do
  i=$((i + 1))
  echo "  chunk $i/$N ($(stat -c%s "$c") bytes)"
  {
    printf "set -eu\ncat >> /mnt/nvme/payload.b64 <<'CHUNK_EOF'\n"
    cat "$c"
    printf "\nCHUNK_EOF\n"
  } > "$TMP/put.sh"
  ./ssm.sh "$TMP/put.sh" 120 > /dev/null
done
printf 'set -eu\nrm -rf /mnt/nvme/dataframe\nmkdir -p /mnt/nvme/dataframe\ncd /mnt/nvme/dataframe\ntr -d "\\n" < /mnt/nvme/payload.b64 | base64 -d > payload.tgz\ntar xzf payload.tgz\nrm -f /mnt/nvme/payload.b64 payload.tgz\nfind /mnt/nvme/dataframe -maxdepth 2 | sort\n' > "$TMP/unpack.sh"
./ssm.sh "$TMP/unpack.sh" 300

echo "== bootstrapping python env =="
printf 'set -eu\nBENCH_ROOT=/mnt/nvme bash /mnt/nvme/dataframe/bootstrap_dataframe.sh\n' > "$TMP/boot.sh"
./ssm.sh "$TMP/boot.sh" 1800

echo "== generating dataset ladder (this is the slow part) =="
printf 'set -eu\nBENCH_ROOT=/mnt/nvme /mnt/nvme/venv/bin/python /mnt/nvme/dataframe/bench/polars_lance/gen.py %s\ndf -h /mnt/nvme | tail -1\n' \
  "$(echo "$LADDER" | tr ',' ' ')" > "$TMP/gen.sh"
./ssm.sh "$TMP/gen.sh" 7200

echo "== running the matrix =="
cat > "$TMP/run.sh" <<EOF
set -eu
rm -f /mnt/nvme/dataframe-results.jsonl /mnt/nvme/run-dataframe.log
mkdir -p /mnt/nvme/tmp
cd /mnt/nvme/dataframe
BENCH_ROOT=/mnt/nvme TMPDIR=/mnt/nvme/tmp DIST_SHARDS=$DIST_SHARDS nohup /mnt/nvme/venv/bin/python -m bench.dataframe.driver.matrix \
  $LADDER $CAP_GIB > /mnt/nvme/run-dataframe.log 2>&1 &
echo started
EOF
./ssm.sh "$TMP/run.sh" 120

printf 'for i in $(seq 1 2000); do\n  grep -q "BENCHMARK COMPLETE" /mnt/nvme/run-dataframe.log && break\n  pgrep -f "bench.dataframe.driver.matrix" > /dev/null || break\n  sleep 20\ndone\ntail -3 /mnt/nvme/run-dataframe.log\n' > "$TMP/wait.sh"
./ssm.sh "$TMP/wait.sh" 7200

echo "== fetching results =="
# A local-tier run is dozens of records: plain text fits in one SSM response.
# The line-count check still guards against a truncated fetch.
printf 'cat /mnt/nvme/dataframe-results.jsonl\n' > "$TMP/fetch.sh"
./ssm.sh "$TMP/fetch.sh" 300 \
  | grep -E '^\{' > ../dataframe-results.jsonl
REMOTE=$(printf 'wc -l < /mnt/nvme/dataframe-results.jsonl\n' > "$TMP/count.sh"; \
         ./ssm.sh "$TMP/count.sh" 120 | grep -E '^[0-9]+$' | head -1)
LOCAL=$(wc -l < ../dataframe-results.jsonl)
echo "records: $LOCAL local, $REMOTE remote"
[ "$LOCAL" = "$REMOTE" ] || { echo "TRUNCATED FETCH, refusing to analyse" >&2; exit 1; }
uv run --group dataframe python -m bench.dataframe.analyse ../dataframe-results.jsonl | tee ../dataframe-results.txt

echo
 echo "Done. The instance is STILL RUNNING and still billing."
echo "Tear it down with:  pulumi destroy --yes"
