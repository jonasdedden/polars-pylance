#!/bin/bash
# End-to-end large-scale benchmark on a fresh EC2 instance with local NVMe.
#
#   cd bench/infra
#   pulumi stack init bench && pulumi config set aws:region eu-central-1
#   ./run_remote.sh
#
# Assumes: AWS credentials, pulumi, uv, and a stack that has been `pulumi up`'d
# (or pass --deploy to do it here). Results land in ../results.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO=$(cd ../.. && pwd)   # repo root: bench/infra -> bench -> here
LADDER=${LADDER:-2000000,4000000,8000000,16000000,32000000,48000000,97000000,194000000,388000000}
BIG_CAP=${BIG_CAP:-55}     # GiB: generous cap, measures how peak RSS scales
SMALL_CAP=${SMALL_CAP:-8}  # GiB: fixed budget, answers "can it proceed at all"

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
# Fresh: `dist/` keeps one wheel per commit, and copying them all would ship a
# stale build as well as inflating the upload.
rm -rf "$HERE/dist"
uv build --wheel --out-dir "$HERE/dist" "$REPO" >/dev/null
TMP=$(mktemp -d)
# The archive must not live in the directory being archived: tar notices it
# growing under itself and exits non-zero, which `set -e` turns into an abort.
mkdir "$TMP/payload"
cp ../gen.py ../index.py ../cases.py ../run_matrix.py bootstrap.sh "$TMP/payload/"
cp "$HERE"/dist/polars_pylance-*.whl "$TMP/payload/"
tar czf "$TMP/payload.tgz" -C "$TMP/payload" .

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
printf 'set -eu\ncd /mnt/nvme\ntr -d "\\n" < payload.b64 | base64 -d > payload.tgz\ntar xzf payload.tgz\nrm -f payload.b64 payload.tgz\nls -1 /mnt/nvme\n' > "$TMP/unpack.sh"
./ssm.sh "$TMP/unpack.sh" 300

echo "== bootstrapping python env =="
printf 'set -eu\nBENCH_ROOT=/mnt/nvme bash /mnt/nvme/bootstrap.sh\n' > "$TMP/boot.sh"
./ssm.sh "$TMP/boot.sh" 1200

echo "== generating dataset ladder (this is the slow part) =="
printf 'set -eu\nBENCH_ROOT=/mnt/nvme /mnt/nvme/venv/bin/python /mnt/nvme/gen.py %s\ndf -h /mnt/nvme | tail -1\n' \
  "$(echo "$LADDER" | tr ',' ' ')" > "$TMP/gen.sh"
./ssm.sh "$TMP/gen.sh" 7200

echo "== running the matrix =="
cat > "$TMP/run.sh" <<EOF
set -eu
rm -f /mnt/nvme/results.jsonl /mnt/nvme/run.log
cd /mnt/nvme
BENCH_ROOT=/mnt/nvme nohup /mnt/nvme/venv/bin/python /mnt/nvme/run_matrix.py \\
  $LADDER $BIG_CAP $SMALL_CAP > /mnt/nvme/run.log 2>&1 &
echo started
EOF
./ssm.sh "$TMP/run.sh" 120

printf 'for i in $(seq 1 2000); do\n  grep -q "BENCHMARK COMPLETE" /mnt/nvme/run.log && break\n  pgrep -f run_matrix.py > /dev/null || break\n  sleep 20\ndone\ntail -3 /mnt/nvme/run.log\n' > "$TMP/wait.sh"
./ssm.sh "$TMP/wait.sh" 7200

echo "== fetching results =="
printf 'cat /mnt/nvme/results.jsonl\n' > "$TMP/fetch.sh"
./ssm.sh "$TMP/fetch.sh" 300 | grep '^{' > ../results.jsonl
wc -l < ../results.jsonl
python3 ../analyse.py ../results.jsonl | tee ../results.txt

echo
echo "Done. The instance is STILL RUNNING and still billing."
echo "Tear it down with:  pulumi destroy --yes"
