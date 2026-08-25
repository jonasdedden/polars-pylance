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
REPO=$(cd ../../.. && pwd)
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
uv build --wheel --out-dir "$HERE/dist" "$REPO" >/dev/null
TMP=$(mktemp -d)
cp ../gen.py ../cases.py ../run_matrix.py bootstrap.sh "$TMP/"
cp "$HERE"/dist/polars_pylance-*.whl "$TMP/"
tar czf "$TMP/payload.tgz" -C "$TMP" .
python3 - "$TMP/payload.tgz" > "$TMP/upload.sh" <<'PY'
import base64, sys, pathlib
b64 = base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
print(f"set -eu\nmkdir -p /mnt/nvme\necho '{b64}' | base64 -d > /mnt/nvme/payload.tgz\ncd /mnt/nvme && tar xzf payload.tgz && ls -1 /mnt/nvme")
PY
./ssm.sh "$TMP/upload.sh" 300

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
