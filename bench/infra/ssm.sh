#!/bin/bash
# ssm.sh <script-file> [timeout-seconds]
# Run a script on the benchmark instance over SSM (no SSH, no inbound rules).
#
# The script is base64-encoded because SSM splits its `commands` parameter on
# commas, which would shred any real shell script. Each invocation gets its own
# temp file so concurrent commands cannot overwrite each other.
set -euo pipefail
export AWS_REGION=${AWS_REGION:-eu-central-1}

IID=${BENCH_INSTANCE_ID:-$(cd "$(dirname "$0")" && pulumi stack output instance_id)}
TMO=${2:-3600}
B64=$(base64 -w0 < "$1")

CMD_ID=$(aws ssm send-command --instance-ids "$IID" \
  --document-name AWS-RunShellScript --timeout-seconds 3600 \
  --parameters commands="f=\$(mktemp /tmp/_run.XXXXXX.sh); echo $B64 | base64 -d > \$f && bash \$f; rc=\$?; rm -f \$f; exit \$rc",executionTimeout="$TMO" \
  --query 'Command.CommandId' --output text)
echo "command-id: $CMD_ID" >&2

for _ in $(seq 1 "$((TMO / 5 + 12))"); do
  ST=$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$IID" \
       --query 'Status' --output text 2>/dev/null || echo Pending)
  case "$ST" in Success|Failed|Cancelled|TimedOut) break;; esac
  sleep 5
done
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$IID" \
  --query '{Status:Status,Out:StandardOutputContent,Err:StandardErrorContent}' --output json \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f\"[{d['Status']}]\")
if d.get('Out'): print(d['Out'].rstrip())
if d.get('Err','').strip(): print('--- stderr (tail) ---'); print(d['Err'].rstrip()[-2500:])
"
