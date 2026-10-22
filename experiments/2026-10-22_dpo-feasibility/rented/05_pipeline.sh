#!/bin/bash
# End-to-end on the GPU pod, so nothing idles while a person is asleep:
#   timing test (concat; falls back to sep if it fails or peaks above 75,000 MiB) -> budget gate ->
#   arm A seed 1 -> push -> arm B seed 1 -> push.
# Usage: HF_TOKEN=... HF_USER=... PRICE_PER_HOUR=15.97 05_pipeline.sh [budget_usd=500]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; source ./env_node.sh
: "${HF_TOKEN:?export HF_TOKEN first (needed to push the models)}"
BUDGET=${1:-500}; PRICE=${PRICE_PER_HOUR:-15.97}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/pipeline.txt"; }
log "pipeline start gpus=$NGPU price=$PRICE budget=$BUDGET"
MODE=concat
PRICE_PER_HOUR=$PRICE bash 02_timing_test.sh concat; rc=$?
peak=$(grep -E "^timing_concat_(64|256) " "$LOGS/timing_summary.txt" | sed -E 's/.*peak_gpu_mib=([0-9]+).*/\1/' | sort -n | tail -1)
if [ "$rc" != 0 ] || [ "${peak:-0}" -gt 75000 ]; then
  log "concat timing rc=$rc peak=${peak:-?} MiB -> switching to sep"
  MODE=sep
  PRICE_PER_HOUR=$PRICE bash 02_timing_test.sh sep || { log "timing failed in both modes"; exit 1; }
fi
cost_ab=$(grep -E "^A\+B" "$LOGS/timing_summary.txt" | tail -1 | sed -E 's/^A\+B ~[0-9.]+ h \$([0-9]+).*/\1/')
log "mode=$MODE projected A+B cost=\$${cost_ab:-?} (budget \$$BUDGET)"
if [ -z "${cost_ab:-}" ] || [ "$cost_ab" -gt "$BUDGET" ]; then log "STOP: over budget or no projection"; exit 2; fi
for arm in A B; do
  bash 03_train_arm.sh "$arm" 1 "$MODE" || { log "training arm $arm failed"; exit 3; }
  bash 04_push_model.sh "$arm" 1 || { log "push arm $arm failed"; exit 4; }
  log "arm $arm done and pushed"
done
log "PIPELINE DONE"
