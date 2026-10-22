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
HERE_CONF="$HERE/ds_stage3_offload_optimizer.conf"
run_timing() {   # $1 = mode; prints rc; sets PEAK
  PRICE_PER_HOUR=$PRICE bash 02_timing_test.sh "$1"; local rc=$?
  PEAK=$(grep -E "^timing_$1_(64|256) " "$LOGS/timing_summary.txt" | tail -2 | sed -E 's/.*peak_gpu_mib=([0-9]+).*/\1/' | sort -n | tail -1)
  return $rc
}
if [ "$NGPU" -ge 8 ]; then
  MODE=concat; run_timing concat; rc=$?
  if [ "$rc" != 0 ] || [ "${PEAK:-0}" -gt 75000 ]; then log "concat rc=$rc peak=${PEAK:-?} -> sep"; MODE=sep; run_timing sep; rc=$?; fi
else
  MODE=sep        # below 8 x 80 GB the concatenated forward's logits peak does not fit
  if [ "$NGPU" -lt 4 ]; then export DS_CONF="$HERE_CONF"; fi   # measured: no-offload peaks at 77-78 GB on 3 x 80 GB
  log "gpus=$NGPU -> sep mode (effective batch $(( (32 / NGPU) * NGPU ))), ds_conf=${DS_CONF:-no-offload}, omp_threads=$OMP_NUM_THREADS"
  run_timing sep; rc=$?
fi
if [ "$rc" != 0 ] || [ "${PEAK:-0}" -gt 75000 ]; then
  # measured on 3 x H100 80 GB: with full gradient checkpointing and separate passes the first backward still
  # peaks at 78 GB (37 GB of ZeRO-3 states + ~41 GB of 16K logits/activations/buffers); moving the optimizer
  # states to CPU drops the per-GPU state share to ~9 GB
  log "no-offload rc=$rc peak=${PEAK:-?} MiB -> retrying with optimizer states offloaded to CPU"
  export DS_CONF="$HERE_CONF"
  mv "$LOGS/timing_summary.txt" "$LOGS/timing_summary_nooffload.txt" 2>/dev/null
  run_timing "$MODE" || { log "timing failed with optimizer offload too (peak=${PEAK:-?})"; exit 1; }
  log "optimizer offload works: peak=${PEAK:-?} MiB"
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
