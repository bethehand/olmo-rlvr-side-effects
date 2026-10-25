#!/bin/bash
# Arm B' on the same pod, right after the A/B pipeline: waits until the pipeline has logged its final marker (a line ending in
# "Z PIPELINE DONE", written after arm B is pushed and its step_* checkpoints removed) AND no trainer process is left on the GPUs,
# rebuilds arm_Bp.jsonl from the public dataset and ../arms/arm_Bp_ids.json unless it already exists, trains with the same command
# as A and B (sep mode, optimizer states offloaded on fewer than 4 GPUs), pushes, and writes "BPRIME DONE".
# The wait condition deliberately matches the end of the line: on 2026-10-23 a first version grepped for the bare phrase, matched
# its own "waiting for ..." log line and started training while arm B was still on the GPUs (killed within a minute).
# Usage (HF_TOKEN in the environment, never on the command line): HF_USER=... 06_train_bprime.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; source ./env_node.sh
: "${HF_TOKEN:?export HF_TOKEN first (needed to push the model)}"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/pipeline.txt"; }
log "B-prime runner started; waiting for the A/B pipeline to finish"
until grep -qE "Z PIPELINE DONE$" "$LOGS/pipeline.txt" 2>/dev/null; do
  if grep -qE "Z (STOP|training arm . failed|push arm . failed)" "$LOGS/pipeline.txt" 2>/dev/null; then log "A/B pipeline reported a failure; B-prime not started"; exit 1; fi
  sleep 120
done
while pgrep -f "[d]po_tune_cache.py" >/dev/null; do sleep 60; done   # never start while another trainer holds the GPUs
MODE=sep; [ "$NGPU" -lt 4 ] && export DS_CONF="$HERE/ds_stage3_offload_optimizer.conf"
log "B-prime start: gpus=$NGPU mode=$MODE ds_conf=${DS_CONF:-no-offload} volume=$(df -h "$WS" | tail -1 | awk '{print $4" free"}')"
if [ "$(wc -l < "$ARMS/arm_Bp.jsonl" 2>/dev/null | tr -d ' ')" != 20000 ]; then
  "$VENV/bin/python" rebuild_arms_from_ids.py "$ARMS" Bp || { log "B-prime rebuild failed"; exit 2; }
fi
n=$(wc -l < "$ARMS/arm_Bp.jsonl" | tr -d ' '); [ "$n" = 20000 ] || { log "arm_Bp.jsonl has $n pairs, expected 20000"; exit 2; }
log "arm_Bp.jsonl ready: $n pairs"
bash 03_train_arm.sh Bp 1 "$MODE" || { log "training arm Bp failed"; exit 3; }
bash 04_push_model.sh Bp 1 || { log "push arm Bp failed"; exit 4; }
log "arm Bp done and pushed"
log "BPRIME DONE"
