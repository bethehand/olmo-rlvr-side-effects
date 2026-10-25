#!/bin/bash
# Runs on the laptop. Waits until the rented node has uploaded an arm's weights (its 04_push_model.sh appends a line to
# /workspace/logs/pushed.txt), then on GS01: fetches the private model into the hub cache (token passed on stdin, never on a
# command line or in a file) and calls start_arm_eval.sh, which queues the 805-prompt length run on one GPU pair and the
# held-out / IFEval / GSM8K runs on the other, both served offline from the cache.
# Usage: HF_TOKEN=... POD_SSH='ssh ...' GS01_SSH='ssh ...' [LEN_GPUS=0,1 GEN_GPUS=2,3] [CONTROL=1] watch_arm_push.sh ARM(e.g. A) [poll seconds, default 300]
#   LEN_GPUS / GEN_GPUS: GPU pairs for the 805-prompt run and for heldout / IFEval / GSM8K (the run scripts wait up to 4 h for them)
#   CONTROL=1: afterwards also run the start model Think-SFT with this arm's chat template (see start_arm_eval.sh)
set -u
ARM=$1 POLL=${2:-300}
REPO=icedduck/olmo3-7b-think-dpo-arm_${ARM}_seed1
# ssh commands for the two machines come from the environment, so no address lands in the repo
POD=${POD_SSH:?set POD_SSH to the ssh command for the rented node}
GS=${GS01_SSH:?set GS01_SSH to the ssh command for the evaluation box}
W=/mnt/nvme0/looped/oi_work
LOG=${LOG:-/tmp/watch_arm_${ARM}.log}
LEN_GPUS=${LEN_GPUS:-0,1} GEN_GPUS=${GEN_GPUS:-2,3} CONTROL=${CONTROL:-}
echo "$(date '+%F %T') watching for arm_${ARM}_seed1 in pushed.txt" >> $LOG
while true; do
  if $POD "grep -q arm_${ARM}_seed1 /workspace/logs/pushed.txt 2>/dev/null"; then break; fi
  sleep $POLL
done
echo "$(date '+%F %T') pushed: $($POD "grep arm_${ARM}_seed1 /workspace/logs/pushed.txt")" >> $LOG
# 1. fetch into GS01's hub cache (private repo: token on stdin)
echo "$HF_TOKEN" | $GS "read -r HF_TOKEN; export HF_TOKEN; source $W/env.sh >/dev/null; $W/safety-eval/.venv-se/bin/python -c \"
from huggingface_hub import snapshot_download
p = snapshot_download('$REPO'); import os; print('fetched', p, sorted(os.listdir(p)))
\"" >> $LOG 2>&1
if ! grep -q "fetched .*config.json" $LOG; then echo "$(date '+%F %T') fetch failed, stopping" >> $LOG; exit 1; fi
# 2. queue the runs, offline from the cache (no token needed)
$GS "bash $W/experiments/2026-10-23_dpo-arms-eval/start_arm_eval.sh $REPO $LEN_GPUS $GEN_GPUS ${CONTROL:+control}" >> $LOG 2>&1
echo "$(date '+%F %T') queued 805-prompt run (GPUs $LEN_GPUS) and heldout/ifeval/gsm8k (GPUs $GEN_GPUS) for $REPO${CONTROL:+, then the Think-SFT control with the template of the arm}" >> $LOG
