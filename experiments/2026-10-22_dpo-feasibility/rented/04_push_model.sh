#!/bin/bash
# Upload a finished arm's final weights to a private model repo on the Hugging Face Hub so the pod
# can be stopped; evaluation runs elsewhere. Usage: 04_push_model.sh A 1
set -euo pipefail
cd "$(dirname "$0")"; source ./env_node.sh; source "$VENV/bin/activate"
: "${HF_TOKEN:?export HF_TOKEN first}"
ARM=${1:?}; SEED=${2:?}; TAG=arm_${ARM}_seed${SEED}; SRC=$OUT/$TAG; REPO=$HF_USER/olmo3-7b-think-dpo-${TAG}
[ -f "$SRC/config.json" ] || { echo "no final model in $SRC (config.json missing)"; exit 1; }
cp "$SRC/run_info.txt" "$SRC/run_info_$TAG.txt" 2>/dev/null || true
hf upload "$REPO" "$SRC" . --repo-type model --private \
  --exclude "step_*/**" --exclude "*.pt" --exclude "*_gpu.csv" \
  --commit-message "DPO $TAG: Ai2 7B Think recipe (dpo_norm, beta 5, lr 8e-8, 16K) on a 20,000-pair arm"
echo "$TAG -> https://huggingface.co/$REPO (private) $(date -u +%FT%TZ)" | tee -a "$LOGS/pushed.txt"
# The upload succeeded (set -e), so the ~100 GB step_* checkpoint (optimizer states included) is no longer
# needed; the volume must hold two checkpoints of the next arm while a new one is being written.
rm -rf "$SRC"/step_* && echo "removed $TAG step_* checkpoints; volume: $(df -h "$WS" | tail -1 | awk '{print $3" used of "$2}')"
