#!/bin/bash
# Usage: run_length.sh MODEL_REPO REVISION RUN_NAME GPUS [LIMIT] [MAX_NEW_TOKENS, default 32768]
#   e.g. run_length.sh allenai/Olmo-3-7B-Think-SFT main run1 2,3
# Same software as Ai2's safety evals (safety-eval's environment: vLLM 0.11.0, transformers 4.57.1).
set -u
MODEL=$1 REV=$2 RUN=$3 GPUS=$4 LIMIT=${5:-} MAXTOK=${6:-32768}
W=/mnt/nvme0/looped/oi_work; source $W/env.sh >/dev/null
E=$W/experiments/2026-10-18_length; cd $E; mkdir -p outputs logs
P=$(ls $HF_HUB_CACHE/datasets--tatsu-lab--alpaca_eval/snapshots/*/alpaca_eval.json | head -1)
NAME=${MODEL#allenai/}@$REV
export CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=/tmp/ray_$USER
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
t0=$(date +%s)
$W/safety-eval/.venv-se/bin/python generate_length.py --model $MODEL --revision $REV --prompts $P --tp $NG \
  --out outputs/${NAME}_$RUN.jsonl ${LIMIT:+--limit $LIMIT} --max_new_tokens $MAXTOK >> logs/${NAME}_$RUN.log 2>&1
echo "$NAME $RUN exit=$? seconds=$(( $(date +%s) - t0 ))" >> logs/done.txt
