#!/bin/bash
# Usage: run_length.sh MODEL_REPO REVISION RUN_NAME GPUS [LIMIT] [MAX_NEW_TOKENS, default 32768]
#   e.g. run_length.sh allenai/Olmo-3-7B-Think-SFT main run1 2,3     (env DTYPE=bfloat16 for fp32 checkpoints)
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
  --out outputs/${NAME}_$RUN.jsonl ${LIMIT:+--limit $LIMIT} --max_new_tokens $MAXTOK ${DTYPE:+--dtype $DTYPE} ${TEMPLATE_FROM:+--template_from $TEMPLATE_FROM} >> logs/${NAME}_$RUN.log 2>&1
echo "$NAME $RUN exit=$? seconds=$(( $(date +%s) - t0 ))" >> logs/done.txt
# If not every prompt got written (crash, out of memory, ...), queue the same run again; the rerun resumes where it
# stopped. At most two retries.
n=$(cat outputs/${NAME}_$RUN.jsonl 2>/dev/null | wc -l); want=${LIMIT:-805}
if [ "$n" -lt "$want" ] && [ "${RETRY:-0}" -lt 2 ]; then
  # requeue on the shared queue (the GPU runners export $GPUS), keeping DTYPE / TEMPLATE_FROM
  ( flock 9; echo "RETRY=$(( ${RETRY:-0} + 1 )) DTYPE=${DTYPE:-} TEMPLATE_FROM=${TEMPLATE_FROM:-} bash $E/run_length.sh $MODEL $REV $RUN \$GPUS '$LIMIT' $MAXTOK" >> ${RETRY_QUEUE:-$W/queues/safety} ) 9>${RETRY_QUEUE:-$W/queues/safety}.lock
  echo "$NAME $RUN: $n of $want written, queued retry $(( ${RETRY:-0} + 1 ))" >> logs/done.txt
fi
