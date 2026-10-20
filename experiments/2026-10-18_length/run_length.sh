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
# Seed: runN uses N (override with SEED=...), so repeated runs are independent samples. Without a seed, vLLM V1 uses 0
# and a rerun can come out as a byte-identical copy (seen on 2026-10-18).
[ -z "${SEED:-}" ] && case "$RUN" in run[0-9]*) SEED=${RUN#run};; esac
export CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=/tmp/ray_$USER
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
# Before starting: clear orphaned vLLM workers of ours on these GPUs (a crashed engine can leave its workers behind,
# reparented to init, still holding memory), then wait up to 10 min for the GPUs to be free.
uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=",$GPUS," 'index(g, "," $1 ",") {print $2}')
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read pid uuid; do
  echo "$uuids" | grep -q "$uuid" || continue
  [ "$(ps -o user= -p $pid 2>/dev/null)" = "$USER" ] && [ "$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')" = 1 ] \
    && ps -o comm= -p $pid | grep -q "^VLLM::" && { echo "$(date '+%F %T') killing orphaned $pid on $uuid" >> logs/${NAME}_$RUN.log; kill -9 $pid; }
done
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=",$GPUS," 'index(g, "," $1 ",") && $2 > 1000' | wc -l)
  [ "$busy" -eq 0 ] && break; sleep 10
done
t0=$(date +%s)
$W/safety-eval/.venv-se/bin/python generate_length.py --model $MODEL --revision $REV --prompts $P --tp $NG \
  --out outputs/${NAME}_$RUN.jsonl ${LIMIT:+--limit $LIMIT} --max_new_tokens $MAXTOK ${DTYPE:+--dtype $DTYPE} ${TEMPLATE_FROM:+--template_from $TEMPLATE_FROM} ${GPU_MEM:+--gpu_mem $GPU_MEM} ${MAX_SEQS:+--max_num_seqs $MAX_SEQS} ${SEED:+--seed $SEED} ${SYSTEM_FILE:+--system_file $SYSTEM_FILE} >> logs/${NAME}_$RUN.log 2>&1
echo "$NAME $RUN exit=$? seconds=$(( $(date +%s) - t0 ))" >> logs/done.txt
# If not every prompt got written (crash, out of memory, ...), queue the same run again; the rerun resumes where it
# stopped. At most two retries.
n=$(cat outputs/${NAME}_$RUN.jsonl 2>/dev/null | wc -l); want=${LIMIT:-805}
if [ "$n" -lt "$want" ] && [ "${RETRY:-0}" -lt 2 ]; then
  # requeue on the shared queue (the GPU runners export $GPUS), keeping DTYPE / TEMPLATE_FROM
  ( flock 9; echo "RETRY=$(( ${RETRY:-0} + 1 )) DTYPE=${DTYPE:-} TEMPLATE_FROM=${TEMPLATE_FROM:-} GPU_MEM=${GPU_MEM:-} MAX_SEQS=${MAX_SEQS:-} RETRY_QUEUE=${RETRY_QUEUE:-} SEED=${SEED:-} SYSTEM_FILE=${SYSTEM_FILE:-} bash $E/run_length.sh $MODEL $REV $RUN \$GPUS '$LIMIT' $MAXTOK" >> ${RETRY_QUEUE:-$W/queues/safety} ) 9>${RETRY_QUEUE:-$W/queues/safety}.lock
  echo "$NAME $RUN: $n of $want written, queued retry $(( ${RETRY:-0} + 1 ))" >> logs/done.txt
fi
