#!/bin/bash
# Usage: run_probe.sh MODEL_REPO REVISION RUN_NAME GPUS
#   env: N (samples, 3) CONDS (default,olmo,generic,robot) MAXTOK (16384) LIMIT DTYPE GPU_MEM MAX_SEQS SEED (1) MAXLEN (max_model_len; 20480 for a 7B on one card)
# Same environment as the length runs (safety-eval's venv: vLLM 0.11.0, transformers 4.57.1).
set -u
MODEL=$1 REV=$2 RUN=$3 GPUS=$4
W=/mnt/nvme0/looped/oi_work; source $W/env.sh >/dev/null
E=$W/experiments/2026-10-20_identity-probe; cd $E; mkdir -p outputs logs
NAME=${MODEL#allenai/}@$REV
export CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=/tmp/ray_$USER
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
# Before starting: clear orphaned vLLM workers of ours on these GPUs, then wait up to 10 min for the GPUs to be free
# (copied from ../2026-10-18_length/run_length.sh).
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
$W/safety-eval/.venv-se/bin/python generate_identity.py --model $MODEL --revision $REV --prompts prompts.json --tp $NG \
  --out outputs/${NAME}_$RUN.jsonl --n ${N:-3} --conditions ${CONDS:-default,olmo,generic,robot} --max_new_tokens ${MAXTOK:-16384} \
  --seed ${SEED:-1} ${LIMIT:+--limit $LIMIT} ${DTYPE:+--dtype $DTYPE} ${GPU_MEM:+--gpu_mem $GPU_MEM} ${MAX_SEQS:+--max_num_seqs $MAX_SEQS} ${MAXLEN:+--max_model_len $MAXLEN} \
  >> logs/${NAME}_$RUN.log 2>&1
rc=$?
echo "$NAME $RUN exit=$rc seconds=$(( $(date +%s) - t0 )) rows=$(cat outputs/${NAME}_$RUN.jsonl 2>/dev/null | wc -l)" >> logs/done.txt
exit $rc
