#!/bin/bash
# Usage: run_rerun.sh MODEL_REPO GPUS NAME [extra args for rerun_h045_turns.py]   e.g. run_rerun.sh allenai/Olmo-3-7B-Think-SFT 0,1 7b_think_sft
# Same software as the 2026-10-18 length runs (safety-eval's environment: vLLM 0.11.0, transformers 4.57.1).
set -u
MODEL=$1 GPUS=$2 NAME=$3; shift 3
W=/mnt/nvme0/looped/oi_work; source $W/env.sh >/dev/null
E=$W/experiments/2026-10-27_think-trace-lengths; cd $E; mkdir -p outputs/h045_rerun logs
export CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=/tmp/ray_$USER
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
# wait up to 2 h for these GPUs to be free (never touch other people's processes)
for i in $(seq 1 720); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=",$GPUS," 'index(g, "," $1 ",") && $2 > 1000' | wc -l)
  [ "$busy" -eq 0 ] && break; sleep 10
done
t0=$(date +%s)
$W/safety-eval/.venv-se/bin/python rerun_h045_turns.py --chat_logs untrusted/chat_logs.md --model $MODEL --tp $NG --out outputs/h045_rerun/${NAME}.jsonl "$@" > logs/h045_rerun_${NAME}.log 2>&1
echo "$(date '+%F %T') $NAME exit=$? seconds=$(( $(date +%s) - t0 ))" >> logs/h045_rerun_done.txt
