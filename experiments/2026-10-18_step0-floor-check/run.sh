#!/bin/bash
# Step 0 on GS01: one model per GPU, four step_100 checkpoints at once; the base model when it has downloaded.
# Usage: run.sh [step100|base]
source /mnt/nvme0/looped/oi_work/env.sh >/dev/null && source .venv/bin/activate
E=/mnt/nvme0/looped/oi_work/experiments/2026-10-18_step0-floor-check; cd $E; mkdir -p outputs logs
[ -f prompts.jsonl ] || python build_prompts.py
V=/mnt/nvme0/looped/oi_work/model_views
gen() { MODEL_VIEWS=$V python ../make_model_views.py "$2@$3" >> logs/views.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 python generate.py --model $2 --revision $3 --model_path "$V/${2#allenai/}@$3" --template $4 --prompts prompts.jsonl --out outputs/$5.jsonl > logs/$5.log 2>&1; echo "$5 exit=$?" >> logs/done.txt; }
if [ "${1:-step100}" = step100 ]; then
  gen 0 allenai/Olmo-3-7B-RL-Zero-Math step_100 olmo_thinker_rlzero math_step100 &
  gen 1 allenai/Olmo-3-7B-RL-Zero-Code step_100 olmo_thinker_code_rlzero code_step100 &
  gen 2 allenai/Olmo-3-7B-RL-Zero-IF step_100 olmo_thinker_rlzero if_step100 &
  gen 3 allenai/Olmo-3-7B-RL-Zero-General step_100 olmo_thinker general_step100 &
  wait
else
  gen 0 allenai/Olmo-3-1025-7B main olmo_thinker_rlzero base &
  wait
fi
python judge_keywords.py outputs | tee logs/table_$(date +%H%M).txt
