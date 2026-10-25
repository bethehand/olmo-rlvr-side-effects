#!/bin/bash
# Judge on the held-out set for all six output files (Qwen3-32B bf16, tp 4, non-thinking, greedy; see judge_heldout.py).
# Pairs, each asked in both orders: Think-SFT (training template) vs A, vs B, vs B'; A vs B, A vs B', B vs B';
# Think-SFT (released template) vs Think-DPO. Rubric coverage on the 50 detail prompts for all six files.
# 7 pairs x 150 prompts x 2 orders + 6 files x 50 = 2400 requests. Output judgments/heldout_run1.jsonl
# (an earlier file of that name is moved aside, never overwritten). Called by bp_pipeline.sh once all four GPUs are idle.
# Analyze with: analyze_heldout.py outputs/heldout/*.jsonl --judgments judgments/heldout_run1.jsonl
# 2026-10-25 05:50: the first launch (gpu_mem 0.90, vLLM default max_num_seqs 256) died in the sampler warm-up with CUDA out of memory
# on three of four workers (weights 16 GB + KV cache 4.35 GB per 24 GB card left no room); relaunched with gpu_mem 0.85 and
# max_num_seqs 64, the settings the 32B runs of 2026-10-19 used on this box, after a three-prompt smoke run.
W=/mnt/nvme0/looped/oi_work; D=$W/experiments/2026-10-23_dpo-arms-eval; O=$D/outputs/heldout
source $W/env.sh >/dev/null; cd $D; export HF_HUB_OFFLINE=1
log() { echo "$(date "+%F %T") JUDGE $*" >> $D/logs/done.txt; }
SFTT=$O/Olmo-3-7B-Think-SFT@main_tmplA.jsonl; SFTR=$O/Olmo-3-7B-Think-SFT@main_run1.jsonl; DPO=$O/Olmo-3-7B-Think-DPO@main_run1.jsonl
A=$O/icedduck_olmo3-7b-think-dpo-arm_A_seed1@main_run1.jsonl; B=$O/icedduck_olmo3-7b-think-dpo-arm_B_seed1@main_run1.jsonl
BP=$O/icedduck_olmo3-7b-think-dpo-arm_Bp_seed1@main_run1.jsonl
for f in $SFTT $SFTR $DPO $A $B $BP; do
  [ "$(wc -l < $f 2>/dev/null)" = 150 ] || { log "$f has $(wc -l < $f 2>/dev/null) rows, expected 150; stopping"; exit 1; }
done
mkdir -p judgments
[ -f judgments/heldout_run1.jsonl ] && mv judgments/heldout_run1.jsonl judgments/heldout_run1_superseded_$(date +%m%d_%H%M).jsonl
log "all started on GPUs 0-3: 7 pairs x 150 x 2 orders + rubric on 6 files (Qwen3-32B bf16, tp 4, gpu_mem 0.85, max_num_seqs 64, max_model_len 20000)"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3 $W/safety-eval/.venv-se/bin/python judge_heldout.py \
  --pairs "$SFTT:$A,$SFTT:$B,$SFTT:$BP,$A:$B,$A:$BP,$B:$BP,$SFTR:$DPO" \
  --rubric $SFTT $SFTR $DPO $A $B $BP --out judgments/heldout_run1.jsonl --tp 4 --gpu_mem 0.85 --max_num_seqs 64 > logs/judge_all.log 2>&1
rc=$?
log "all finished exit=$rc records=$(wc -l < judgments/heldout_run1.jsonl 2>/dev/null) seconds=$(( $(date +%s) - t0 ))"
