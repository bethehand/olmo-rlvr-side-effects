#!/bin/bash
# Second generation seed on the 805 AlpacaEval prompts (run2 = engine seed 2; everything else as run1: 32K cap, temperature 0.6,
# top_p 0.95, own template) for arms A, B, B' and the training-template Think-SFT control (tmplA2 = seed 2 with arm A's template).
# Purpose: show how much of B − A and B' − A moves under re-sampling alone (the "one sample per prompt" caveat), not a second
# training seed. GS01, 2026-10-25. Starts only after the judge has finished with exit 0 and all four GPUs have been idle for a
# minute; two runs at a time (GPUs 0,1 and 2,3), about 3 h in all. Own log lines carry the prefix SEED2 (lesson 4).
set -u
W=/mnt/nvme0/looped/oi_work; E=$W/experiments/2026-10-23_dpo-arms-eval; LE=$W/experiments/2026-10-18_length
source $W/env.sh >/dev/null; export HF_HUB_OFFLINE=1
log() { echo "$(date '+%F %T') SEED2 $*" >> $E/logs/done.txt; }
gpus_busy() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=",$1," 'index(g, "," $1 ",") && $2 > 1000 {n++} END {exit n > 0 ? 0 : 1}'; }
rows() { cat "$1" 2>/dev/null | wc -l; }
JUDGE_RE="^[0-9-]+ [0-9:]+ JUDGE all finished exit=0 records=[0-9]+ seconds=[0-9]+\$"
A=icedduck/olmo3-7b-think-dpo-arm_A_seed1; B=icedduck/olmo3-7b-think-dpo-arm_B_seed1; BP=icedduck/olmo3-7b-think-dpo-arm_Bp_seed1
for f in icedduck_olmo3-7b-think-dpo-arm_A_seed1@main_run2 icedduck_olmo3-7b-think-dpo-arm_B_seed1@main_run2 icedduck_olmo3-7b-think-dpo-arm_Bp_seed1@main_run2 Olmo-3-7B-Think-SFT@main_tmplA2; do
  [ -e $LE/outputs/$f.jsonl ] && { log "$f.jsonl already exists; not starting"; exit 1; }
done
for i in $(seq 1 600); do grep -qE "$JUDGE_RE" $E/logs/done.txt && break; sleep 30; done   # up to 5 h
grep -qE "$JUDGE_RE" $E/logs/done.txt || { log "judge not finished with exit 0 within 5 h; run2 not started"; exit 1; }
sleep 30; while gpus_busy 0,1,2,3; do sleep 60; done; sleep 60; while gpus_busy 0,1,2,3; do sleep 60; done
log "started run2 (seed 2) on the 805 prompts: A on GPUs 0,1, B on GPUs 2,3; B' and the tmplA2 control follow on the pair that frees first"
cd $LE
bash run_length.sh $A main run2 0,1 '' 32768 > $E/logs/seed2_A.out 2>&1 &
pA=$!
bash run_length.sh $B main run2 2,3 '' 32768 > $E/logs/seed2_B.out 2>&1 &
pB=$!
wait $pA
bash run_length.sh $BP main run2 0,1 '' 32768 > $E/logs/seed2_Bp.out 2>&1 &
pBP=$!
wait $pB
SEED=2 TEMPLATE_FROM=$A bash run_length.sh allenai/Olmo-3-7B-Think-SFT main tmplA2 2,3 '' 32768 > $E/logs/seed2_tmplA.out 2>&1 &
pT=$!
wait $pBP $pT
log "run2 complete: rows A $(rows $LE/outputs/icedduck_olmo3-7b-think-dpo-arm_A_seed1@main_run2.jsonl), B $(rows $LE/outputs/icedduck_olmo3-7b-think-dpo-arm_B_seed1@main_run2.jsonl), B' $(rows $LE/outputs/icedduck_olmo3-7b-think-dpo-arm_Bp_seed1@main_run2.jsonl), tmplA2 $(rows $LE/outputs/Olmo-3-7B-Think-SFT@main_tmplA2.jsonl) of 805 each"
