#!/bin/bash
# Template control, full protocol, part 2a: IFEval 541 for Think-SFT with arm A's template (tmplA, seed 1, 32K) on the GPU pair $1.
# Run on the idle pair while arm B trains; the GSM8K part follows later (ctrl GSM8K 1,319 takes ~1.6 h and would delay arm B's 805 run).
E=/mnt/nvme0/looped/oi_work/experiments; D=$E/2026-10-23_dpo-arms-eval; G=${1:-0,1}
export MAXTOK=32768 TEMPLATE_FROM=icedduck/olmo3-7b-think-dpo-arm_A_seed1 SEED=1 HF_HUB_OFFLINE=1
echo "$(date "+%F %T") template control part 2a (IFEval 541, tmplA, 32K) started on GPUs $G" >> $D/logs/done.txt
cd $D && bash run_gen.sh allenai/Olmo-3-7B-Think-SFT main ifeval tmplA $G
echo "$(date "+%F %T") template control part 2a done" >> $D/logs/done.txt
