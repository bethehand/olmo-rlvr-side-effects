#!/bin/bash
# Template control, full protocol, part 2b: GSM8K 1,319 for Think-SFT with arm A's template (tmplA, seed 1, 32K) on GPU pair $1.
E=/mnt/nvme0/looped/oi_work/experiments; D=$E/2026-10-23_dpo-arms-eval; G=${1:-0,1}
export MAXTOK=32768 TEMPLATE_FROM=icedduck/olmo3-7b-think-dpo-arm_A_seed1 SEED=1 HF_HUB_OFFLINE=1
echo "$(date "+%F %T") template control part 2b (GSM8K 1319, tmplA, 32K) started on GPUs $G" >> $D/logs/done.txt
cd $D && bash run_gen.sh allenai/Olmo-3-7B-Think-SFT main gsm8k tmplA $G
echo "$(date "+%F %T") template control part 2b done; the template control is now complete on all four sets" >> $D/logs/done.txt
