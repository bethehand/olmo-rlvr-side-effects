#!/bin/bash
# Template control on the full protocol (32K cap, full sets): Think-SFT with arm A's chat template, seed 1. Part 1 (this script):
# held-out set, then the 805 AlpacaEval prompts, on the GPU pair given as $1 (free between arm A's 805 run and arm B's arrival).
# Part 2 (IFEval 541 + GSM8K 1319, queue_model.sh ... tmplA) is launched separately when a GPU pair is free after arm B's runs.
E=/mnt/nvme0/looped/oi_work/experiments; D=$E/2026-10-23_dpo-arms-eval; G=${1:-2,3}
export MAXTOK=32768 TEMPLATE_FROM=icedduck/olmo3-7b-think-dpo-arm_A_seed1 SEED=1 HF_HUB_OFFLINE=1
echo "$(date "+%F %T") template control, full protocol, part 1 (heldout + 805, 32K) started on GPUs $G" >> $D/logs/done.txt
cd $D && bash run_gen.sh allenai/Olmo-3-7B-Think-SFT main heldout tmplA $G
cd $E/2026-10-18_length && bash run_length.sh allenai/Olmo-3-7B-Think-SFT main tmplA $G "" 32768
echo "$(date "+%F %T") template control part 1 done (heldout + 805, tmplA, 32K)" >> $D/logs/done.txt
