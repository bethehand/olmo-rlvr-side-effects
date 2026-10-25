#!/bin/bash
# Template control for arm A: the start model Think-SFT generated with A's chat template (olmo_thinker, short system prompt),
# same seed and 8K cap as A, on GPUs 0,1 after A's chain (PID given as $1) has finished. Runs heldout, IFEval-200, GSM8K-200,
# then the 805 AlpacaEval prompts. Run name tmplA.
E=/mnt/nvme0/looped/oi_work/experiments; D=$E/2026-10-23_dpo-arms-eval
export MAXTOK=8192 LIMIT_ifeval=200 LIMIT_gsm8k=200 TEMPLATE_FROM=icedduck/olmo3-7b-think-dpo-arm_A_seed1 SEED=1 HF_HUB_OFFLINE=1
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date "+%F %T") template control started: Think-SFT with the template of arm A (tmplA), GPUs 0,1, cap 8192, ifeval/gsm8k 200" >> $D/logs/done.txt
bash $D/queue_model.sh allenai/Olmo-3-7B-Think-SFT main tmplA 0,1
cd $E/2026-10-18_length && bash run_length.sh allenai/Olmo-3-7B-Think-SFT main tmplA 0,1 "" 8192
echo "$(date "+%F %T") template control done (tmplA)" >> $D/logs/done.txt
