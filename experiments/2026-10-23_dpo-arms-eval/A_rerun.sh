#!/bin/bash
# Re-evaluation of arm A after fix_tf5_compat.py (its config had loaded with rope_theta 10000 and no YaRN under transformers 4.57),
# back on the pre-registered protocol: 32K cap, full IFEval and GSM8K. GPUs 2,3: held-out set first (quick check that the
# degeneration is gone), then the 805 AlpacaEval prompts. GPUs 0,1: held-out (resume) / IFEval / GSM8K once the template-control
# chain (PID $1) has finished.
E=/mnt/nvme0/looped/oi_work/experiments; D=$E/2026-10-23_dpo-arms-eval; R=icedduck/olmo3-7b-think-dpo-arm_A_seed1
export HF_HUB_OFFLINE=1 MAXTOK=32768
echo "$(date "+%F %T") arm A re-evaluation started (fixed config, 32K cap, full sets): heldout then 805 on GPUs 2,3; heldout/ifeval/gsm8k on GPUs 0,1 after the template control" >> $D/logs/done.txt
( cd $D && bash run_gen.sh $R main heldout run1 2,3; cd $E/2026-10-18_length && bash run_length.sh $R main run1 2,3 "" 32768 ) > $D/logs/chain_len_arm_A_fixed.out 2>&1 &
( while kill -0 "$1" 2>/dev/null; do sleep 60; done; cd $D && bash queue_model.sh $R main run1 0,1 ) > $D/logs/chain_gen_arm_A_fixed.out 2>&1 &
wait
echo "$(date "+%F %T") arm A re-evaluation chains finished" >> $D/logs/done.txt
