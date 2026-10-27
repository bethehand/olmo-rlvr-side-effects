#!/bin/bash
# The 32K-budget reruns of the same 22 turns: 7B Think-SFT and 7B Think-DPO in parallel on two GPUs each, same seeds as the 8K runs.
E=/mnt/nvme0/looped/oi_work/experiments/2026-10-27_think-trace-lengths; cd $E
bash run_rerun.sh allenai/Olmo-3-7B-Think-SFT 0,1 7b_think_sft_32k --max_tokens 32768 --max_model_len 40960 &
bash run_rerun.sh allenai/Olmo-3-7B-Think-DPO 2,3 7b_think_dpo_32k --max_tokens 32768 --max_model_len 40960 &
wait
echo "$(date '+%F %T') QUEUE32K_DONE" >> logs/h045_rerun_done.txt
