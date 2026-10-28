#!/bin/bash
# Temperature-1.0 reruns of the same 22 turns, same seeds as the T=0.6 runs (top_p stays 0.95, templates and budgets unchanged):
# both 7B models in parallel on two GPUs each, first at max_tokens 8,192, then at 32,768. Motivated by the maintainers' reading on
# open-instruct #2080 (Ai2 evaluates these checkpoints at T=1.0, top-p 0.95; the issue author's run and our T=0.6 table used 0.6).
E=/mnt/nvme0/looped/oi_work/experiments/2026-10-27_think-trace-lengths; cd $E
bash run_rerun.sh allenai/Olmo-3-7B-Think-SFT 0,1 7b_think_sft_t10 --temperature 1.0 &
bash run_rerun.sh allenai/Olmo-3-7B-Think-DPO 2,3 7b_think_dpo_t10 --temperature 1.0 &
wait
bash run_rerun.sh allenai/Olmo-3-7B-Think-SFT 0,1 7b_think_sft_t10_32k --temperature 1.0 --max_tokens 32768 --max_model_len 40960 &
bash run_rerun.sh allenai/Olmo-3-7B-Think-DPO 2,3 7b_think_dpo_t10_32k --temperature 1.0 --max_tokens 32768 --max_model_len 40960 &
wait
echo "$(date '+%F %T') QUEUE_T10_DONE" >> logs/h045_rerun_done.txt
