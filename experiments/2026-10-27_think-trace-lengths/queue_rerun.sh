#!/bin/bash
# The four full runs: the two 7B models in parallel on two GPUs each, then the two 32B models on all four GPUs.
E=/mnt/nvme0/looped/oi_work/experiments/2026-10-27_think-trace-lengths; cd $E
bash run_rerun.sh allenai/Olmo-3-7B-Think-SFT 0,1 7b_think_sft &
bash run_rerun.sh allenai/Olmo-3-7B-Think-DPO 2,3 7b_think_dpo &
wait
bash run_rerun.sh allenai/Olmo-3-32B-Think-SFT 0,1,2,3 32b_think_sft --max_num_seqs 64
bash run_rerun.sh allenai/Olmo-3.1-32B-Think 0,1,2,3 31_32b_think --max_num_seqs 64
echo "$(date '+%F %T') QUEUE_DONE" >> logs/h045_rerun_done.txt
