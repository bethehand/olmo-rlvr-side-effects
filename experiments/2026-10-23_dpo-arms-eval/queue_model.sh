#!/bin/bash
# Usage: [MAXTOK=8192] [LIMIT_gsm8k=300] queue_model.sh MODEL REV RUN GPUS   -- runs heldout, ifeval, gsm8k for one model in sequence on the given GPUs
M=$1 R=$2 RUN=$3 G=$4; cd "$(dirname "$0")"
# LIMIT_<task>=N (env) runs only the first N prompts of that set (a later run without the limit resumes and completes it)
for t in heldout ifeval gsm8k; do lim="LIMIT_$t"; bash run_gen.sh "$M" "$R" $t $RUN $G ${!lim:-}; done
echo "$(date "+%F %T") queue done $M $RUN" >> logs/done.txt
