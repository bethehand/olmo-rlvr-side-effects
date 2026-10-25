#!/bin/bash
# Usage: run_gen.sh MODEL_REPO REVISION TASK RUN GPUS [LIMIT]
#   TASK: heldout | ifeval | gsm8k;  RUN: runN uses seed N (override with SEED=...)
#   e.g. run_gen.sh allenai/Olmo-3-7B-Think-SFT main gsm8k run1 2,3
# Same software as the 805-prompt runs (safety-eval's environment: vLLM 0.11.0, transformers 4.57.1).
# Env: TEMPLATE_FROM=repo to use another model's chat template; HF_HUB_OFFLINE=1 to serve a private model already in the hub cache without a token; DTYPE, GPU_MEM, MAX_SEQS as in run_length.sh.
set -u
MODEL=$1 REV=$2 TASK=$3 RUN=$4 GPUS=$5 LIMIT=${6:-}
W=/mnt/nvme0/looped/oi_work; source $W/env.sh >/dev/null
E=$W/experiments/2026-10-23_dpo-arms-eval; cd $E; mkdir -p outputs/$TASK logs
NAME=$(echo "${MODEL#allenai/}" | tr "/" "_")@$REV
[ -z "${SEED:-}" ] && case "$RUN" in run[0-9]*) SEED=${RUN#run};; esac
export CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=/tmp/ray_$USER
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
# wait up to 8 h for the GPUs to be free, e.g. while another run of ours finishes (never touches other users' processes)
for i in $(seq 1 2880); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=",$GPUS," 'index(g, "," $1 ",") && $2 > 1000' | wc -l)
  [ "$busy" -eq 0 ] && break; sleep 10
done
PR=""; [ "$TASK" = heldout ] && PR="--prompts heldout_prompts.jsonl"
t0=$(date +%s)
$W/safety-eval/.venv-se/bin/python generate_prompts.py --model $MODEL --revision $REV --task $TASK --tp $NG ${MAXTOK:+--max_new_tokens $MAXTOK} ${TEMP:+--temperature $TEMP} \
  $PR --out outputs/$TASK/${NAME}_$RUN.jsonl ${LIMIT:+--limit $LIMIT} \
  ${DTYPE:+--dtype $DTYPE} ${TEMPLATE_FROM:+--template_from $TEMPLATE_FROM} ${GPU_MEM:+--gpu_mem $GPU_MEM} ${MAX_SEQS:+--max_num_seqs $MAX_SEQS} ${SEED:+--seed $SEED} \
  >> logs/${TASK}_${NAME}_$RUN.log 2>&1
rc=$?
n=$(cat outputs/$TASK/${NAME}_$RUN.jsonl 2>/dev/null | wc -l)
echo "$(date '+%F %T') $TASK $NAME $RUN exit=$rc rows=$n seconds=$(( $(date +%s) - t0 ))" >> logs/done.txt
