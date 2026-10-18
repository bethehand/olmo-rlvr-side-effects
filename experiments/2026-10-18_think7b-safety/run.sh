#!/bin/bash
# Ai2's own safety protocol for Olmo 3 Think, run unchanged: OLMES `safety_reasoning::olmo3` calls safety-eval's
# run_all_generation_benchmarks.py with the model's own chat template ("hf": default system prompt, the reply
# starts inside <think>), temperature 0.7, top_p 0.95, up to 32K new tokens, two GPUs per task, and judges that
# read only the final answer (text after </think>).
# Usage: run.sh smoke|full MODEL_REPO RUN_NAME [GPUS, default 0,1,2,3] [SET: issue1964 (default) | table52]
#   issue1964 = the three tasks open-instruct #1964 asks for; table52 = all nine tasks Table 52 reports.
#   smoke = 5 prompts per task and at most 8K new tokens, to check the pipeline end to end.
# A rerun with the same RUN_NAME skips tasks that already have a metrics file, so an interrupted run resumes.
set -u
MODE=$1 MODEL=$2 RUN=$3 GPUS=${4:-0,1,2,3} SET=${5:-issue1964}
W=/mnt/nvme0/looped/oi_work
source $W/env.sh >/dev/null
E=$W/experiments/2026-10-18_think7b-safety
NAME=${MODEL#allenai/}
OUT=$E/outputs/$NAME/$RUN; LOG=$E/logs/${NAME}_$RUN.log; mkdir -p $OUT $E/logs
# The nine tasks of OLMES `safety_reasoning::olmo3` that Table 52 reports (BBQ and WMDP are left out), in three groups:
# the three open-instruct #1964 asks for; five more; StrongReject on its own (below).
ISSUE1964='["xstest:wildguard_reasoning_answer","wildjailbreak:benign_wildguard_reasoning_answer","wildguardtest:wildguard_reasoning_answer"]'
TABLE52_REST='["harmbench:wildguard_reasoning_answer","do_anything_now:wildguard_reasoning_answer","trustllm_jailbreaktrigger:wildguard_reasoning_answer","wildjailbreak:harmful_wildguard_reasoning_answer","toxigen:tiny_reasoning_answer"]'
SR='["strongreject:logprobs_reasoning_answer"]'
EXTRA=""
[ "$MODE" = smoke ] && EXTRA='--limit true --hparam_overrides {"max_new_tokens":8192}'
# Seed: runN uses N (override with SEED=...), so repeated runs are independent samples. Without a seed, vLLM V1 uses 0
# and a rerun can come out as a byte-identical copy (seen on 2026-10-18).
[ -z "${SEED:-}" ] && case "$RUN" in run[0-9]*) SEED=${RUN#run};; esac
cd $W/safety-eval && source .venv-se/bin/activate   # the runner starts each task with a bare `python`
export PYTHONPATH=$W/safety-eval   # OLMES pip-installs safety-eval in editable mode; this is the same import path
export OPENAI_API_KEY=placeholder-not-used   # safety-eval builds an OpenAI client at import time; these tasks never call it
                                            # (all judges are local), and a real call would fail loudly with this key
export RAY_TMPDIR=/tmp/ray_$USER CUDA_VISIBLE_DEVICES=$GPUS
export SAFETY_EVAL_GPU_MEM=0.85   # read by our patch (safety-eval-local-changes.patch); vLLM's default 0.9 OOMs at start-up on 24 GB cards
export SAFETY_EVAL_SEED=${SEED:-}   # read by the same patch; empty = vLLM default (0)
tasks() {
  python evaluation/run_all_generation_benchmarks.py \
    --model_name_or_path $MODEL --model_input_template_path_or_name hf --hf_revision main \
    --report_output_path $OUT/metrics.json --save_individual_results_path $OUT/all.json \
    --min_gpus_per_task 2 --task_names "$1" $EXTRA >> $LOG 2>&1
}
t0=$(date +%s)
tasks "$ISSUE1964"; rc=$?; rc2=-
if [ "$SET" = table52 ]; then
  tasks "$TABLE52_REST"; rc=$((rc + $?))
  # StrongReject's judge merges a LoRA into gemma-2b and saves it to one fixed folder (safety-eval/evaluation/.tmp/),
  # so two runs must not build it at the same time: one StrongReject task at a time on this machine.
  ( flock 8; tasks "$SR" ) 8>$W/queues/strongreject.lock; rc2=$?
fi
echo "$NAME $RUN $SET seed=${SEED:-default} exit=$rc/$rc2 seconds=$(( $(date +%s) - t0 ))" >> $E/logs/done.txt
# If a task of this set has no metrics file (the run crashed, ran out of memory, ...), queue the same run again at
# the end of the queue; finished tasks are skipped on the rerun. At most two retries; smoke runs are not retried.
want="$ISSUE1964"; [ "$SET" = table52 ] && want="$ISSUE1964,$TABLE52_REST,$SR"
missing=0; for t in $(echo "$want" | tr -d '[]"' | tr ',' ' '); do [ -f "$OUT/metrics.json.$t" ] || missing=$((missing + 1)); done
if [ "$MODE" = full ] && [ $missing -gt 0 ] && [ "${RETRY:-0}" -lt 2 ]; then
  ( flock 9; echo "RETRY=$(( ${RETRY:-0} + 1 )) SEED=${SEED:-} bash $E/run.sh full $MODEL $RUN \$GPUS $SET" >> $W/queues/safety ) 9>$W/queues/safety.lock
  echo "$NAME $RUN $SET: $missing task(s) missing, queued retry $(( ${RETRY:-0} + 1 ))" >> $E/logs/done.txt
fi
