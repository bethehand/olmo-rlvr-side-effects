#!/bin/bash
# Runs on GS01, called by watch_arm_push.sh once an arm's weights are in the hub cache (everything is served offline from it).
# Starts two background chains: the 805-prompt length run on LEN_GPUS, and the held-out / IFEval / GSM8K runs on GEN_GPUS.
# With a 4th argument "control", each chain then repeats its runs for the start model allenai/Olmo-3-7B-Think-SFT with the
# arm's chat template (run name tmplA, seed 1), so the arm and its start model are compared under one template.
# The run scripts wait up to 4 h for their GPUs, so a chain may queue behind runs that are still going.
# Usage: [MAXTOK=32768] [LIMIT_ifeval=N] [LIMIT_gsm8k=N] start_arm_eval.sh REPO LEN_GPUS GEN_GPUS [control]
set -u
REPO=$1 LG=$2 GG=$3 CTRL=${4:-}
# Generation cap and GSM8K size for the arms (and the control, so it stays comparable). Arm A at the 32K cap ran away into
# repetitive loops on a large share of prompts (2026-10-23), which made the 32K runs infeasible on 2 GPUs; the baselines
# finish under 8K tokens on 98% of prompts, and a baseline output longer than the cap is counted as a cap hit in the analysis.
# Pre-registered protocol: 32K cap, full IFEval and GSM8K (the 8K cap and 200-prompt limits used for arm A on 2026-10-23 were a
# response to its apparent collapse, which turned out to be the config artifact that fix_tf5_compat.py repairs).
export MAXTOK=${MAXTOK:-32768} LIMIT_ifeval=${LIMIT_ifeval:-} LIMIT_gsm8k=${LIMIT_gsm8k:-}
W=/mnt/nvme0/looped/oi_work; E=$W/experiments; L=$E/2026-10-23_dpo-arms-eval/logs; mkdir -p $L
TAG=$(basename $REPO)
source $W/env.sh >/dev/null
export HF_HUB_OFFLINE=1
# the arms were saved by transformers 5.x; make the cached tokenizer_config.json loadable by transformers 4.57 / vLLM 0.11
$W/safety-eval/.venv-se/bin/python $E/2026-10-23_dpo-arms-eval/fix_tf5_compat.py $REPO --check_against allenai/Olmo-3-7B-Think-SFT || exit 1
LEN="cd $E/2026-10-18_length && bash run_length.sh $REPO main run1 $LG '' $MAXTOK"
GEN="cd $E/2026-10-23_dpo-arms-eval && bash queue_model.sh $REPO main run1 $GG"
if [ "$CTRL" = control ]; then
  LEN="$LEN; TEMPLATE_FROM=$REPO SEED=1 bash run_length.sh allenai/Olmo-3-7B-Think-SFT main tmplA $LG '' $MAXTOK"
  GEN="$GEN; TEMPLATE_FROM=$REPO SEED=1 bash queue_model.sh allenai/Olmo-3-7B-Think-SFT main tmplA $GG"
fi
nohup bash -c "$LEN" > $L/chain_len_$TAG.out 2>&1 < /dev/null &
nohup bash -c "$GEN" > $L/chain_gen_$TAG.out 2>&1 < /dev/null &
sleep 1
echo "$(date '+%F %T') started chains for $REPO (cap $MAXTOK tokens, ifeval limit $LIMIT_ifeval, gsm8k limit $LIMIT_gsm8k): length run on GPUs $LG, heldout/ifeval/gsm8k on GPUs $GG${CTRL:+, then the Think-SFT control with the template of $TAG}" | tee -a $L/done.txt
pgrep -u $USER -af 'run_length|queue_model' | grep -v pgrep
