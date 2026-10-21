#!/bin/bash
# GS01 driver: verbosity reports for the 14 AlpacaEval runs (2026-10-18_length outputs), one report per model family,
# reference = that family's SFT run1; plus one table over every 805-prompt file. CPU only.
set -u
W=/mnt/nvme0/looped/oi_work; O=$W/experiments/2026-10-18_length/outputs; E=$W/experiments/2026-10-23_verbosity-report
mkdir -p $E/results; source $W/env.sh >/dev/null; cd $W/safety-eval && source .venv-se/bin/activate
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
TOK=allenai/Olmo-3-7B-Think
T="python $E/verbosity_report.py --tokenizer $TOK --per-dataset"
cd $O
t0=$(date +%s)
$T --reference Olmo-3-7B-Think-SFT@main_run1.jsonl --out-md $E/results/7b_think.md --out-json $E/results/7b_think.json \
   Olmo-3-7B-Think-SFT@main_run1.jsonl Olmo-3-7B-Think-SFT@main_ctrl1.jsonl Olmo-3-7B-Think-DPO@main_run1.jsonl Olmo-3-7B-Think-DPO@main_run2.jsonl Olmo-3-7B-Think@main_run1.jsonl > $E/results/7b_think.log 2>&1; echo "7b_think exit=$? $(( $(date +%s) - t0 ))s"
$T --reference Olmo-3-32B-Think-SFT@main_run1.jsonl --out-md $E/results/32b_think.md --out-json $E/results/32b_think.json \
   Olmo-3-32B-Think-SFT@main_run1.jsonl Olmo-3-32B-Think-SFT@main_rel1.jsonl Olmo-3-32B-Think-SFT@main_rob1.jsonl Olmo-3-32B-Think-DPO@main_run1.jsonl Olmo-3-32B-Think-DPO@main_rel1.jsonl Olmo-3-32B-Think@main_run1.jsonl Olmo-3-32B-Think@main_rel1.jsonl Olmo-3.1-32B-Think@main_run1.jsonl Olmo-3.1-32B-Think@main_ctrl1.jsonl > $E/results/32b_think.log 2>&1; echo "32b_think exit=$? $(( $(date +%s) - t0 ))s"
$T --reference Olmo-3.1-32B-Instruct-SFT@main_run1.jsonl --out-md $E/results/32b_instruct.md --out-json $E/results/32b_instruct.json \
   Olmo-3.1-32B-Instruct-SFT@main_run1.jsonl Olmo-3.1-32B-Instruct-SFT@main_rel1.jsonl Olmo-3.1-32B-Instruct-DPO@main_run1.jsonl Olmo-3.1-32B-Instruct-DPO@main_rel1.jsonl Olmo-3.1-32B-Instruct-DPO@main_rob1.jsonl Olmo-3.1-32B-Instruct@main_run1.jsonl Olmo-3.1-32B-Instruct@main_ctrl1.jsonl > $E/results/32b_instruct.log 2>&1; echo "32b_instruct exit=$? $(( $(date +%s) - t0 ))s"
echo ALL_DONE > $E/results/DONE
