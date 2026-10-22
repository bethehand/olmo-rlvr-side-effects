#!/bin/bash
# Timing and memory test before committing to full runs. Runs the exact training command on 64 and
# 256 pairs of arm A (both subsets include arm A's eight longest pairs, so the 16K cap is exercised)
# and projects the time and cost of a full 20,000-pair run from the difference, with fixed overhead
# (model load, tokenization, final save) removed.
# Usage: 02_timing_test.sh [concat|sep]      PRICE_PER_HOUR=15.96 is the default (4x H100 SXM, Secure Cloud)
set -uo pipefail
cd "$(dirname "$0")"; source ./env_node.sh; source "$VENV/bin/activate"; cd "$OI"
MODE=${1:-concat}; EXTRA=""; [ "$MODE" = sep ] && EXTRA="--no_concatenated_forward"
PRICE=${PRICE_PER_HOUR:-15.96}
[ "$NGPU" -gt 0 ] || { echo "no GPU visible"; exit 2; }
ACC=$((32 / NGPU))        # effective batch 32 on any GPU count (see README for why not Ai2's 128)
# --activation_memory_budget < 1 just turns on HF per-layer gradient checkpointing for this model (dpo_tune_cache.py:377); the value itself is unused
# DS_CONF: DeepSpeed config; the pipeline switches to the optimizer-offload config (ds_stage3_offload_optimizer.conf) when the no-offload peak does not fit
run_one() {               # $1 = number of pairs (64 | 256); prints "rc seconds peak_mib" as its last line
  local n=$1 tag=timing_${MODE}_$1 outd=$OUT/timing_${MODE}_$1
  rm -rf "$outd"
  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 5 > "$LOGS/${tag}_gpu.csv" 2>/dev/null &
  local gp=$! t0=$(date +%s)
  accelerate launch --mixed_precision bf16 --num_processes "$NGPU" --use_deepspeed \
    --deepspeed_config_file "${DS_CONF:-configs/ds_configs/stage3_no_offloading_accelerate.conf}" \
    open_instruct/dpo_tune_cache.py \
    --exp_name "$tag" --do_not_randomize_output_dir --output_dir "$outd" \
    --model_name_or_path allenai/Olmo-3-7B-Think-SFT --tokenizer_name allenai/Olmo-3-7B-Think-SFT \
    --mixer_list "$ARMS/arm_A_timing${n}.jsonl" 1.0 --mixer_list_splits train \
    --cache_mode local --local_cache_dir "$CACHE" \
    --chat_template_name olmo_thinker --max_seq_length 16384 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACC" \
    --learning_rate 8e-8 --lr_scheduler_type linear --warmup_ratio 0.0 --weight_decay 0.0 --num_epochs 1 \
    --loss_type dpo_norm --beta 5 --activation_memory_budget 0.5 --attn_implementation flash_2 \
    --logging_steps 1 --seed 1 $EXTRA \
    --no_push_to_hub --no_try_launch_beaker_eval_jobs --no_try_auto_save_to_beaker \
    > "$LOGS/${tag}.log" 2>&1
  local rc=$?
  kill $gp 2>/dev/null
  local secs=$(( $(date +%s) - t0 ))
  local peak; peak=$(awk -F', ' '{gsub(/ MiB/,"",$3); if ($3+0>m) m=$3+0} END{print m+0}' "$LOGS/${tag}_gpu.csv")
  echo "$tag exit=$rc wall_s=$secs peak_gpu_mib=$peak" >> "$LOGS/timing_summary.txt"
  rm -rf "$outd"          # the 14 GB model written at the end of a test run is not needed
  echo "$rc $secs $peak"
}
read -r rc64 t64 p64 < <(run_one 64 | tail -1)
[ "$rc64" = 0 ] || { echo "64-pair run failed (exit $rc64), see $LOGS/timing_${MODE}_64.log"; tail -5 "$LOGS/timing_${MODE}_64.log"; exit 1; }
read -r rc256 t256 p256 < <(run_one 256 | tail -1)
[ "$rc256" = 0 ] || { echo "256-pair run failed (exit $rc256), see $LOGS/timing_${MODE}_256.log"; tail -5 "$LOGS/timing_${MODE}_256.log"; exit 1; }
python - "$t64" "$t256" "$p64" "$p256" "$PRICE" "$MODE" "$NGPU" <<'PY' | tee -a "$LOGS/timing_summary.txt"
import sys
t64, t256, p64, p256 = (float(x) for x in sys.argv[1:5]); price, mode, ngpu = float(sys.argv[5]), sys.argv[6], sys.argv[7]
per_pair = (t256 - t64) / 192                # seconds per training pair, fixed overhead cancelled
fixed = t64 - 64 * per_pair                  # model load + tokenization + final save
full = fixed + 20000 * per_pair
h = full / 3600
print(f"mode={mode} gpus={ngpu}: {per_pair:.2f} s/pair, fixed {fixed/60:.1f} min -> one 20,000-pair run ~{h:.1f} h, ${h*price:.0f} at ${price}/h; peak GPU memory {max(p64,p256)} MiB of 81,559")
print(f"A+B ~{2*h:.1f} h ${2*h*price:.0f} | +C ~{3*h:.1f} h ${3*h*price:.0f} | +2 extra seeds ~{5*h:.1f} h ${5*h*price:.0f}")
PY
