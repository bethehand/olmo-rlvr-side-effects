#!/bin/bash
# Full run of one arm with one seed under Ai2's 7B Think DPO recipe (deviations listed in README.md).
# Usage: 03_train_arm.sh A 1 [concat|sep]
# Re-running the same command after an interruption resumes from the last checkpoint in the output
# directory: the trainer picks it up by itself (get_last_checkpoint_path in open_instruct/utils.py).
set -uo pipefail
cd "$(dirname "$0")"; source ./env_node.sh; source "$VENV/bin/activate"; cd "$OI"
ARM=${1:?arm letter A|B|C|D}; SEED=${2:?training seed}; MODE=${3:-concat}
EXTRA=""; [ "$MODE" = sep ] && EXTRA="--no_concatenated_forward"
[ "$NGPU" -gt 0 ] || { echo "no GPU visible"; exit 2; }
ACC=$((32 / NGPU))
TAG=arm_${ARM}_seed${SEED}; OUTD=$OUT/$TAG; mkdir -p "$OUTD"
[ -f "$ARMS/arm_${ARM}.jsonl" ] || { echo "missing $ARMS/arm_${ARM}.jsonl"; exit 2; }
{ echo "start $(date -u +%FT%TZ) gpus=$NGPU mode=$MODE grad_accum=$ACC effective_batch=$((ACC*NGPU)) open-instruct=$(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1; } >> "$OUTD/run_info.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 30 > "$LOGS/${TAG}_gpu.csv" 2>/dev/null &
GP=$!; t0=$(date +%s)
accelerate launch --mixed_precision bf16 --num_processes "$NGPU" --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/dpo_tune_cache.py \
  --exp_name "$TAG" --do_not_randomize_output_dir --output_dir "$OUTD" \
  --model_name_or_path allenai/Olmo-3-7B-Think-SFT --tokenizer_name allenai/Olmo-3-7B-Think-SFT \
  --mixer_list "$ARMS/arm_${ARM}.jsonl" 1.0 --mixer_list_splits train \
  --cache_mode local --local_cache_dir "$CACHE" \
  --chat_template_name olmo_thinker --max_seq_length 16384 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACC" \
  --learning_rate 8e-8 --lr_scheduler_type linear --warmup_ratio 0.0 --weight_decay 0.0 --num_epochs 1 \
  --loss_type dpo_norm --beta 5 --activation_memory_budget 0.5 --attn_implementation flash_2 \
  --checkpointing_steps 100 --keep_last_n_checkpoints 1 \
  --logging_steps 10 --seed "$SEED" $EXTRA \
  --no_push_to_hub --no_try_launch_beaker_eval_jobs --no_try_auto_save_to_beaker \
  >> "$LOGS/${TAG}.log" 2>&1
rc=$?; kill $GP 2>/dev/null
echo "end $(date -u +%FT%TZ) exit=$rc wall_s=$(( $(date +%s) - t0 ))" | tee -a "$OUTD/run_info.txt" "$LOGS/done.txt"
[ "$rc" = 0 ] && ls "$OUTD" | head -20
exit $rc
