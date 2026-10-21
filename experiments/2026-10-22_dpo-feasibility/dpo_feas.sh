#!/bin/bash
# Feasibility run of Ai2's 7B Think DPO recipe on 4x RTX 4090 (ZeRO-3 with CPU offload). Usage: dpo_feas.sh smoke|full16k
set -u
W=/mnt/nvme0/looped/oi_work; E=$W/experiments/2026-10-23_dpo-feasibility; OI=$W/open-instruct
source $W/env.sh >/dev/null; cd $OI; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1,2,3
export REFERENCE_LOGPROBS_CACHE_PATH=$E/ref_cache   # the trainer's default is an Ai2 cluster path (/weka/...)
MODE=$1
EXTRA=""; DSCONF=configs/ds_configs/stage3_offloading_accelerate.conf
case $MODE in
  smoke) N=256; L=8192;;
  smoke_sep) N=256; L=8192; EXTRA="--no_concatenated_forward";;   # chosen and rejected in separate forward passes: halves the logits peak
  smoke4k_sep) N=256; L=4096; EXTRA="--no_concatenated_forward";;
  smoke8k_lowlive) N=256; L=8192; DSCONF=$E/ds_stage3_offload_lowlive.conf; export PYTORCH_ALLOC_CONF=expandable_segments:True;;   # fewer live gathered params + no fragmentation
  smoke16k_lowlive) N=256; L=16384; DSCONF=$E/ds_stage3_offload_lowlive.conf; export PYTORCH_ALLOC_CONF=expandable_segments:True;;
  full16k) N=512; L=16384;;
  *) echo "mode?"; exit 2;;
esac
OUT=$E/out_$MODE; LOG=$E/logs; mkdir -p $OUT $LOG $E/cache
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 5 > $LOG/${MODE}_gpu.csv 2>/dev/null &
GPUPID=$!
( while true; do echo "$(date +%T) $(free -g | awk 'NR==2{print $3" GB used, "$7" GB avail"}')"; sleep 10; done ) > $LOG/${MODE}_ram.log 2>&1 &
RAMPID=$!
t0=$(date +%s)
accelerate launch --mixed_precision bf16 --num_processes 4 --use_deepspeed \
  --deepspeed_config_file $DSCONF \
  open_instruct/dpo_tune_cache.py \
  --exp_name dpo_feas_$MODE --do_not_randomize_output_dir --output_dir $OUT \
  --model_name_or_path allenai/Olmo-3-7B-Think-SFT --tokenizer_name allenai/Olmo-3-7B-Think-SFT \
  --mixer_list allenai/Dolci-Think-DPO-7B $N --mixer_list_splits train \
  --cache_mode local --local_cache_dir $E/cache \
  --chat_template_name olmo_thinker --max_seq_length $L \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 8e-8 --lr_scheduler_type linear --warmup_ratio 0.0 --weight_decay 0.0 --num_epochs 1 \
  --loss_type dpo_norm --beta 5 --activation_memory_budget 0.5 --attn_implementation flash_2 \
  --logging_steps 1 --seed 1 $EXTRA \
  --no_push_to_hub --no_try_launch_beaker_eval_jobs --no_try_auto_save_to_beaker \
  > $LOG/${MODE}_train.log 2>&1; rc=$?
kill $GPUPID $RAMPID 2>/dev/null
echo "$MODE exit=$rc seconds=$(( $(date +%s) - t0 )) finished=$(date +%T)" | tee -a $LOG/done.txt
