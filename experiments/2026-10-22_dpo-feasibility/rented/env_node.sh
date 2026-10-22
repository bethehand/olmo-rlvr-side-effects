#!/bin/bash
# Shared paths for the rented node. Everything lives on the persistent volume (default /workspace on
# RunPod) so a pod can be stopped and recreated without redoing setup. Override WS if mounted elsewhere.
export WS=${WS:-/workspace}
export OI=$WS/open-instruct
export HF_HOME=$WS/hf
export HF_HUB_CACHE=$WS/hf/hub
export HF_DATASETS_CACHE=$WS/hf/datasets
export REFERENCE_LOGPROBS_CACHE_PATH=$WS/ref_cache   # the trainer's default is an Ai2 cluster path (/weka/...)
export ARMS=$WS/arms
export OUT=$WS/out
export LOGS=$WS/logs
export CACHE=$WS/cache
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
export PATH=$HOME/.local/bin:$PATH
# uv would otherwise put the interpreter and the environment under /root (the pod's ephemeral container
# disk) and leave only a symlink in the project; pin all three to the volume so a new pod reuses them.
export UV_CACHE_DIR=$WS/uv/cache
export UV_PYTHON_INSTALL_DIR=$WS/uv/python
export UV_PROJECT_ENVIRONMENT=$WS/uv/venv
export UV_LINK_MODE=copy
export VENV=$UV_PROJECT_ENVIRONMENT
# Hugging Face account that holds the private arms dataset and receives the trained models.
export HF_USER=${HF_USER:-CHANGE_ME}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
export NGPU=${NGPU:-0}
# DeepSpeed's CPU Adam (used when optimizer states are offloaded) is OpenMP-parallel; torchrun sets
# OMP_NUM_THREADS=1 unless it is already set, which made one optimizer step take ~20 s on 224 vCPUs.
_t=$(( $(nproc) / (NGPU > 0 ? NGPU : 1) )); [ "$_t" -gt 48 ] && _t=48; [ "$_t" -lt 1 ] && _t=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$_t}
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$REFERENCE_LOGPROBS_CACHE_PATH" "$ARMS" "$OUT" "$LOGS" "$CACHE"
