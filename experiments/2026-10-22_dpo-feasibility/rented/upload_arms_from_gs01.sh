#!/bin/bash
# Run on the machine that holds the arms (our shared GPU box). Uploads the training arms to a private
# dataset repo so the rented node can pull them with `hf download`. Needs HF_USER and a logged-in
# write token (hf auth login); nothing secret is stored in this file.
set -euo pipefail
: "${HF_USER:?export HF_USER=<account>}"
cd /mnt/nvme0/looped/oi_work/experiments/2026-10-22_dpo-feasibility/arms
hf upload "$HF_USER/dolci-think-dpo-arms" . . --repo-type dataset --private \
  --include "arm_A.jsonl" --include "arm_B.jsonl" --include "arm_A_timing64.jsonl" --include "arm_A_timing256.jsonl" \
  --include "*_ids.json" \
  --commit-message "Training arms A and B (20,000 pairs each, build_arms.py seed 1) and the 64/256-pair timing subsets of A"
echo "uploaded to https://huggingface.co/datasets/$HF_USER/dolci-think-dpo-arms (private)"
