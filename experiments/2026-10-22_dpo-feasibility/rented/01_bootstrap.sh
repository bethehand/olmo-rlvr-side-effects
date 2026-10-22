#!/bin/bash
# One-time setup, run on a cheap pod attached to the persistent volume (no H100 needed for this step).
# Requires in the shell: export HF_TOKEN=<token>   (never written to disk by these scripts)
#                        export HF_USER=<account>  (or edit env_node.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; source ./env_node.sh
: "${HF_TOKEN:?export HF_TOKEN=<your Hugging Face token> first}"
[ "$HF_USER" != CHANGE_ME ] || { echo "export HF_USER=<account> first"; exit 2; }
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git curl; }
command -v uv  >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$OI/.git" ] || git clone -q https://github.com/allenai/open-instruct "$OI"
cd "$OI"; git fetch -q origin; git checkout -q ff32794a3          # same commit as the GS01 feasibility runs
uv python install 3.12 -q
uv sync -q                                                         # default groups dev+cuda12: torch 2.10.0+cu128, deepspeed 0.18.4, prebuilt flash-attn 2.8.3 (uv.lock)
source .venv/bin/activate
python - <<'PY'
import torch, flash_attn, deepspeed, transformers, accelerate
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("flash_attn", flash_attn.__version__, "deepspeed", deepspeed.__version__,
      "transformers", transformers.__version__, "accelerate", accelerate.__version__)
PY
hf download allenai/Olmo-3-7B-Think-SFT --quiet >/dev/null && echo "model cached in $HF_HUB_CACHE"
if hf download "$HF_USER/dolci-think-dpo-arms" --repo-type dataset --local-dir "$ARMS" --quiet >/dev/null 2>&1 \
   && [ "$(wc -l < "$ARMS/arm_A.jsonl" 2>/dev/null | tr -d ' ')" = 20000 ] && [ "$(wc -l < "$ARMS/arm_B.jsonl" 2>/dev/null | tr -d ' ')" = 20000 ]; then
  echo "arms downloaded from $HF_USER/dolci-think-dpo-arms"
else
  echo "private arms repo unavailable or incomplete: rebuilding A and B from the public dataset and the id lists"
  python "$HERE/rebuild_arms_from_ids.py" "$ARMS" A B
fi
ls -la "$ARMS"; df -h "$WS" | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
echo "bootstrap done $(date -u +%FT%TZ) open-instruct=$(git rev-parse --short HEAD)" | tee -a "$LOGS/bootstrap.txt"
