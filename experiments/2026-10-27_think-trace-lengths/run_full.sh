#!/bin/bash
# Waits for download_shards.py to finish, then runs trace_lengths.py over all sampled shards with 24 worker processes.
W=/mnt/nvme0/looped/oi_work/experiments/2026-10-27_think-trace-lengths
PY=/mnt/nvme0/looped/oi_work/open-instruct/.venv/bin/python
cd "$W" || exit 1
until grep -q DOWNLOAD_DONE logs/download.log 2>/dev/null; do sleep 10; done
echo "$(date '+%H:%M:%S') download done, starting trace_lengths.py" > logs/trace_lengths.log
export HF_HUB_CACHE=/mnt/nvme0/looped/oi_work/hf-hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4
"$PY" trace_lengths.py data results --procs 24 >> logs/trace_lengths.log 2>&1
echo "FULL_DONE exit=$?" >> logs/trace_lengths.log
