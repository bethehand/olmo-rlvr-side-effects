#!/bin/bash
# Final pass: wait for the Clean-v1 footer scan, fetch up to two Clean-v1 shards that contain Wildchat rows (if the scan finds any),
# recompute trace_lengths.py over every downloaded shard, run the composition checks and build the markdown tables.
W=/mnt/nvme0/looped/oi_work/experiments/2026-10-27_think-trace-lengths
PY=/mnt/nvme0/looped/oi_work/open-instruct/.venv/bin/python
cd "$W" || exit 1
export HF_HUB_CACHE=/mnt/nvme0/looped/oi_work/hf-hub
until grep -q FAST_SCAN_DONE logs/fast_scan_clean_v1.log 2>/dev/null; do sleep 10; done
WC=$(grep -E "^[0-9:]+ shard" logs/fast_scan_clean_v1.log | grep "Wildchat" | sed -E 's/^[0-9:]+ shard +([0-9]+) .*/\1/' | head -2 | tr '\n' ' ')
echo "$(date '+%H:%M:%S') Clean-v1 shards with Wildchat rows (first two): [$WC]" > logs/run_final.log
if [ -n "$WC" ]; then "$PY" download_extra.py allenai/Dolci-Think-SFT-Clean-v1 128 clean_v1 $WC >> logs/run_final.log 2>&1; fi
mv logs/trace_lengths.log logs/trace_lengths_attempt2_before_extra_shards.log 2>/dev/null
mv results/trace_lengths.txt results/trace_lengths_attempt2_before_extra_shards.txt 2>/dev/null
mv results/trace_lengths.json results/trace_lengths_attempt2_before_extra_shards.json 2>/dev/null
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4
"$PY" trace_lengths.py data results --procs 32 > logs/trace_lengths.log 2>&1
echo "trace_lengths exit=$?" >> logs/run_final.log
"$PY" composition_check.py results/rows_clean_v1.parquet 32b results/composition_clean_v1.txt > /dev/null 2>> logs/run_final.log
"$PY" composition_check.py results/rows_think_sft_7b.parquet 7b results/composition_think_sft_7b.txt > /dev/null 2>> logs/run_final.log
"$PY" shard_sources_fast.py --help > /dev/null 2>&1
"$PY" make_tables.py results > results/tables.md 2>> logs/run_final.log
echo "FINAL_DONE exit=$?" >> logs/run_final.log
