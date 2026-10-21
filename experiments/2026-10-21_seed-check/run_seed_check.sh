#!/bin/bash
# GS01 driver: apply the PR patch in a separate worktree, run the unit tests, then the seed comparison on one GPU.
# Usage: bash run_seed_check.sh GPU MODEL   (writes to $W/experiments/2026-10-21_seed-check/)
set -u
GPU=$1 MODEL=$2
W=/mnt/nvme0/looped/oi_work
E=$W/experiments/2026-10-21_seed-check; mkdir -p $E/outputs $E/logs
source $W/env.sh >/dev/null
cd $W/safety-eval && source .venv-se/bin/activate
export OPENAI_API_KEY=placeholder-not-used RAY_TMPDIR=/tmp/ray_$USER PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
# 1. A clean worktree at upstream main (060cc90) plus the PR patch; the patched working copy stays untouched.
PR=$W/safety-eval-pr
if [ ! -d $PR ]; then git worktree add --detach $PR 060cc90 >/dev/null && (cd $PR && git apply $E/0001-*.patch && git log --oneline -1 && git status --short | head -3); fi
cd $PR; export PYTHONPATH=$PR:$E/pytest_lib   # pytest copied from the Mac (pure Python), nothing installed into the venv
# 2. Unit tests in the real environment.
python -m pytest -q evaluation/tests/test_seed.py > $E/logs/pytest.log 2>&1; echo "pytest exit=$? $(tail -1 $E/logs/pytest.log)"
# 3. fire's parsing of the flag as run_all passes it.
python -c "import fire; fire.Fire(lambda seed=None: f'parsed seed={seed!r} type={type(seed).__name__}')" -- --seed None
python -c "import fire; fire.Fire(lambda seed=None: f'parsed seed={seed!r} type={type(seed).__name__}')" -- --seed 3
# 4. Seed comparison: six engines in a row on one GPU.
export CUDA_VISIBLE_DEVICES=$GPU
t0=$(date +%s)
for run in none_a none_b s0 s1_a s1_b s2; do
  seed=${run%%_*}; seed=${seed#s}; [ "$seed" = none ] || true
  python $E/seed_check.py $MODEL $E/outputs/$run.json $seed > $E/logs/$run.log 2>&1; echo "$run exit=$? $(( $(date +%s) - t0 ))s $(tail -1 $E/logs/$run.log | cut -c1-120)"
done
python $E/compare_seed_runs.py $E/outputs/none_a.json $E/outputs/none_b.json $E/outputs/s0.json $E/outputs/s1_a.json $E/outputs/s1_b.json $E/outputs/s2.json | tee $E/comparison.txt
echo ALL_DONE
