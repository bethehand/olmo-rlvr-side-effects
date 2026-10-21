# safety-eval's vLLM seed: does `--seed` do what the PR says? (2026-10-21)

One check behind the `--seed` PR to [allenai/safety-eval](https://github.com/allenai/safety-eval) (link added once the PR is open). Claim: without a seed, vLLM's V1 engine uses 0 every time, so a rerun over the same prompts is a byte-identical copy; a seed per run gives independent samples; the same seed reproduces a run.

**Setup.** One RTX 4090, vLLM 0.11.0, safety-eval `main` (060cc90) plus the PR patch in a separate worktree. `allenai/Olmo-3-7B-Think-SFT` with its own chat template, 8 benign prompts (listed in `seed_check.py`), temperature 0.7, top_p 0.95, 256 new tokens, one sample per prompt. Each run is a fresh engine in its own process through the patched `create_and_inference_with_vllm`; the only addition is memory settings for a 24 GB card (`gpu_memory_utilization` 0.85, `max_model_len` 4096, `max_num_seqs` 16). Six runs: no seed ×2, `--seed 0`, `--seed 1` ×2, `--seed 2`; 40–75 s each.

**Result** (`results/comparison.txt`; identical outputs out of 8 prompts):

| Pair | Identical |
|---|---|
| no seed vs no seed | 8/8 |
| no seed vs `--seed 0` | 8/8 |
| `--seed 1` vs `--seed 1` | 8/8 |
| `--seed 1` vs `--seed 2` | 0/8 |
| no seed (= 0) vs `--seed 1` or `--seed 2` | 0/8 |

So the default really is seed 0, a seed reproduces a run, and different seeds give different samples. The PR's 10 unit tests pass in the same environment (`results/pytest_output.txt`).

**Files.** `run_seed_check.sh` (driver: worktree + patch, pytest, six engine runs, comparison), `seed_check.py` (one run, writes prompts and outputs), `compare_seed_runs.py` (pairwise identical counts), `results/*.json` (one file per run), `results/comparison.txt`.
