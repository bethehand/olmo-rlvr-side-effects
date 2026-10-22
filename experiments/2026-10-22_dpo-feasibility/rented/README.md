# Training the arms on a rented node

**Why.** The shared 4 × RTX 4090 box cannot run Ai2's 7B Think DPO recipe (five OOM attempts, see
[`../README.md`](../README.md)). These scripts run the same recipe on a rented single node with 4 or
8 × 80 GB GPUs (H100 SXM preferred) and hand the weights back for evaluation elsewhere.

**Order of runs (cost first).**
1. Arms A (original pairs) and B (chosen/rejected length ratio ≤ 1.5, matched to A's chosen-length
   deciles), one training seed each.
2. Arm C (whole-pool control matched to B on source × length) only if A and B differ.
3. A second seed for A and B only if the A–B gap is within about twice the evaluation noise.
4. D (Ai2's 100-token cap) is deferred.

Evaluation (AlpacaEval generations + verbosity report, held-out prompts, IFEval and GSM8K subsets)
runs on the shared box after the weights are uploaded, so the node is billed only for training.

## Recipe

Ai2's reference is `scripts/train/olmo3/7b_think_dpo.sh` in open-instruct at commit `ff32794a3`
(the commit these scripts pin).

| Setting | Ai2 | Here |
|---|---|---|
| Start model | Olmo 3 7B Think SFT | `allenai/Olmo-3-7B-Think-SFT` |
| Data | Dolci-Think-DPO-7B, 150,000 pairs, 1 epoch | 20,000-pair arms drawn from the same set by `../build_arms.py`, 1 epoch |
| Loss | `dpo_norm`, β = 5 | same |
| Learning rate / schedule / weight decay | 8e-8, linear, 0 | same |
| `max_seq_length` | 16,384 | same |
| Chat template | `olmo_thinker` | same |
| Effective batch | 1 × 4 accumulation × 32 GPUs = 128 | 1 × (32 / #GPUs) accumulation × #GPUs = 32 (30 on 3 GPUs) |
| Gradient checkpointing | `--gradient_checkpointing` (full per-layer) | same mechanism: `--activation_memory_budget 0.5` (< 1 turns on HF per-layer checkpointing for this model; the value is otherwise unused) |
| Optimizer states | on GPU | on CPU when fewer than 4 × 80 GB (`ds_stage3_offload_optimizer.conf`): measured on 3 × H100, the first backward peaks at 78 GB without offload |
| DeepSpeed | `stage3_no_offloading_accelerate.conf` | same |
| Hardware | 32 × H100 (4 nodes) | 4 × H100 SXM 80 GB (8 also works; accumulation adjusts) |

Two deviations, both reported with the results:

1. **20,000 pairs per arm instead of 150,000.** The arms are the experiment.
2. **Effective batch 32 instead of 128.** With 13 % of Ai2's data, batch 128 would give 156
   optimizer steps against Ai2's ≈ 1,170; batch 32 gives 625, keeping the number of updates in the
   same order of magnitude while every other hyperparameter stays as published.

## Memory

Full-parameter 7B DPO holds about 112 GB of weights, gradients, fp32 Adam moments and fp32 master
weights. ZeRO-3 over 4 GPUs leaves ≈ 28 GB of states per GPU plus ≈ 25–40 GB of activations and
fp32 logits at 16K. Two 80 GB GPUs are not enough (56 GB of states plus the rest exceeds 80 GB).
`sep` mode (`--no_concatenated_forward`) runs chosen and rejected in separate forward passes and
halves the logits peak at a small speed cost; the timing test prints peak memory so the mode can be
chosen before the full runs.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `env_node.sh` | node | shared paths on the persistent volume; `HF_USER` must be set |
| `01_bootstrap.sh` | cheap pod with the volume | install open-instruct @ `ff32794a3` with `uv sync`, download the model and the arms |
| `02_timing_test.sh [concat\|sep]` | GPU pod | exact training command on 64 and 256 pairs of A (both include A's eight longest pairs); projects time and cost of a 20,000-pair run with fixed overhead removed; prints peak GPU memory |
| `03_train_arm.sh <arm> <seed> [concat\|sep]` | GPU pod | one full run; checkpoint every 100 steps, keep the last; re-running the same command resumes |
| `04_push_model.sh <arm> <seed>` | GPU pod | upload the final weights to a private model repo `<account>/olmo3-7b-think-dpo-arm_<X>_seed<N>` |
| `rebuild_arms_from_ids.py` | node (called by the bootstrap when the private repo is missing) | rebuild A and B from the public dataset and `../arms/arm_*_ids.json`, plus the timing subsets |
| `05_pipeline.sh [budget]` | GPU pod | unattended sequence: timing test (concat, falls back to sep), budget gate, arm A, push, arm B, push |
| `upload_arms_from_gs01.sh` | the box that holds the arms | upload A, B and the timing subsets to a private dataset repo `<account>/dolci-think-dpo-arms` |

Sequence: `01_bootstrap.sh` → `02_timing_test.sh concat` → read `$WS/logs/timing_summary.txt`
(choose `sep` if peak memory is above ≈ 75,000 MiB) → `03_train_arm.sh A 1` → `04_push_model.sh A 1`
→ `03_train_arm.sh B 1` → `04_push_model.sh B 1` → stop the pod.

Notes
- `patches/save_generation_config.patch` (applied by the bootstrap to the open-instruct clone): with transformers 5.4 the
  final `save_pretrained` raised `GenerationConfig is invalid` because the eos-only generation config that
  `save_with_accelerate` builds was set on the DeepSpeed wrapper, not on the model being saved; the patch copies it over.
  Ai2's environment (transformers 4.57) only warned.
- `OMP_NUM_THREADS` is set per rank (≤ 48) so DeepSpeed's CPU Adam is parallel when optimizer states are offloaded.
- The Hugging Face token is read from the shell (`export HF_TOKEN=…`) and never written to disk.
- The reference-logprob cache is keyed by model, data, `max_seq_length`, loss type and forward mode,
  not by seed (`compute_reference_cache_hash`), so a second seed of the same arm skips that pass.
- Each run writes `run_info.txt` (commit, GPUs, mode, wall time) next to the weights; logs and GPU
  memory traces go to `$WS/logs`.
