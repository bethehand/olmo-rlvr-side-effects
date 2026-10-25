# Can Ai2's 7B Think DPO recipe run on 4 × RTX 4090? No. Plus the training arms for the length test

Step 3 of the #2002 plan. Before the three-arm DPO data test (original pairs vs pairs without the length gap vs a matched control), we tried Ai2's own recipe for Olmo 3 7B Think DPO on the 4 × 24 GB machine used for the other experiments: `open-instruct` at ff32794a3, `dpo_tune_cache.py`, start `allenai/Olmo-3-7B-Think-SFT`, data `allenai/Dolci-Think-DPO-7B`, `dpo_norm` with β = 5, LR 8e-8, 1 epoch, effective batch 32 pairs (1 per GPU × 8 accumulation × 4 GPUs), gradient checkpointing, flash attention, ZeRO-3 with parameters and optimizer offloaded to CPU (`dpo_feas.sh`; 256 pairs, 8 optimizer steps).

| attempt | sequence length | forward | ZeRO-3 config | result |
|---|---|---|---|---|
| 1 | 8K | concatenated | offload | failed before training: the trainer's default reference-logprob cache path is an Ai2 cluster path (`/weka/...`); set `REFERENCE_LOGPROBS_CACHE_PATH` |
| 2 | 8K | concatenated | offload | reference cache OK (64 batches, 3.7 s each, ≈ 5,300 tokens per pair); first backward OOM: 16.9 GiB allocated, asked for 6.12 GiB more (the fp32 logits of 2 × 8K × 100K vocab) |
| 3 | 8K | separate chosen / rejected passes | offload | cache 6.7 s per batch; first backward OOM: 18.6 GiB allocated, asked for 2.4 GiB |
| 4 | 4K | separate | offload | first backward OOM: 18.3 GiB allocated, asked for 1.4 GiB |
| 5 | 8K | concatenated | offload, `stage3_max_live_parameters` and `max_reuse_distance` 1e8, `expandable_segments` | first backward OOM: 17.9 GiB allocated, asked for 6.12 GiB |

Per-GPU peaks 22–24 GB in every attempt; CPU RAM peaked at 204–213 GB of 1 TB, never the constraint. About 18 GiB of GPU memory is taken at the first backward regardless of sequence length, with gradient checkpointing confirmed on (the trainer logs `use_cache=True is incompatible with gradient checkpointing`); its source was not pinned down (`logs/oom_summary.txt`). LoRA would not remove it, since activations and logits dominate, so it was not tried.

**Conclusion.** The recipe needs 80 GB GPUs: 7B full-parameter training states are 112 GB (bf16 weights and gradients, fp32 Adam moments and master weights), so 4 × 80 GB without offload leaves ~28 GB per GPU for states plus ~25 GB of activations and logits at 16K; 8 × 80 GB is comfortable. The 4090 machine does the evaluation side (805-prompt generations, `tools/verbosity_report.py`, safety-eval) and the data preparation.

## Training arms (`build_arms.py`, `arms/`)

From the per-pair table of [`../2026-10-20_dpo-pairs`](../2026-10-20_dpo-pairs) (answer tokens after `</think>`), seed 1, 20,000 pairs each; the pair ids are in `arms/arm_*_ids.json`, the JSONL files stay off the repo (1.2 GB).

| arm | rule | median chosen/rejected ratio | chosen longer | chosen / rejected median tokens | thinking ratio |
|---|---|---|---|---|---|
| pool (150,000) | – | 1.79 | 84% | 527 / 254 | 1.44 |
| A original | random | 1.80 | 84% | 523 / 253 | 1.44 |
| B gap removed | ratio ≤ 1.5, sampled to match A's chosen-length deciles | 1.13 | 65% | 523 / 473 | 1.19 |
| C control | whole pool, matched to B cell by cell on source × chosen-length decile | 1.53 | 80% | 523 / 289 | 1.33 |
| D Ai2's cap | chosen ≤ rejected + 100 tokens | 1.08 | 59% | 107 / 90 | 1.22 |

Why B is a ratio rule and not the report's 100-token cap: on the Think pairs the cap selects pairs where both answers are short (D: medians 107 / 90 tokens against 523 / 253), so a model trained on it would learn "short" directly. The ratio rule keeps the chosen length (B's chosen-length distribution equals A's by construction) and removes most of the gap. B vs C then differs only in the rejected side; A vs C differs in the source mix (B leans to math, 13% against 5%, where pairs are naturally more balanced). D is kept as a fourth arm to show what the report's rule does to Think data. The thinking-length ratio also falls under the rule (1.44 → 1.19), which cannot be separated and is noted.

## Arm B' (approved and queued 2026-10-23 13:25, pre-registered before training; `build_arm_bprime.py`, `arms/arm_Bp_ids.json`, `arms/arm_Bp_summary.md`)

B matched A on chosen-length decile only, so selecting small-gap pairs also changed the prompt population (math sources 19.7% vs
8.4%, median prompt 710 vs 482 chars). B' repeats B with the selection matched to A cell by cell on **source x chosen-length decile x
prompt-length tercile** (19,215 of 20,000 pairs filled at that level; 469 topped up within source x decile, 316 within decile).
Result: source mix within 0.3 points of A on every major source, prompt median 535 chars, math 9.0%; what still differs from A is
tied to the manipulation itself: rejected-answer length (467 vs 253 median tokens), chosen/rejected thinking-length ratio
(1.26 vs 1.44; the thinking gap in characters is about the same, ~1,150), and the rejected answers' list-style formatting. B' shares
2,688 pairs with A. Training recipe identical to A and B (including `--warmup_ratio 0.0`, kept for internal consistency although
the Olmo 3 report uses 0.1), training seed 1.

Pre-registered reading (805-prompt paired answer-length change is primary; B', B and A are all compared with Think-SFT under the
training template):
- B' ≈ 0 and B ≈ 0 while A ≈ +37: the answer-length gap drives the shift; B and B' replicate each other on different pair samples.
- B' ≈ A and B ≈ 0: B's effect came from its prompt population, not the gap.
- B' ≈ 0 and B ≈ A: the gap matters and B's math-heavy population masked it.
- all three ≈ A: at this dose (13% of Ai2's pairs) the gap is not the driver.
Held-out, IFEval, GSM8K and the judge follow the plan in `../2026-10-23_dpo-arms-eval`.

`build_arm_prompts.py ARM_IDS.json OUT.jsonl` rebuilds an arm's prompts (last user turn of the chosen conversation) from the public
dataset for the prompt-overlap check against the evaluation sets; for B' (2026-10-25) the overlap is 0 with all four sets, and the
2,688 prompts shared with A are byte-identical to A's file (`../2026-10-23_dpo-arms-eval/results/arms_data_checks.txt`).

## Rented-node scripts (`rented/`)

The training itself runs on a rented 4 × 80 GB node; `rented/README.md` has the recipe table (two
deviations from Ai2's script: 20,000 pairs per arm, effective batch 32), the cost-first order of runs
(A and B first, C only if they differ, extra seeds only if the gap is small) and the scripts.
