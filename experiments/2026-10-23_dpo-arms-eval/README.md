# Evaluating the DPO data arms: does removing the length gap change the model, and does it cost quality?

Step 4 of the #2002 plan. Two models were trained with Ai2's 7B Think DPO recipe on 20,000 pairs each from
`allenai/Dolci-Think-DPO-7B`, differing only in the pairs (see [`../2026-10-22_dpo-feasibility`](../2026-10-22_dpo-feasibility)):
**A** is a random sample (chosen longer than rejected in 84% of pairs, median chosen/rejected ratio 1.80), **B** keeps the
chosen-length distribution of A but removes most of the gap (ratio ≤ 1.5, 65%, 1.13). Start model `allenai/Olmo-3-7B-Think-SFT`,
effective batch 30, one epoch, seed 1, trained on a rented 3 × H100 node. The recipe follows Table 48 of the Olmo 3 report (LR 8e-8, β 5, length-normalized DPO, one epoch, 16K; `results/olmo3_table48_dpo_hparams.txt`) except for 20,000 pairs instead of 150,000, batch 30 instead of 128, and no warmup instead of a 0.1 ratio. Weights: `icedduck/olmo3-7b-think-dpo-arm_{A,B}_seed1` (private).

Reference points, same evaluation: the start model **Think-SFT**, and Ai2's own **Think-DPO** (`allenai/Olmo-3-7B-Think-DPO`,
trained on the full pool).

**What the pairs are.** Every pair in both arms (and, as far as the arms show, in the pool) is a *delta-learning* pair: the chosen
answer is from `qwen3-reasoning-32b` and the rejected one from `qwen3-reasoning-0.6b` for the same prompt. The length gap in A is
a by-product of that model-size gap; B keeps the chosen-length distribution and selects the prompts on which the small model's
answer is nearly as long as the large model's. B therefore also differs from A in source mix (more math), prompt length (median 710
vs 482 chars) and rejected-answer length (473 vs 253 tokens); arm C, built but not trained, would separate these. In hindsight B should have been matched to A on source × chosen-length decile as well: the ratio ≤ 1.5 subset holds 58,067 pairs (38.7% of the pool) and such a match falls short by only 245 of 20,000 pairs (`results/arms_data_checks.txt`); a next training run should use that construction (B′) rather than C. The arms share no
prompts with IFEval, GSM8K or the held-out set, and one AlpacaEval prompt appears in A (`results/arms_data_checks.txt`).

## Pre-registered acceptance criteria (from the plan, written before any arm's outputs were seen)

1. Answer length on the 805 AlpacaEval prompts: B's median is below A's (paired per prompt).
2. Length-controlled pairwise judgment on the held-out set: B does not lose to A.
3. Detail questions: B's rubric coverage does not drop against A.
4. IFEval and GSM8K: A and B within noise of each other (paired difference interval includes 0).
5. If B wins, the winning arm is re-run with a second seed before anything is claimed.

## Protocol

**Generation**, all four prompt sets, same for every model: the model's own chat template, temperature 0.6, top_p 0.95, one
sample per prompt, seed 1, generation capped at 32,768 new tokens; the answer is the text after the last `</think>`
(OLMES `r1_style`). Software: safety-eval's environment on GS01 (vLLM 0.11.0, transformers 4.57.1), 2 × RTX 4090 per model.
The settings are those of [`../2026-10-18_length`](../2026-10-18_length), which follow OLMES `alpaca_eval_v3::olmo3:adapt`.

| set | prompts | script | what is measured |
|---|---|---|---|
| AlpacaEval | 805 | `../2026-10-18_length/run_length.sh` | answer length (median chars/tokens, paired change), thinking length, share hitting the cap (runaway generation); `tools/verbosity_report.py` |
| held-out `heldout_prompts.jsonl` | 150, three categories of 50 | `run_gen.sh MODEL REV heldout RUN GPUS` | see below |
| IFEval (`google/IFEval`) | 541 | `run_gen.sh ... ifeval ...` | prompt- and instruction-level strict/loose accuracy, `score_ifeval.py` with Google's checkers (their two random draws seeded, so a score is reproducible) |
| GSM8K (`openai/gsm8k` main test) | 1,319 | `run_gen.sh ... gsm8k ...` | accuracy, `score_gsm8k.py` (last `\boxed{}`, else `####`, else last number) |

**Held-out set** (`build_heldout.py`, written by hand on 2026-10-23 before any arm's outputs existed; the script reports the
nearest AlpacaEval instruction per prompt, and the three prompts that matched one were replaced):
- `short`, 50 questions with a short checkable answer (`gold`): correctness by gold-string match, and over-answering (answer > 60 tokens);
- `detail`, 50 how/why questions with a rubric of 3–7 points: coverage judged point by point;
- `open`, 50 open-ended requests: length-controlled pairwise judgment.

**Judge** (`judge_heldout.py`): `Qwen/Qwen3-32B` served locally with vLLM on 4 × 4090 (the judge Ai2 uses in its RL-Zero General
recipe), non-thinking mode, greedy. Pairwise judgments are asked in both orders with an instruction not to reward length as such;
a model wins a prompt only when it wins in both orders, everything else counts as a tie. Rubric coverage is asked per answer.
Engine: bf16, tensor parallel 4, `gpu_memory_utilization` 0.85, `max_num_seqs` 64 (vLLM's default 256 ran out of memory in the
sampler warm-up on 4 × 24 GB, 2026-10-25), `max_model_len` 20,000; answers longer than 6,000 tokens are cut before judging; at most
400 output tokens per judgment. `analyze_heldout.py` prints the tables (bootstrap and Wilson 95% intervals).

**IFEval and GSM8K** are checked on the full sets (541 / 1,319), one sample each; the question is only whether A and B differ,
so the paired difference is what is reported. GSM8K prompts are the bare question, as in Ai2's zero-shot RLVR set
`ai2-adapt-dev/rlvr_gsm8k_zs`; the Think models answer with `\boxed{}`.

**Protocol change on 2026-10-23 (arms only), withdrawn at 10:45 the same day.** *The runaway generations that motivated this were the config artifact described in the compatibility note, not the model; the arms go back to the pre-registered 32K cap and full sets. The paragraph stays as the record.* Arm A's first outputs at the 32K cap showed that a large share of its generations
never terminate: they decay into repetitive loops ("in in in in …", "So. So, but the problem. Hmm. So,") and run to the cap
(9 of the first 16 AlpacaEval outputs, 8 of the first 36 held-out outputs; `*_cap32k_partial.jsonl`). The weights are intact
(all 355 tensors present, no NaN, each within 0.04% of Think-SFT's), so this is what the training did, not a broken save. At
the 32K cap these runs would have taken days on two GPUs, so the arms (and the Think-SFT template control, to stay comparable)
are generated with an **8K cap**, and GSM8K is run on its **first 300 problems** first (the runs resume, so the full set can be
added later). The baselines keep their 32K outputs; when a baseline is compared with an arm, a baseline output longer than
8K tokens is counted as a cap hit, which makes the two capped identically (the first 8K tokens of a sample do not depend on
the cap). The baselines exceed 8K tokens on 0 of 150 held-out, 11–27 of 541 IFEval and 25–27 of 1,319 GSM8K prompts.
**Share of outputs hitting the cap (runaway generation) becomes a primary outcome**, reported per set for every model.

**Compatibility note (the arm A artifact).** The arms were saved by transformers 5.4, whose `config.json` keeps the RoPE
settings under `rope_parameters` (with `rope_theta` inside) and whose `tokenizer_config.json` names the class
`TokenizersBackend`. transformers 4.57.1 does not know either key: the tokenizer fails to load, and, worse, `Olmo3Config`
silently falls back to its defaults, **rope_theta 10000 and no scaling**, instead of 500000 with YaRN (factor 8). vLLM 0.11 reads
`config.rope_theta` / `rope_scaling`, so arm A was first served with the wrong positional encoding, and its generations
degenerated into repetition on anything long (found 2026-10-23 10:35 by diffing the loaded configs; the weights themselves are
within 0.04% of Think-SFT's). Everything measured for A before the fix is kept under `*_badrope` names and is not a result.
`fix_tf5_compat.py` (run by `start_arm_eval.sh` after the fetch) rewrites the cached copy into the 4.57 form; the loaded config
then matches Think-SFT's. The arms' chat template is the training template `olmo_thinker` (system prompt "You are a helpful AI
assistant."), not the released models' longer Olmo system prompt; the Think-SFT control run with the arm's template (`tmplA`)
measures what that difference alone does. So far it changes nothing measurable in **length**: held-out lengths within noise and 0 cap hits, AlpacaEval paired change +0 [−5, +8] tokens; its IFEval and GSM8K accuracy (the full-protocol `tmplA` runs) are still pending, so until they are in, arm comparisons on those two sets use the released-template baseline with that caveat.

## Files

| file | role |
|---|---|
| `build_heldout.py`, `heldout_prompts.jsonl` | the 150 held-out prompts and their construction |
| `generate_prompts.py`, `run_gen.sh`, `queue_model.sh` | generation for heldout / ifeval / gsm8k on GS01 |
| `watch_arm_push.sh`, `start_arm_eval.sh` | laptop-side watcher that, once the rented node has uploaded an arm's weights, fetches them to GS01 and runs `start_arm_eval.sh` there: the 805-prompt run on one GPU pair, heldout / IFEval / GSM8K on the other, then (arm A) the same runs for Think-SFT with the arm's chat template as a template control |
| `fix_tf5_compat.py` | makes a transformers-5-saved model load correctly under transformers 4.57 / vLLM 0.11: tokenizer class, and the RoPE settings that otherwise fall back to rope_theta 10000 without YaRN (see the compatibility note) |
| `A_rerun.sh` | re-evaluation of arm A after the config fix (32K cap, full sets) |
| `ctrl_tmplA.sh`, `ctrl_tmplA_full.sh`, `ctrl_tmplA_ifeval.sh`, `ctrl_tmplA_gsm8k.sh` | template control: Think-SFT generated with arm A's chat template (`tmplA`, seed 1). `ctrl_tmplA.sh` is the first version (8K cap, partial sets, killed at 10:45 with the withdrawn protocol); the three `_full` / `_ifeval` / `_gsm8k` scripts re-ran the control at the pre-registered 32K cap on all four sets |
| `bp_pipeline.sh`, `judge_all.sh` | 2026-10-25: arm B' evaluation with a smoke gate (held-out run first; the 805-prompt run starts only if cap hits ≤ 5 % and `</think>` is present on ≥ 95 %), one resume attempt per set, then `judge_all.sh`: the held-out judge over all six output files (7 pairs in both orders + rubric coverage, Qwen3-32B tp 4) |
| `seed2_805.sh` | 2026-10-25: after the judge, a second generation seed (run2 = engine seed 2, otherwise run1's settings) of the 805 prompts for A, B, B' and the training-template control (`tmplA2`), two runs at a time (about 3 h); answers how much B − A and B' − A move under re-sampling alone. Not a second training seed. |
| `train_loss_table.py` | 2026-10-25: mean logged DPO loss per third of training for A, B, B' from the rented node's logs (`../2026-10-22_dpo-feasibility/rented/logs_node/`), and exp(−loss) as the implied preference probability; `results/train_loss_run1.txt`. |
| `weight_shift.py` | 2026-10-25: how far each arm's weights moved from Think-SFT (relative L2 delta, overall and by parameter group) and the cosine similarity between the arms' deltas, read from the safetensors files on CPU; `results/weight_shift_run1.txt`. Together with the loss table it addresses the alternative explanation "B and B' simply learned less". |
| `judge_stats.py` | 2026-10-25: per judged pair the win-share difference with a bootstrap interval and a sign test, how often the longer answer won, the decided prompts split by which side was longer, and the near-equal-length stratum; its output is the second block of `results/heldout_judge_run1.txt`. |
| `length_table.py` | the 805-prompt length table (`results/alpaca805_lengths_run1.txt`, `_run2.txt`): per-model medians with bootstrap intervals, paired changes against the training-template control, arm-vs-arm differences with sign tests and robustness checks; `length_table.py OUT_DIR [run2] [--compare DIR]` since 2026-10-25 (second generation seed and the run-to-run noise block). |
| `truncate_to_cap.py` | re-caps the baselines' 32K outputs at 8K after the fact (`*_cap8k.jsonl`), so that arms and baselines are compared under the same cap (see the protocol change) |
| `score_ifeval.py`, `score_gsm8k.py`, `analyze_heldout.py`, `judge_heldout.py` | scoring |
| `outputs/`, `judgments/`, `results/` | per-sample outputs (not committed here; the six models' generations for all four sets, plus the second-seed 805 runs, are public on the Hub as `icedduck/olmo3-7b-think-dpo-arms-eval-outputs`), judge records (`judgments/heldout_run1.jsonl`, committed), tables |

## What is established so far (2026-10-25 10:15; all four sets, the judge and a second generation seed done for A, B and B')

Issue #2002 asks where Olmo 3's verbosity comes from. Three layers are measured under one protocol (805 AlpacaEval prompts paired
per prompt, 150 held-out prompts, IFEval, GSM8K; temperature 0.6, seed 1, 32K cap; tables in `results/`):

1. **The DPO stage is the source.** Think-SFT → Think-DPO (one stage apart) lengthens answers from 600 to 874 median tokens,
   +243 [+231, +258] per prompt, longer on 94% of prompts; this replicates the independent 2026-10-19 measurement (1.5×, 96%).
2. **It is verbosity, with a cost.** Short factual questions grow from 138 to 295 tokens; thinking grows too (+256 tokens); the
   share of answers over 1,000 tokens rises from 12% to 38%; IFEval strict falls 0.765 → 0.719 (−4.6 points) and GSM8K 0.946 → 0.908
   (−3.9 points), with `\boxed{}` kept in 625 of 1,319 answers instead of 1,288. A Qwen3-32B judge nevertheless prefers Think-DPO
   115 to 7 on the held-out set, and it picked the longer answer on 112 of the 117 decided prompts where Think-DPO was longer: a
   judge of this kind rewards the behaviour the issue is about.
3. **Why: the chosen-longer pattern in the pairs carries about half of the shift at this dose.** Three arms with the same recipe
   (Table 48 settings except 20,000 pairs, effective batch 30, no warm-up), start model and seed, differing only in which pairs they
   see: A, random pairs (chosen longer in 84%, median length ratio 1.80); B and B', pairs with chosen/rejected ratio ≤ 1.5 (chosen
   longer in 65–67%, ratio 1.13–1.15), B matched to A on chosen-length deciles, B' matched cell by cell on source × chosen-length
   decile × prompt-length tercile. Answers against the training-template Think-SFT: A +35 [+25, +46] tokens, B +15 [+6, +22],
   B' +16 [+5, +22]; B − A −19 [−30, −7], B' − A −15 [−24, −7], B' − B +3 [−2, +9]. On the held-out set A lengthens detail answers
   by +58 [+12, +88] and open answers by +28 [+14, +66] while B' sits at +4 / +4 / +6 (every interval includes 0). IFEval and GSM8K:
   the three arms are within noise of each other and of the start model. Judge: A vs B a tie (+0.007 [−0.107, +0.120]); A vs B'
   −0.113 [−0.227, +0.007] for B', with the longer answer winning 79% of decided arm-vs-arm prompts. The filtered pairs are a weaker
   signal by construction (final loss 0.56 vs 0.50; weights moved 20% less than A's), so part of the smaller shift may be "less
   learned"; the shift shrank by 55% on AlpacaEval and to zero on the held-out set, more than the 20%, which weighs against that
   reading without settling it.

Pre-registered criteria: 1 met for B and B'; 2 met for B and at the edge for B'; 3 met (at the ceiling, uninformative); 4 met;
5 not met (B' is a second selection trained with the same seed, not a second training seed).

A second generation seed reproduces the length result (B − A −14 [−25, −6], B' − A −14 [−21, −8]; same-model run-to-run paired
medians within ±4 tokens; `results/alpaca805_lengths_run2.txt`). Restricting the judge's decided prompts to answers within 20% of
each other in length leaves A vs B at 20–19 and A vs B' at 33–20, so B' may carry a small quality cost that length alone does not
explain, at the edge of noise; B shows none.

Caveats: one training seed; 13% of Ai2's pair count, batch 30 instead of 128, no warm-up; the judge follows length; the released template's system prompt lengthens Think-SFT's thinking by 63
tokens but not its answers.

## Status

- 2026-10-23 01:33 CST: Think-SFT and Think-DPO baseline runs (heldout, ifeval, gsm8k) started on GS01; arm A training on the
  rented node at step 400/667; arm B queued after it.
- 2026-10-23 03:40 CST: IFEval baselines scored (`results/ifeval_baselines_run1.txt`): prompt-level strict 0.765 (Think-SFT) vs
  0.719 (Think-DPO), paired difference −0.046 [−0.081, −0.011]; Think-SFT hit the 32K cap on 23 of 541 prompts, Think-DPO on 3.
  Held-out baselines done, GSM8K baselines running; judge model downloaded; arm A at step 570/667.
- 2026-10-23 04:19 CST: arm A trained (667 steps, 6 h 59 min, final loss ≈ 0.50) and uploaded; arm B started on the rented node.
  A's weights fetched to GS01 (tokenizer_config fixed, see the compatibility note); its runs start once the fetch completes.
- 2026-10-23 05:30 CST: arm A's 32K-cap runs stopped after 1 h: runaway repetitive generations on a large share of prompts (see the
  protocol change above); weights verified intact; runs restarted with the 8K cap (805 on GPUs 2,3; held-out / IFEval / GSM8K-300
  on GPUs 0,1; then the Think-SFT template control). GSM8K baselines scored (`results/gsm8k_baselines_run1.txt`): Think-SFT
  0.945, Think-DPO 0.879, paired difference −0.066 [−0.082, −0.051]. Arm B training on the rented node (reference pass).
- 2026-10-23 06:25 CST: arm A at the 8K cap, so far: AlpacaEval 91 of 104 finished outputs hit the cap; held-out 93 of 125
  (short 23/50, detail 47/50, open 23/25). Given this, the Think-SFT template control was dropped and A's IFEval / GSM8K runs
  limited to 200 prompts each; a greedy-decoding run of the held-out set (`greedy1`) is queued to see whether sampling triggers
  the collapse. Arm B started its training steps at 06:16 CST (step 10 loss 0.693).
- 2026-10-23 07:30 CST: arm A's held-out run finished (8K cap): 117 of 150 outputs hit the cap (short 23/50, detail 47/50, open 47/50);
  `results/heldout_lengths_sft_dpo_A_run1.txt`. On the same set Think-SFT and Think-DPO hit no cap, and Think-DPO's answers are
  longer than Think-SFT's on every category (paired median +116 / +282 / +241 tokens for short / detail / open). AlpacaEval 192 of
  217 at the cap so far, IFEval 79 of 96. Arm B at step 106/667 (41 s/step, loss 0.683 at step 100); push expected about 14:15 CST;
  balance $100 at 07:24 lasts to about 15:45 CST.
- 2026-10-23 08:30 CST: arm A's IFEval run (first 200 prompts) finished: 178 of 200 hit the 8K cap; prompt-level strict
  accuracy 0.175 against Think-SFT 0.735 and Think-DPO 0.685 on the same 200 prompts with their outputs re-capped at 8K
  (`results/ifeval_A_vs_baselines_first200_run1.txt`; `truncate_to_cap.py` added, and `results/ifeval_baselines_cap8k_run1.txt`
  gives the full-set baselines under the 8K cap: 0.762 vs 0.715, difference −0.046 [−0.081, −0.011], unchanged from 32K).
  A's GSM8K-200 run started; AlpacaEval 294 of 332 at the cap. Arm B at step 196/667 (loss 0.655 at step 190).
- 2026-10-23 09:50 CST: arm A's GSM8K-200 run finished at 09:31 (165 of 170 seen at 09:23 had hit the cap; scored below once copied);
  greedy held-out diagnostic running on GPUs 0,1. The Think-SFT template control is back on (`ctrl_tmplA.sh`, queued behind the
  greedy run on GPUs 0,1), so that A is compared with its start model under the very same template, seed and cap. AlpacaEval 390 of
  446 at the cap. Arm B at step 282/667 (loss 0.625 at step 280).
- 2026-10-23 09:55 CST: arm A's GSM8K-200 scored (`results/gsm8k_A_vs_baselines_first200_run1.txt`): 195 of 200 hit the 8K cap,
  accuracy 0.040 against Think-SFT 0.940 and Think-DPO 0.855 on the same 200 problems re-capped at 8K (A − SFT: −0.900 [−0.940, −0.855];
  DPO − SFT on these 200: −0.085 [−0.130, −0.040]).
- 2026-10-23 10:25 CST: greedy diagnostic finished (`heldout`, run `greedy1`, temperature 0): 124 of 150 outputs hit the 8K cap
  (short 25/50, detail 50/50, open 49/50), against 117 of 150 with sampling. The collapse is not a sampling artefact. Template
  control (`tmplA`) started 10:18 on GPUs 0,1. AlpacaEval 488 of 551 at the cap. Arm B at step 366/667 (loss 0.595 at step 360).
- 2026-10-23 10:40 CST: template control on the held-out set: Think-SFT with arm A's template hits the cap on 0 of 150 prompts
  (median answer tokens 138 / 742 / 729 for short / detail / open, against 173 / 764 / 712 with its released template), so the
  template plays no part in A's collapse. Plan agreed with the user: the start model is the primary reference for both arms, and
  the `tmplA` run serves as the template control for B too, provided B's saved `chat_template.jinja` is byte-identical to A's
  (checked after B's upload; otherwise a `tmplB` run is added). `start_arm_eval.sh` now defaults IFEval and GSM8K to the first
  200 prompts, A's limits, so B is compared with A on identical prompts first; the full sets are added by resume if B is stable.
- 2026-10-23 10:45 CST: **all arm A results above are withdrawn.** An independent check of the experiment found that A's
  `config.json`, written by transformers 5.4, loads under GS01's transformers 4.57 with rope_theta 10000 and no YaRN (Think-SFT:
  500000 with YaRN), so vLLM served A with the wrong positional encoding; that, not the training, is what made its long
  generations degenerate (the template control, served from Think-SFT's own config, was unaffected). `fix_tf5_compat.py` repairs
  the cached config (verified: the loaded config now equals Think-SFT's), the invalid outputs and result files are renamed
  `*_badrope`, and A is being re-evaluated on the pre-registered protocol (32K cap, full IFEval and GSM8K; `A_rerun.sh`): held-out
  set first on GPUs 2,3, then the 805 prompts; held-out / IFEval / GSM8K on GPUs 0,1 after the template control. B will go through
  the same fix before its runs. The same check also improved `score_gsm8k.py` (answer markers and bold numbers before the
  last-number fallback): Think-DPO 0.879 → 0.908, Think-SFT 0.945 → 0.946, paired difference −0.039 [−0.053, −0.025] instead of −0.066; `results/gsm8k_baselines_run1.txt` regenerated.
- 2026-10-23 10:52 CST: **diagnosis confirmed.** With the repaired config, arm A's held-out run finished in 429 s with 0 of 150
  outputs at the cap (before the fix: 117 of 150 at an 8K cap); median answer tokens 139 / 821 / 751 for short / detail / open,
  against 138 / 742 / 729 for Think-SFT under the same template (`results/heldout_lengths_run1.txt`). A is an ordinary model;
  the collapse was the positional-encoding artifact. Its 805-prompt run (32K cap) started on GPUs 2,3; IFEval and GSM8K follow on
  GPUs 0,1 after the template control. Arm B at step 401/667 (loss 0.595).
- 2026-10-23 11:00 CST: the template-control chain (`ctrl_tmplA.sh`, launched under the withdrawn 8K / 200-prompt settings) was
  stopped after its held-out, IFEval-200 and part of its GSM8K-200 runs so that GPUs 0,1 go to arm A's full IFEval and GSM8K;
  those outputs are kept as `*_tmplA_cap8k.jsonl`. The control is re-run on the full protocol (32K cap, full sets, `tmplA`)
  after arm B's runs, as the lowest-priority job. GPU plan: A's 805 prompts on GPUs 2,3 (until about 13:30), A's IFEval / GSM8K
  on GPUs 0,1 (until about 14:10); B's runs start when its weights arrive (about 14:20); the control and then the judge follow.
- 2026-10-23 11:50 CST: arm A re-evaluation (repaired config, 32K cap) running normally: AlpacaEval 545 of 805, 6 at the cap,
  median answer 670 tokens so far; IFEval 475 of 541, 11 at the cap; GSM8K (full set) follows on GPUs 0,1. `fix_tf5_compat.py`
  now also checks the loaded config and tokenizer against Think-SFT (`--check_against`), and `start_arm_eval.sh` refuses to
  start when they differ; `A_rerun.sh` added. Arm B at step 498/667 (loss 0.580).
- 2026-10-23 12:25 CST: arm A IFEval (full 541, 32K) scored, `results/ifeval_run1.txt`: prompt-level strict 0.754 [0.716, 0.789],
  21 at the cap; against Think-SFT (released template) −0.011 [−0.043, +0.020], against Think-DPO's −0.046. AlpacaEval 803 of 805
  done (15 at the cap, median answer 647 tokens so far); GSM8K full set running on GPUs 0,1. Arm B at step 539/667 (loss 0.563).
- 2026-10-23 12:30 CST: arm A's 805-prompt run finished (1 h 34 min, 32K cap): 16 at the cap, median answer 647 tokens; paired change
  against Think-SFT (released template) +37 [+27, +49] tokens, longer on 63% of prompts, against Think-DPO's +242 [+226, +261] on 92%
  (`results/alpaca805_lengths_run1.txt`). The same-template control on the full protocol started on the freed GPUs 2,3 (held-out, then
  805; `ctrl_tmplA_full.sh`); its IFEval / GSM8K part is queued after arm B. A's GSM8K at 178 of 1,319. Arm B at step 542/667.
- 2026-10-23 13:25 CST: plan approved by the user: arm B' (`../2026-10-22_dpo-feasibility/build_arm_bprime.py`, pre-registered there)
  trains on the same rented node right after B's upload (`rented/06_train_bprime.sh`, about 8.5–9 h), is the last training run,
  and goes through the same fetch / compatibility check / evaluation here (watcher started). The user tops up the node's balance
  as needed and terminates the node after B' is uploaded. Arm A: GSM8K at 1,0xx of 1,319; template control 805 run in progress.
- 2026-10-23 14:25 CST: arm A complete on all four sets (repaired config, 32K). GSM8K full set (`results/gsm8k_run1.txt`): A 0.939
  [0.925, 0.951], 3 at the cap, against Think-SFT 0.946 (−0.007 [−0.017, +0.003]) and Think-DPO 0.908 (−0.039); A keeps `\boxed{}`
  in 1,214 of 1,319 answers (Think-SFT 1,288, Think-DPO 625). Summary for A so far: answers +37 tokens on AlpacaEval, IFEval and
  GSM8K within noise of the start model. Template control: 805 run at 576 of 805 on GPUs 2,3; its IFEval started on the idle GPUs
  0,1 (`ctrl_tmplA_ifeval.sh`), GSM8K part later. Arm B at step 640/667 (loss 0.562), upload expected about 15:10; B' follows.
- 2026-10-23 15:05 CST: **arm B trained and uploaded** (667 steps, exit 0, 10 h 25 min wall including tokenization and the reference
  pass; final loss ≈ 0.56; pushed 14:46). The B' runner started B' at 14:47 on the same node (waited for the pipeline's end marker and
  for the GPUs to be free). The laptop watcher for B had stopped without reporting (it was running through the 10:55 file-permission
  outage), so B's fetch, compatibility check and evaluation launch were done by hand at 15:05: held-out / IFEval / GSM8K on GPUs 2,3
  now, the 805 prompts on GPUs 0,1 after the template control's IFEval. Template control part 1 done (805: 16 at the cap).
- 2026-10-23 16:00 CST: template control IFEval done (`results/ifeval_run1.txt`, scorer now re-seeds its random draws per file so a
  score no longer depends on file order): Think-SFT with the training template 0.784 [0.747, 0.816] strict, 25 at the cap, against
  0.765 with the released template (released − training −0.018 [−0.054, +0.013]). Against this exact control, arm A is −0.028
  [−0.061, +0.006] strict and Think-DPO −0.065 [−0.102, −0.031]; against the released-template SFT, A is −0.009 and Think-DPO
  −0.046. So the system prompt moves IFEval by about two points (not significant) and A's IFEval change sits between −1 and −3
  points depending on the reference, with both intervals including 0. The control's GSM8K runs tonight. Arm B's download
  (5–6 MB/s) is about to finish; its evaluation starts automatically.
- 2026-10-23 16:00 CST: **arm B evaluation started** (15:57): its `chat_template.jinja` is byte-identical to A's (sha256 79483a77…), so
  the `tmplA` control stands for both arms; `fix_tf5_compat.py --check_against` rewrote the config and confirmed the loaded config and
  tokenizer match Think-SFT. 805 prompts on GPUs 0,1, held-out / IFEval / GSM8K on GPUs 2,3 (32K cap, full sets). The template
  control's GSM8K (`ctrl_tmplA_gsm8k.sh`) is queued on GPUs 0,1 behind B's 805 run; with it the control is complete on all four sets.
- 2026-10-23 16:50 CST: arm B held-out set done (591 s, 0 at the cap): median answer tokens 150 / 774 / 770 (short / detail / open),
  paired change against the training-template Think-SFT +6 [−12, +24] / +32 [−24, +97] / +46 [+5, +64]; arm A was −4 / +58 / +28.
  All short questions correct for both arms. On the 805 prompts B is at 467 of 805 with a running median of 659 answer tokens (A
  finished at 647, the control at 577): so far B does not look shorter than A. Arm B' training started its steps at about 16:30
  (step 30, loss 0.69); upload expected about 01:30.
- 2026-10-23 17:15 CST: arm B IFEval done (541, 14 at the cap; `results/ifeval_run1.txt`): prompt-level strict 0.765 [0.728, 0.799],
  against the training-template Think-SFT −0.018 [−0.052, +0.011], against arm A +0.009 [−0.022, +0.043]. Criterion 4 (A and B within
  noise of each other) holds on IFEval. B's GSM8K running on GPUs 2,3; B's 805 prompts at 610 of 805 (running median 629 tokens).
- 2026-10-23 17:45 CST: **arm B on the 805 prompts, preview on 803 of 805 finished rows**: median answer 611 tokens; paired change
  against the training-template Think-SFT +14 [+5, +23] (longer on 56% of prompts) against A's +35 [+25, +46] (62%); **B − A = −20
  [−30, −8]** on the same prompts, B shorter on 57%. Thinking length unchanged for both (868 / 876 vs 885). Pre-registered criterion 1
  (B's median length below A's, paired) is met on this preview; IFEval already within noise (criterion 4); GSM8K and the judge pending.
  Caveats: one training seed per arm; B's pair selection also shifted the prompt mix (B' tests that); the shift is small in absolute
  terms (+14 vs +35 tokens on a 577-token median).
- 2026-10-23 18:20 CST: **arm B's 805-prompt run finished** (1 h 48 min; `results/alpaca805_lengths_run1.txt`): median answer 613 tokens,
  14 at the cap; paired change against the training-template Think-SFT **+15 [+6, +22]** (longer on 56% of prompts) against A's
  **+35 [+25, +46]** (62%); **B − A = −19 [−30, −7]**, B shorter on 451 prompts and longer on 347 (sign test z = 3.7). The
  difference has the same sign in all five AlpacaEval subsets (−10 to −48 tokens) and holds for chars and trimmed means. Thinking
  length unchanged for both arms (868 / 877 vs 885). Training loss at the end: A 0.50, B 0.56 (B's pairs are harder to separate).
- 2026-10-23 19:05 CST: arm B GSM8K done (1,319, 0 at the cap; `results/gsm8k_run1.txt`): 0.947 [0.933, 0.958], against Think-SFT
  (released template) +0.001 [−0.008, +0.010], against A +0.008 [−0.002, +0.017]. With IFEval (+0.9 points vs A) this settles criterion
  4: A and B are within noise of each other on both accuracy sets, and neither is below the start model. B's four sets are complete;
  the template control's GSM8K (719 of 1,319) finishes about 19:45, then the judge runs on all four GPUs.
- 2026-10-23 23:20 CST: template control complete on all four sets. GSM8K (`results/gsm8k_run1.txt`): Think-SFT with the training
  template 0.946 [0.933, 0.957], identical to the released template (paired +0.000 [−0.009, +0.008]); against this exact control
  arm A is −0.007 [−0.017, +0.004] and arm B +0.001 [−0.008, +0.009]; B − A +0.008 [−0.002, +0.017]. So the system prompt changes
  thinking length (+63 tokens) and possibly IFEval (−1.8 points, n.s.) but not answer length or GSM8K. The judge run started at
  20:31 and failed at engine start: another user's jobs hold about 10 GB on GPUs 1–3 (shared box), below vLLM's required free
  memory for Qwen3-32B at tp=4; the judge is re-queued to start when all four GPUs are free. Arm B' at step 610/667; upload
  expected about 00:15.
- 2026-10-23 23:25 CST: **GS01 paused on the user's instruction** (someone else needs the box): the queued judge runner and the B'
  watcher were stopped; nothing of ours is running or waiting there. The judge for A/B and everything for B' (fetch, check, four
  sets, judge) start only when the user gives the go; a monitor reports when the GPUs are free. Arm B' training is unaffected
  (step 620/667 on the rented node, upload expected about 00:15).
- 2026-10-23 23:58 CST: **arm B' trained and uploaded** (667 steps, exit 0, 9 h 03 min wall including the reference pass, final
  loss ≈ 0.57, pushed 23:53 to `icedduck/olmo3-7b-think-dpo-arm_Bp_seed1`). The rented node has nothing left to do; its logs and
  run records are copied to `../2026-10-22_dpo-feasibility/rented/logs_node/`. B' is fetched and evaluated on GS01 only when the
  user gives the go (the box is in use by someone else).
- 2026-10-24 00:28 CST: rented node terminated after the checks (three repos complete on the Hub, logs copied, nothing running):
  29.5 h at $11.97/h ≈ $353 for this node; the network volume (119 GB used: model copies, arm data, reference caches, environment)
  is kept for now at about $0.03/h as a backup until B' is evaluated.
- 2026-10-25 02:50 CST: the user lifted the GS01 pause ("现在4张4090都空出来了 你都可以用 不用过问我"; order: B' evaluation, then the
  judge, then a full re-check). The B' weights were already in GS01's hub cache: the laptop watcher started on 10/23 13:23 had not
  been stopped at 23:25 and fetched them at 23:53–01:19 (network and disk only, no GPU; the evaluation did not start because the
  download's ssh session dropped and the watcher died with it). Logged as lesson 5 in the project's mistake log; the stale download
  process was confirmed complete (9 files, `model.safetensors` byte-identical in size to A and B) and killed at 10:29.
- 2026-10-25 03:05 CST: **B' evaluation started** (`bp_pipeline.sh`). Before it: `fix_tf5_compat.py ... --check_against
  allenai/Olmo-3-7B-Think-SFT` rewrote B's cached config (rope_theta 500000 to the top level, YaRN rope_scaling, tokenizer class)
  and reported "loaded config and tokenizer match"; the chat templates of A, B and B' are identical (sha256 79483a77…). Held-out /
  IFEval / GSM8K chain on GPUs 2,3; the 805-prompt run on GPUs 0,1 after the smoke gate.
- 2026-10-25 03:12 CST: smoke gate passed (held-out: 150 rows, 0 cap hits, `</think>` on all, median answer 587 tokens); the 805-prompt
  run started 03:11. B' prompt overlap with the four evaluation sets is 0 / 0 / 0 everywhere (`results/arms_data_checks.txt`; prompts
  rebuilt from the public dataset with `../2026-10-22_dpo-feasibility/build_arm_prompts.py`, byte-identical to A's on the 2,688
  shared ids). B' held-out medians (short / detail / open): 149 / 765 / 749 tokens; paired change against the training-template
  Think-SFT, median of per-prompt differences from `analyze_heldout.py` (`results/heldout_lengths_run1.txt`): +4 [−18, +16] /
  +4 [−46, +34] / +6 [−28, +42] (A: −4 / +58 / +28; B: +6 / +32 / +46); short questions 50/50 correct. [Corrected 05:45: this entry
  first quoted +6 / −8 / +12, A +6 / +48 / +15, B +5 / +16 / +26 from an ad-hoc quick look that did not use the canonical script.] IFEval expected about 04:10,
  the 805-prompt run about 05:00, GSM8K about 06:00, then the judge on all four GPUs (about 1 h).
- 2026-10-25 04:23 CST: **B' IFEval scored** (`results/ifeval_run1.txt`, now six rows; the five-row table of 10-23 kept as
  `ifeval_run1_5rows_1023.txt`): prompt strict 0.769 [0.732, 0.802], loose 0.817, 16 cap hits. Paired against the training-template
  Think-SFT −0.015 [−0.048, +0.020] strict; against A +0.013 [−0.018, +0.043]; against B +0.004 [−0.030, +0.037]. Criterion 4 holds
  for B' as for B.
- 2026-10-25 05:25 CST: **B' 805-prompt run finished** (1 h 26 min, 805 rows, 13 cap hits; `results/alpaca805_lengths_run1.txt`, six
  rows, produced by `length_table.py`; five-row version kept as `alpaca805_lengths_run1_5rows_1023.txt`): median answer 622 [577, 658]
  tokens, +16 [+5, +22] against the training-template Think-SFT (A +35, B +15). **B' − A: −15 [−24, −7] tokens, B' shorter on 454 of
  805, longer on 344, sign test z = 3.89**; chars −87 [−132, −55]; 10% trimmed mean −29 tokens; shorter in four of the five AlpacaEval
  sources (oasst +2). B' − B: +3 [−2, +9] tokens (381 shorter / 414 longer, z = −1.17), so the source-mix drift in B's selection did not
  drive B's result: matching source × chosen-length decile × prompt-length tercile cell by cell gives the same answer. Criterion 1 holds
  for B'. Criterion 5 is not met by B': it is a second selection of the same intervention, trained with the same seed, not a second
  training seed of B, so the training-seed gap stays open. All six 805 files carry the same 805 instructions
  (sha256 prefix 80812b05). GSM8K at 1,118/1,319 at 05:23; judge follows automatically.
- 2026-10-25 05:50 CST: **B' GSM8K scored; the B' evaluation is complete** (GSM8K 1 h 30 min, 1,319 rows, 1 cap hit;
  `results/gsm8k_run1.txt`, six rows, five-row version kept as `gsm8k_run1_5rows_1023.txt`): accuracy 0.940 [0.926, 0.952], `\boxed{}`
  in 1,258 of 1,319; against the training-template Think-SFT −0.006 [−0.016, +0.005]; against A +0.001 [−0.009, +0.010]; against B
  −0.007 [−0.017, +0.002]. Criterion 4 holds for B' on IFEval and GSM8K. `results/heldout_lengths_run1.txt` regenerated with six
  rows (four-row version kept as `heldout_lengths_run1_4rows_1023.txt`): B' paired change against the training-template Think-SFT
  +4 / +4 / +6 tokens (short / detail / open; every interval includes 0), A −4 / +58 [+12, +88] / +28 [+14, +66], B +6 / +32 / +46.
  The pipeline logged "B' evaluation complete: length 805/805, heldout 150/150, ifeval 541/541, gsm8k 1319/1319" at 05:45 and started
  `judge_all.sh` on all four GPUs at 05:47 (about 1 h). `seed2_805.sh` is waiting for the judge's exit-0 marker and idle GPUs, then
  runs the second generation seed of the 805 prompts (A, B, B', tmplA2; about 3 h).
- 2026-10-25 05:57 CST: **the judge's first launch died at engine start-up; fixed and relaunched.** At 05:47 `judge_all.sh` started
  Qwen3-32B with `gpu_memory_utilization` 0.90 and vLLM's default `max_num_seqs` 256; the sampler warm-up ran out of GPU memory on three
  of four workers (16 GB of weights + 4.35 GB KV cache per 24 GB card left no room for 256 rows of logits), exit 1 after 172 s
  (`logs/judge_all_oom_0547.log`). This is the issue `../2026-10-18_length/generate_length.py` already documents for 32B models on
  this box (`--max_num_seqs 64`); the judge script had never been started on an idle box before (the 10-23 attempt failed earlier,
  on other users' memory). Fix: `judge_heldout.py` gained `--max_num_seqs` (default 64) and `judge_all.sh` passes `--gpu_mem 0.85
  --max_num_seqs 64`. A smoke run on three detail prompts (2 orders + rubric = 9 records, 66 s, JSON verdicts parsed) passed at 05:56
  and the full judge relaunched at 05:56:50. `seed2_805.sh` only recognises an exit-0 judge marker, so the failure did not start the
  second-seed runs. Logged as lesson 7 in the project's mistake log (every GPU stage needs its own smoke gate; reuse proven engine
  settings for a model size already run on the box).
- 2026-10-25 06:05 CST: **training-loss table** (`results/train_loss_run1.txt`, from the node logs): mean logged loss over the last
  third of training A 0.499, B 0.566, B' 0.562 (implied preference probability exp(−loss) 0.61 vs 0.57 vs 0.57); B and B' are
  indistinguishable here too. The length-filtered pairs give a weaker preference signal by construction, so part of the smaller
  length shift could be "less learned" rather than "no length signal"; `weight_shift.py` (running on GS01's CPUs, no GPU) measures
  how much less the weights moved. Judge at 2,400 requests on all four GPUs since 05:57.
- 2026-10-25 06:25 CST: **weight shift measured** (`results/weight_shift_run1.txt`): relative L2 distance from Think-SFT (× 1e-3)
  A 0.102, B 0.080, B' 0.081, so the length-filtered arms moved the weights about 20 % less (B/A 0.785, B'/A 0.796, B'/B 1.015);
  the split between attention (45–49 %) and MLP (51–54 %) is the same for all three; cosine similarity of the deltas A~B 0.68,
  A~B' 0.73, B~B' 0.84. Reading: B and B' are near-identical training runs (loss, weight delta, direction, and every evaluation
  agree), and their smaller length shift on the 805 prompts (+15 / +16 against A's +35, a 55 % reduction) is larger than the 20 %
  reduction in how far the weights moved, while on the held-out set they are at the start model's level. This weighs against, but
  does not rule out, "B and B' simply learned less": the dose-to-behaviour mapping is not known to be linear. Judge at 1,823 / 2,400.
- 2026-10-25 06:45 CST: **judge finished** (`judge_all.sh` 05:57–06:30, 2,400 records in 2,012 s; 31 rubric records unparsed, every
  pairwise record parsed; `results/heldout_judge_run1.txt`, intervals and the length check from `judge_stats.py`). Pairwise, a win
  needing both orders: every arm beats the training-template Think-SFT (win-share difference A +0.147 [+0.033, +0.253], B +0.127
  [+0.007, +0.240], B' +0.100 [−0.020, +0.213]); **A vs B is a tie (34 / 81 / 35, +0.007 [−0.107, +0.120]); A vs B' 49 / 69 / 32,
  −0.113 [−0.227, +0.007] for B', z −1.89**; B vs B' 45 / 67 / 38 (−0.047 [−0.167, +0.067]); Think-SFT vs Think-DPO (released
  templates) 7 / 28 / 115 (+0.720). Rubric coverage on the detail questions is at the ceiling for every model (0.986–0.996).
  **The judge follows length**: among decided arm-vs-arm prompts the longer answer won 185 of 233 (79 %); when A's answer is the
  longer one A beats B' 36 to 9, when B''s is longer B' wins 23 to 13; Think-DPO beat Think-SFT on 112 of the 117 decided prompts
  where it was longer. So criterion 2 holds for B and sits at the edge for B' (the interval touches 0), and this judge cannot separate
  "better" from "longer" at this scale: a length-matched or length-penalised judgment is needed before any quality claim. Criterion
  3 holds but is uninformative at the ceiling. The second generation seed (run2) started 06:32 (A on GPUs 0,1 and B on 2,3; B' and
  the tmplA2 control follow), about 3 h in all.
- 2026-10-25 10:10 CST: **second generation seed done** (`seed2_805.sh`, 06:32–10:04: A 1 h 38 min, B 1 h 18 min, B' 1 h 52 min,
  tmplA2 1 h 55 min; all 805 rows; `results/alpaca805_lengths_run2.txt`; `length_table.py` now takes a RUN argument and
  `--compare DIR` for the run-to-run check). Against the seed-2 control: A +30 [+19, +41] tokens, B +17 [+8, +24], B' +14 [+5, +23]
  (run1: +35 / +15 / +16). **B − A −14 [−25, −6] (B shorter on 449 of 805, z 3.54); B' − A −14 [−21, −8] (457 of 805, z 4.07)**;
  B' − B +0 [−7, +10]. Same model, run2 minus run1 on the same prompts: control +0 [−8, +6], A +2 [−3, +9], B +0 [−9, +11],
  B' −4 [−11, +3], so the arm-vs-arm differences are several times the generation noise. Thinking length is noisier: A's thinking
  change against the control is +58 tokens in run2 and +0 in run1, so "thinking unchanged" should be read as "within run-to-run
  noise". Nothing of ours is running on GS01 any more; all four GPUs are free.
- 2026-10-25 10:15 CST: `judge_stats.py` gained a near-equal-length stratum (decided prompts whose answers are within 20 % of each
  other in length; `results/heldout_judge_run1.txt` regenerated): A vs B 20–19, A vs B' 33–20, B vs B' 22–20, Think-SFT vs
  Think-DPO 3–22. So the judge's preference for A over B' does not vanish at matched lengths (z ≈ 1.8 on 53 prompts): B' may carry a
  small quality cost that length does not explain, at the edge of noise, while B shows none. The draft for #2002 says so.
- 2026-10-25 11:50 CST: the four Status entries above (06:25–10:15) and the rewritten summary section were written into this file at
  11:45 from notes kept in the session's scratchpad: from 06:23 to 11:40 the session could not open any pre-existing file under the
  Desktop folder (macOS denied the process after the auto-updater removed its executable; fixed by toggling Terminal's Full Disk
  Access). New files, including every results file of today, were unaffected.
- 2026-10-26 01:25 CST: **published.** This directory and `../2026-10-22_dpo-feasibility/` pushed (commit 15374ee); the three
  checkpoints `icedduck/olmo3-7b-think-dpo-arm_{A,B,Bp}_seed1` set public with model cards and a `config.json` /
  `tokenizer_config.json` that carry the transformers 4.x keys next to the 5.x ones (the fix of 2026-10-23, so a 4.57 user does not
  get the silent θ = 10000 fallback); the per-sample generations of all six models (held-out, IFEval, GSM8K, the 805 prompts for
  run1 and the second seed) uploaded as the public dataset `icedduck/olmo3-7b-think-dpo-arms-eval-outputs` (28 files, 226 MB).
