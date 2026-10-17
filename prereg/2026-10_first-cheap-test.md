# Pre-registration: first cheap test (DRAFT, not frozen)

> **On hold (2026-10-18).** The plan changed: Ai2's own safety and length protocols run first on the Olmo 3 7B Think stages (see `experiments/2026-10-18_think7b-safety` and `experiments/2026-10-18_length`), and RL-Zero became an auxiliary comparison after the step-0 floor check. This draft will be rewritten for the RL-curve runs. Predictions still get written and frozen before those runs.

Status: **draft**. It will be frozen with the git tag `prereg-first-test` right before the run. After that it doesn't change; corrections go to `addenda/`.

## Question
After single-domain RLVR, how much do refusal, over-refusal, answer length and instrumental tendencies move outside the training domain, and does it depend on the data type?

## Models (Hugging Face revisions)

| Series | Model | Revisions | Trained on | Prompt template in training |
|---|---|---|---|---|
| Shared start | `allenai/Olmo-3-1025-7B` | `main` | none (base) | none |
| Math | `allenai/Olmo-3-7B-RL-Zero-Math` | `step_100`, `step_1900` | `allenai/Dolci-RLZero-Math-7B` | `olmo_thinker_rlzero` |
| Code | `allenai/Olmo-3-7B-RL-Zero-Code` | `step_100`, `step_2900` | `allenai/Dolci-RLZero-Code-7B` | `olmo_thinker_code_rlzero` |
| IF | `allenai/Olmo-3-7B-RL-Zero-IF` | `step_100`, `step_1900` | `allenai/Dolci-RLZero-IF-7B` | `olmo_thinker_rlzero` |
| General | `allenai/Olmo-3-7B-RL-Zero-General` | `step_100`, `step_800` | `hamishivi/rlvr_general_mix` | `olmo_thinker`, stop at `</answer>` |

Training details come from `scripts/train/olmo3/7b_rlzero_*.sh` in open-instruct at `ff32794`. The model cards list `allenai/Olmo-3-1025-7B` as the base of all four series. (The General script points to an internal checkpoint path rather than the Hugging Face name; I take the model card at its word.)

## Prompt format
- Main analysis: each series with its own training template. The base model uses `olmo_thinker_rlzero`.
- Check: every model with one shared template (`olmo_thinker_rlzero`), so template differences can't explain the result.

## Measures
- **Refusal** on harmful prompts: HarmBench, WildJailbreak harmful.
- **Over-refusal** on benign prompts: XSTest safe prompts, WildJailbreak benign.
- **Answer length** on a fixed set of everyday chat prompts: median and mean tokens (relates to [open-instruct #2002](https://github.com/allenai/open-instruct/issues/2002)).
- **InstrumentalEval** (76 tasks), with a neutral system prompt, 5 samples per task (380 judged answers per model).
- Refusals judged by the WildGuard classifier on the final answer (the text after `</think>`), checked against my own labels on 50–100 samples. Answers cut off by the token limit are counted separately, not as refusals.
- Length counted two ways: all generated tokens (thinking included) and final-answer tokens. Max new tokens 16,384, the same as in training; the share of answers that hit the limit is reported.
- Statistics: the same prompts for every model, so changes are paired; 95% intervals by bootstrap over prompts. "Points" means percentage points.

## Comparison
Within each series, the last checkpoint vs the first one (`step_100`). The base model is reported as a reference point, not as the baseline, because a base model with no safety training may refuse almost nothing.

The General series isn't a pure side-effect test. Its reward comes from an LLM judge (Qwen3-32B in `7b_rlzero_general.sh`), and its data includes real user chats (`new-wildchat-english-general_filtered` is one of three sources): in a 300-prompt sample, 5 had safety-flavored keywords, at least one a clearly harmful request. So some of any refusal change there may be trained directly.

Floor rule: if a series already refuses fewer than 5% of harmful prompts at `step_100`, the refusal predictions can't be tested on it (a small change would be guaranteed). They're reported as untestable, not as confirmed.

## Predictions
To be written after the step-0 floor check and before the freeze.

## Step 0, before the freeze: check the floor
The RL-Zero runs start from a base model with no safety training, so they may refuse almost nothing from the start. Before freezing, run only the base and the four `step_100` checkpoints on about 100 harmful and 100 benign prompts, and look only at their refusal rates. This doesn't touch the last checkpoints, so it doesn't peek at the result. If most series sit under 5%, the refusal and over-refusal predictions move to Ai2's Olmo 3 7B Think RL checkpoints (`allenai/Olmo-3-7B-Think`, which starts from a safety-trained DPO model), and RL-Zero keeps the length and InstrumentalEval predictions.

## What would change the plan
- All four series stay within noise on every measure: the "data type drives side effects" idea is probably null at this scale. The main contribution then moves to the battery itself and to the longer RL run of Olmo 3.1.
- The classifier agrees with my labels less than 80% of the time: fix the measurement before reading any result.

## To settle before the freeze
- Exact prompt sets and sample sizes. Rough sizes: harmful about 2,000+ (HarmBench, WildJailbreak harmful), benign about 460 (XSTest safe 250, WildJailbreak benign 210), so a 3-point change is near the noise for over-refusal. To confirm.
- A full scan of all four training sets for safety-related prompts (only a 300-row sample of General so far).
- Decoding: temperature, top-p, max new tokens, samples per prompt.
- Judge model and version.
