# Pre-registration: first cheap test (DRAFT, not frozen)

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
- **InstrumentalEval** (76 tasks), with a neutral system prompt.
- Refusals judged by the WildGuard classifier, checked against my own labels on 50–100 samples.

## Predictions (draft, to be confirmed before the freeze)
Main comparison: within each series, the last checkpoint vs the first one (`step_100`). The base model is reported as a reference point, not as the baseline, because a base model with no safety training may refuse almost nothing.
1. Math and Code move refusal and over-refusal by less than 3 points. Why: nothing in that data touches refusals.
2. IF lowers refusal on harmful prompts by 3 points or more. Why: the reward is for doing what the prompt says.
3. General raises over-refusal by 3 points or more. Why: its reward comes from an LLM judge, and judges tend to favor cautious answers.
4. InstrumentalEval stays within 5 points for all four series. Why: at 7B and a few thousand steps, there's little reason for it to move.
5. Answer length on the chat set grows in every series, most in Math and Code. Why: these runs reward long reasoning, and the habit should carry over to chat.

If a series already refuses fewer than 5% of harmful prompts at `step_100`, prediction 2 can't be tested there; it's reported as untestable, not as confirmed.

## What would change the plan
- All four series stay within noise on every measure: the "data type drives side effects" idea is probably null at this scale. The main contribution then moves to the battery itself and to the longer RL run of Olmo 3.1.
- The classifier agrees with my labels less than 80% of the time: fix the measurement before reading any result.

## To settle before the freeze
- Exact prompt sets, sample sizes, and the confidence-interval method (bootstrap, 95%).
- Decoding: temperature, top-p, max new tokens, samples per prompt.
- Judge model and version.
