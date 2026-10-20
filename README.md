# olmo-rlvr-side-effects

**Which RLVR training data changes what, outside the training domain?**

Ai2 trains its Olmo reasoning models with RL on verifiable rewards (RLVR): math, code, instruction following and general chat. Training is checked on the skills it targets. What else moves (refusals, over-refusals, answer length, following new kinds of instructions) often gets noticed late. Ai2 flagged Olmo 3.1's wordiness as a P0 for the next release ([open-instruct #2002](https://github.com/allenai/open-instruct/issues/2002)), and the eval battery it runs while building Olmo 3.5 has no refusal or over-refusal test yet ([#1964](https://github.com/allenai/open-instruct/issues/1964): "the battery has neither").

Ai2 releases every post-training stage of Olmo 3 Think (SFT, DPO, and RL with intermediate checkpoints) and its own eval code, so you can see where along training a behaviour changes, measured the way Ai2 measures it. Which data causes a change is then tested with small runs of my own, using Ai2's recipe ([open-instruct](https://github.com/allenai/open-instruct)) and changing one data type at a time.

Independent work by [Qirun Li](https://github.com/bethehand). Not affiliated with Ai2.

## Status (2026-10-20)
- **Done, shared with Ai2 (#1964):** Ai2's own safety protocol (OLMES `safety_reasoning::olmo3` → allenai/safety-eval) on the Olmo 3 7B Think stages, 3 runs each, for the three tests #1964 names (XSTest, WildJailbreak-benign, WildGuardTest). All means are within 1 point of the Olmo 3 report's Table 52. Run-to-run noise and the smallest change each test can detect are measured too. Reruns without a seed can come out byte-identical. [`experiments/2026-10-18_think7b-safety`](experiments/2026-10-18_think7b-safety) · [comment](https://github.com/allenai/open-instruct/issues/1964#issuecomment-6247442515)
- **Done, shared with Ai2 (#2002):** answer length across the SFT, DPO and final stages of Olmo 3 7B Think, Olmo 3.1 32B Instruct and Olmo 3 / 3.1 32B Think, with Ai2's AlpacaEval setting and system-prompt controls. DPO adds most of the length (1.3–1.6× per prompt). 3.1's longer RL makes answers about 10% longer, which 3.1's new default system prompt hides. [`experiments/2026-10-18_length`](experiments/2026-10-18_length) · [comment](https://github.com/allenai/open-instruct/issues/2002#issuecomment-6256732503)
- **Done, shared with Ai2 (#2001):** links and unprompted self-mentions in the same 14 AlpacaEval runs. About 1 in 3 of the links that could be checked is broken, at every stage (433 of 1,332). With 3.1's release system prompt, 32B Think brings up Olmo, Ai2 or its cutoff in 42 of 805 answers to unrelated prompts and 32B Instruct in 31; with the earlier stage's prompt, 0 for both. Some of the bad links are in the 58 identity rows of the public SFT data. [`experiments/2026-10-19_olmo-voice`](experiments/2026-10-19_olmo-voice) · [comment](https://github.com/allenai/open-instruct/issues/2001#issuecomment-6272692332)
- **Done, shared with Ai2 (#2001 follow-up, #1948):** the probe. Every 32B stage on the 805 AlpacaEval prompts with 3.1's release system prompt: the habit is already there after SFT (unrelated answers that bring up Olmo, Ai2 or the cutoff: Think 29 → 26 → 29 → 44, Instruct 14 → 43 → 31), 0 with each stage's own prompt, and a "Robot / Acme Corp." control in the same slot leaks 2.4–10× less. The 7B Think stages don't call themselves Olmo or Ai2 on the probe even with the release sentence (0 of 15 identity answers). 29% of the checkable links in these runs are broken. Under the release sentence every 32B checkpoint also gets its own size wrong ("7 billion" in 15 of 21 answers, 32B in 0), the Think ones matching the `olmo-7b` rows of the identity data. [`experiments/2026-10-20_identity-probe`](experiments/2026-10-20_identity-probe) · [#2001 comment](https://github.com/allenai/open-instruct/issues/2001#issuecomment-6293946727) · [#1948 comment](https://github.com/allenai/open-instruct/issues/1948#issuecomment-6294469634)
- **Done:** step 0. On RL-Zero, refusal of harmful prompts is off the floor and moves with the data type; over-refusal is at the floor. [`experiments/2026-10-18_step0-floor-check`](experiments/2026-10-18_step0-floor-check)
- The pre-registration in [`prereg/`](prereg/) is on hold while the plan is re-based on the Think stages. Predictions get written and frozen with a git tag before the main runs.
- **Next:** the other six Table 52 safety tests on the same stages, then the curve along the Think RL run (plan step 2) and the controlled 1B runs (step 4).

## Plan
1. **Replicate:** Ai2's safety numbers for the Olmo 3 7B Think stages (Table 52), with run-to-run noise and the smallest change a given test size can detect; answer length per stage. *Done for the three #1964 tests and for answer length; the other six Table 52 tests are next.*
2. **Curve:** about 8 intermediate checkpoints of the Think RL run, to see where along RL the changes happen.
3. **Auxiliary:** the RL-Zero single-domain series (Math, Code, IF, General), on measures that aren't at their floor.
4. **Controlled runs:** OLMo 2 1B with open-instruct's GRPO. Math only, instruction following only, and the official mix, 2 seeds each, to see which data causes which change.
5. **Battery:** package the out-of-domain tests so they can run during training, and offer them to open-instruct / OLMES.

## Keeping it honest
- Predictions are written down before each run (`prereg/`) and frozen with a git tag.
- Anything added later goes to `addenda/` and never edits a frozen file.
- Every experiment gets a dated folder in `experiments/` with its configs, logs and results.

## Related work
- Goodfire traced a safety regression in Olmo 3's preference training back to individual preference examples ([Ai2 blog](https://allenai.org/blog/goodfire-olmo), [paper](https://www.goodfire.com/research/predictive-data-debugging)). This repo asks a similar question for RLVR data.
- Bharadwaj & Kirk followed eval awareness through Olmo 3's training ([LessWrong](https://www.lesswrong.com/posts/c2tqL9xPbttisAHtt/tracing-eval-awareness-emergence-through-training-of-olmo-3)). Cairns compared safety across the 32B model's training stages (LessWrong, 2026-09-11).

## Layout

| Path | What's there |
|---|---|
| [`prereg/`](prereg/) | Pre-registrations, one per test |
| [`experiments/`](experiments/) | One dated folder per experiment |
| [`battery/`](battery/) | The out-of-domain test battery (coming) |
| [`addenda/`](addenda/) | Notes added after a freeze |

Fixes I make to Ai2's code along the way are listed in [olmo-hands-on](https://github.com/bethehand/olmo-hands-on).
