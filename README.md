# olmo-rlvr-side-effects

**Which RLVR training data changes what, outside the training domain?**

Ai2 trains its Olmo reasoning models with RL on verifiable rewards (RLVR): math, code, instruction following and general chat. Training is checked on the skills it targets. What else moves (refusals, over-refusals, answer length, following new kinds of instructions) often gets noticed late. Ai2 flagged Olmo 3.1's wordiness as a P0 for the next release ([open-instruct #2002](https://github.com/allenai/open-instruct/issues/2002)), and the eval battery it runs while building Olmo 3.5 has no refusal or over-refusal test yet ([#1964](https://github.com/allenai/open-instruct/issues/1964): "the battery has neither").

Ai2 releases every post-training stage of Olmo 3 Think (SFT, DPO, and RL with intermediate checkpoints) and its own eval code, so you can see where along training a behaviour changes, measured the way Ai2 measures it. Which data causes a change is then tested with small runs of my own, using Ai2's recipe ([open-instruct](https://github.com/allenai/open-instruct)) and changing one data type at a time.

Independent work by [Qirun Li](https://github.com/bethehand). Not affiliated with Ai2.

## Status (2026-10-18)
- **Running:** Ai2's own safety protocol (OLMES `safety_reasoning::olmo3` → allenai/safety-eval) on the Olmo 3 7B Think stages, 3 runs each, starting with the three tests #1964 names (XSTest, WildJailbreak-benign, WildGuardTest). The first SFT run is within 1 point of the Olmo 3 report's Table 52 on all three. [`experiments/2026-10-18_think7b-safety`](experiments/2026-10-18_think7b-safety)
- **Running:** answer length across the same stages with Ai2's AlpacaEval setting, for #2002. [`experiments/2026-10-18_length`](experiments/2026-10-18_length)
- **Done:** step 0. On RL-Zero, refusal of harmful prompts is off the floor and moves with the data type; over-refusal is at the floor. [`experiments/2026-10-18_step0-floor-check`](experiments/2026-10-18_step0-floor-check)
- The pre-registration in [`prereg/`](prereg/) is on hold while the plan is re-based on the Think stages. Predictions get written and frozen with a git tag before the main runs.

## Plan
1. **Replicate:** Ai2's safety numbers for the Olmo 3 7B Think stages (Table 52), with run-to-run noise and the smallest change a given test size can detect; answer length per stage.
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
