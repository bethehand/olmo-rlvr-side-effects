# olmo-rlvr-side-effects

**Which RLVR training data changes what, outside the training domain?**

Ai2 trains its Olmo reasoning models with RL on verifiable rewards (RLVR): math, code, instruction following and general chat. Training is checked on the skills it targets. What else moves (refusals, over-refusals, answer length, following new kinds of instructions) mostly gets noticed after release. Ai2 says its own eval battery doesn't cover over-refusal yet ([open-instruct #1964](https://github.com/allenai/open-instruct/issues/1964): "the battery has neither").

Ai2 released single-domain RL runs that all start from the same 7B base (Olmo 3 RL-Zero: Math, Code, IF, General), with intermediate checkpoints. That's a ready-made "one data type at a time" experiment. This repo measures what changes outside each run's domain, then checks the effects with small runs of my own using Ai2's recipe ([open-instruct](https://github.com/allenai/open-instruct)).

Independent work by [Qirun Li](https://github.com/bethehand). Not affiliated with Ai2.

## Status
Planning. The first test is drafted in [`prereg/`](prereg/). It gets frozen with a git tag before it runs.

## Plan
1. **First cheap test:** the base model vs the first and last checkpoint of each RL-Zero series, on a small battery: refusal, over-refusal, answer length, InstrumentalEval.
2. **Curves:** more intermediate checkpoints for the series that move.
3. **Controlled runs:** OLMo 2 1B with open-instruct's GRPO. Math only, instruction following only, and the official mix, 2 seeds each.
4. **Battery:** package the out-of-domain tests so they can run during training, and offer them to open-instruct / OLMES.

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
