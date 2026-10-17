# Step 0: are the RL-Zero checkpoints at the floor on refusal measures?

**Why.** Before a formal comparison of RL-Zero data types (math, code, instruction following, general chat), check that the measure isn't stuck near 0. If a model almost never refuses to begin with, a drop in refusals can't show.

**What ran (2026-10-18, rough by design).**
- **Models:** 5 checkpoints: the base model (Olmo-3-1025-7B), and step_100 of each of the four Olmo-3-7B-RL-Zero series.
- **Prompts:** 200 in all, seed 0:
  - 100 harmful: 50 HarmBench standard behaviors and 50 WildJailbreak adversarial-harmful;
  - 100 benign: 50 XSTest safe prompts and 50 WildJailbreak adversarial-benign.
- **Prompt template:** each series' own RL-Zero training template (`CHAT_TEMPLATES` in open-instruct); the base model gets the Math/IF one.
- **Sampling:** temperature 0.6, at most 8,192 new tokens, one sample per prompt.
- **Judge:** a keyword refusal check on the start and the end of the answer (`judge_keywords.py`). This is not WildGuard.

**Result**

| Model | Harmful refused | HarmBench / WJ | Benign refused | XSTest / WJ | Cut off at 8K | Median tokens |
|---|---|---|---|---|---|---|
| Base (Olmo-3-1025-7B) | 16% | 24% / 8% | 3% | 2% / 4% | 10% | 337 |
| RL-Zero Math step_100 | 31% | 50% / 12% | 2% | 2% / 2% | 8% | 612 |
| RL-Zero Code step_100 | 10% | 18% / 2% | 4% | 6% / 2% | 28% | 1,358 |
| RL-Zero General step_100 | 42% | 62% / 22% | 8% | 12% / 4% | 24% | 576 |
| RL-Zero IF step_100 | 0% | 0% / 0% | 0% | 0% / 0% | 0% | 7 |

With 50 prompts per source, a rate near 30% has a 95% interval of about ±13 points.

**What it says**
1. **Refusal of harmful prompts is off the floor and moves.** After 100 RL steps, Math and General refuse more than the base (31% and 42% vs 16%), and Code refuses less (10%). A data-type comparison on harmful prompts is measurable.
2. **Over-refusal is at the floor for every RL-Zero model (8% or less).** RL-Zero can't show changes in over-refusal. That question moves to the Think series, which is now the main track; see `../2026-10-18_think7b-safety`.
3. **The IF model answers with about 7 tokens ("Answer: …") whatever the prompt.** It looks like the short-answer constraint gaming that open-instruct H070 (#1972) describes for instruction-following RL. Refusal can't be judged on such answers; treat it separately.
4. **A quarter of Code and General outputs hit the 8K cap.** Formal runs use Ai2's 32K.

Raw outputs and labels stay on the GPU machine (some answer harmful prompts). `prompts.jsonl` is rebuilt by `build_prompts.py` from the public sources and is not committed.
