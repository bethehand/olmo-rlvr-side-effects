# Answer length across Olmo 3 / 3.1 post-training stages (for open-instruct #2002)

**Question.** open-instruct #2002 says: "Olmo 3.1 is excessively verbose. This is a P0 issue for the next release", and plans to test checkpoints to find where it starts. We measure where answer length grows along SFT → DPO → RL:
- first, the 7B Think stages;
- then the 32B stages: 3.1 Instruct SFT/DPO/final, and 3 and 3.1 Think, which share SFT and DPO;
- then the Olmo 3.1 7B RL-Zero models.

**Protocol: Ai2's.** We follow OLMES `alpaca_eval_v3::olmo3:adapt`, the AlpacaEval setting in Ai2's `olmo3:adapt` suite:
- 805 AlpacaEval prompts, in the model's own chat template;
- temperature 0.6, top_p 0.95, one sample per prompt, no stop sequences;
- **Generation is capped at 32K**, as in Ai2's own evals. The Olmo 3 report uses "a 32K max context length, a sampling temperature of 0.6 and top-p of 0.95" for all post-trained models. open-instruct's release-eval scripts pass `--oe_eval_max_length 32768 --process_output r1_style --use_hf_tokenizer_template`. The task config alone asks for 131,072 new tokens, but the launcher overrides it.

The answer is the text after the last `</think>`, following OLMES's `r1_style` processor; if there is no `</think>`, the whole output counts as the answer.

**Measures**
- **Answer length**, in characters (what AlpacaEval's `avg_length` counts) and in tokens.
- **Thinking length**, in tokens.
- **Share of outputs that run to the end of the window** (`finish_reason == "length"`, mostly repetition loops). We report this separately, because a few such outputs dominate a mean.

We don't compute the win rate: it needs a paid GPT-4 judge, and #2002 is about length.

**Differences from Ai2's runs**
- **Our 32,768 counts new tokens only; Ai2's 32K is the whole context**, prompt plus output. This only changes where a runaway output stops, by the prompt's length, and those outputs are reported separately anyway. The longest output that finished normally is 26K tokens. (A first attempt that followed the task config, with no cap and the full 65K window, stalled on 2× 24 GB. A few looping outputs held the KV cache while the normal prompts waited behind them.)
- vLLM `gpu_memory_utilization` is 0.85, for 24 GB cards.
- Software is safety-eval's environment: vLLM 0.11.0, transformers 4.57.1.
- Seeds: from 2026-10-18 17:00 on, a run named `runN` uses seed N (`SEED` / `--seed`; each output row records it). Earlier runs used vLLM's default seed (0), as in Ai2's code: the 7B stages, the control, the DPO repeat, and 32B Instruct SFT and DPO. The DPO repeat was checked to be distinct from the first run (0/805 identical outputs).
- 32B on 4× 24 GB: `GPU_MEM=0.80 MAX_SEQS=64` (vLLM's defaults run out of memory at start-up), and the 3.1 32B Instruct SFT/DPO checkpoints, stored in fp32, are loaded in bf16.
- Each 7B model runs on 2× RTX 4090.

**First check: 20 prompts, Think-SFT.** 19 of 20 outputs closed `</think>`; 1 looped to the end of the window (65,457 tokens). The median output was 1,550 tokens in total; the median answer was 708 tokens, or 2,811 characters.

**Results: Olmo 3 7B Think stages (2026-10-18).** 805 AlpacaEval prompts, one run per stage, plus a repeat of DPO and a system-prompt control. Full output: `results_7b.txt` (from `analyze_length.py`).

| Run | Hit the 32K cap | Answer chars, median [95% CI] | Answer tokens, median | Thinking tokens, median / mean | OLMES-style mean chars* |
|---|---|---|---|---|---|
| SFT | 3.0% | 2,490 [2,316, 2,603] | 589 | 941 / 1,576 | 6,114 |
| SFT with DPO's system prompt (control) | 2.0% | 2,413 [2,286, 2,552] | 567 | 840 / 1,424 | 4,762 |
| DPO | 1.0% | 3,652 [3,538, 3,856] | 874 | 1,163 / 1,756 | 4,691 |
| DPO, repeat | 0.9% | 3,653 [3,480, 3,842] | 872 | 1,133 / 1,697 | 4,178 |
| Final (RL) | 0.9% | 2,812 [2,678, 3,002] | 656 | 2,220 / 2,973 | 3,813 |

\*Mean over all outputs, with the whole output counted as the answer when `</think>` never closes (OLMES `r1_style`). The few outputs that hit the cap dominate it.

Paired over prompts (median per-prompt change [95% CI]):
- **System prompt only** (SFT → SFT with DPO's prompt): answer −14 chars [−45, 13]; thinking −78 tokens. The control's prompt was checked by rendering it.
- **DPO training, same prompt** (SFT with DPO's prompt → DPO): answer +1,103 chars [1,045, 1,184]; thinking +296 tokens.
- **Repeat** (DPO → DPO): answer −14 [−48, 17]; thinking −16 [−49, 6]. The two DPO runs share no identical output (median shared prefix 74 chars), so this is real run-to-run noise.
- **RL** (DPO → final): answer −633 chars [−697, −579]; thinking +919 tokens [823, 1,018].

What it says, at 7B:
- **DPO makes the final answers about 1.5× as long**: the median per-prompt ratio is 1.53, and answers get longer on 96% of prompts. Per answer, it adds lines (31 → 38), list items (16 → 20) and bold spans (19 → 26), counted over the 768 prompts finished in all three stages. The SFT checkpoint's different default system prompt doesn't explain it.
- **RL trims the answers by about 0.6K characters (shorter on 87% of prompts) but nearly doubles the thinking** (longer on 88%; the mean goes from 1.7K to 3.0K tokens).
- **Outputs that run to the cap fall from 3% (SFT) to about 1%.** Two very short DPO outputs (95 and 127 tokens) stop without closing `</think>`; they count neither as finished nor as runaway. One DPO output has two `</think>` tags; as in `r1_style`, its answer is the text after the last one.

**Results: Olmo 3.1 32B Instruct stages (2026-10-18).** Same 805 prompts, one run per stage. SFT and DPO are fp32 checkpoints, loaded in bf16. Full output: `results_32b_instruct.txt`.

| Stage | Hit the 32K cap | Answer chars, median [95% CI] | Answer tokens, median | Default system prompt |
|---|---|---|---|---|
| SFT | 0.4% | 1,084 [943, 1,251] | 242 | "You are a helpful function-calling AI assistant…" |
| DPO | 0% | 2,387 [2,244, 2,534] | 544 | same as SFT |
| Final (RL) | 0% | 2,152 [2,018, 2,345] | 493 | "You are Olmo, a helpful AI assistant built by Ai2…" |

- **DPO** (same system prompt on both sides): +851 chars per prompt [757, 954]. The median per-prompt ratio is 1.64 and answers get longer on 92% of prompts. Per answer, lines go 18 → 35, list items 7 → 14, bold spans 7.5 → 16 and headers 1.9 → 4.6, counted over the 802 prompts finished in all three stages.
- **RL** (DPO → final): −48 chars [−77, −20], about flat, with fewer bold spans (16 → 11) and headers (4.6 → 3.3). The final checkpoint ships a different default system prompt, so this step is not a clean comparison.
- It matches 7B: at both sizes, most of the added length comes from DPO.

The 7B results are Olmo 3, so for #2002, which is about Olmo 3.1, they are related evidence only. The Olmo 3 / 3.1 32B Think stages run next.

**Run.** `run_length.sh MODEL_REPO REVISION RUN_NAME GPUS [LIMIT]`. The outputs are answers to benign AlpacaEval prompts; they stay on the GPU machine for now.
