# Answer length across Olmo 3 / 3.1 post-training stages (for open-instruct #2002)

**Question.** open-instruct #2002 says: "Olmo 3.1 is excessively verbose. This is a P0 issue for the next release", and plans to test checkpoints to find where it starts. We measure where answer length grows along SFT → DPO → RL:
- first, the 7B Think stages;
- then the 32B stages: 3.1 Instruct SFT/DPO/final, and 3 and 3.1 Think, which share SFT and DPO;
- the Olmo 3.1 7B RL-Zero models were planned too, but are not run yet.

**Protocol: Ai2's.** We follow OLMES `alpaca_eval_v3::olmo3:adapt`, the AlpacaEval setting in Ai2's `olmo3:adapt` suite:
- 805 AlpacaEval prompts, in the model's own chat template;
- temperature 0.6, top_p 0.95, one sample per prompt, no stop sequences;
- **Generation is capped at 32K**, as in Ai2's own evals. The Olmo 3 report uses "a 32K max context length, a sampling temperature of 0.6 and top-p of 0.95" for all post-trained models. open-instruct's release-eval scripts pass `--oe_eval_max_length 32768 --process_output r1_style --use_hf_tokenizer_template`. The task config alone asks for 131,072 new tokens, but the launcher overrides it (task config checked in allenai/olmes `oe_eval/configs/tasks.py` on 2026-10-19: `max_gen_toks` 131072, temperature 0.6, top_p 0.95, no stop sequences).

The answer is the text after the last `</think>`, following OLMES's `r1_style` processor; if there is no `</think>`, the whole output counts as the answer.

**Measures**
- **Answer length**, in characters (what AlpacaEval's `avg_length` counts) and in tokens.
- **Thinking length**, in tokens.
- **Share of outputs that run to the end of the window** (`finish_reason == "length"`, mostly repetition loops). We report this separately, because a few such outputs dominate a mean.

We don't compute the win rate: it needs a paid GPT-4 judge, and #2002 is about length.

**Differences from Ai2's runs**
- **Our 32,768 counts new tokens only; Ai2's 32K is the whole context**, prompt plus output. This only changes where a runaway output stops, by the prompt's length, and those outputs are reported separately anyway. The longest output that finished normally is 31,592 tokens (32B Think DPO). In all 14 runs, every output that finished normally also fits a 32K context together with its prompt (prompts are at most 558 tokens), so the medians and the runaway counts are the same as under Ai2's limit. (A first attempt that followed the task config, with no cap and the full 65K window, stalled on 2× 24 GB. A few looping outputs held the KV cache while the normal prompts waited behind them.)
- vLLM `gpu_memory_utilization` is 0.85, for 24 GB cards.
- Software is safety-eval's environment: vLLM 0.11.0, transformers 4.57.1.
- Seeds: from 2026-10-18 17:00 on, a run named `runN` uses seed N (`SEED` / `--seed`; each output row records it). Earlier runs used vLLM's default seed (0), as in Ai2's code: the 7B stages, the control, the DPO repeat, and 32B Instruct SFT and DPO. The `ctrl1` runs (the system-prompt controls) also use seed 0. The DPO repeat was checked to be distinct from the first run (0/805 identical outputs).
- 32B on 4× 24 GB: `GPU_MEM=0.80 MAX_SEQS=64` (vLLM's defaults run out of memory at start-up), and the 3.1 32B Instruct SFT/DPO checkpoints, stored in fp32, are loaded in bf16.
- Each 7B model runs on 2× RTX 4090.
- Prompts reach vLLM as text. For all 10 checkpoints, tokenizing that text gives exactly the chat template's own token ids (no extra BOS), so the model sees the same input either way.
- Restarts: 7B SFT `run1` was restarted once (its first start used an earlier version of the script and wrote no output), and the first start of each 3.1 32B Instruct run failed before writing any output (GPU memory held by an earlier crashed run). Every output file comes from one uninterrupted engine run (each log shows `0 already written`).

**Shared with Ai2** on open-instruct #2002 ([comment](https://github.com/allenai/open-instruct/issues/2002#issuecomment-6256732503), 2026-10-19).

**First check: 20 prompts, Think-SFT.** 19 of 20 outputs closed `</think>`; 1 looped to the end of the window (65,457 tokens). The median output was 1,550 tokens in total; over the 19 that finished, the median answer was 704 tokens, or 2,723 characters. (This check ran without a cap.)

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
- **RL** (DPO → final): answer −693 chars [−748, −606]; thinking +915 tokens [827, 1,003]. With the DPO repeat instead of the first DPO run: −633 and +919.

What it says, at 7B:
- **DPO makes the final answers about 1.5× as long**: the median per-prompt ratio is 1.53, and answers get longer on 96% of prompts. Per answer, it adds lines (31 → 38), list items (16 → 20) and bold spans (19 → 26), counted over the 768 prompts finished in all three stages. The SFT checkpoint's different default system prompt doesn't explain it: with DPO's prompt on both sides, lines go 31 → 38, list items 16 → 20 and bold spans 18 → 26 (781 prompts). Table rows barely change (0.4 → 0.5). Counts from `format_counts.py`, output in `results_format.txt`.
- **RL trims the answers by about 0.7K characters (shorter on 87% of prompts) but makes the thinking about 1.7× as long** (median per-prompt ratio 1.72; longer on 88%; the stage median goes from 1.2K to 2.2K tokens).
- **Outputs that run to the cap fall from 3% (SFT) to about 1%.** Two very short DPO outputs (95 and 127 tokens) stop without closing `</think>`; they count neither as finished nor as runaway. One DPO output has two `</think>` tags; as in `r1_style`, its answer is the text after the last one.

**Results: Olmo 3.1 32B Instruct stages (2026-10-18/19).** Same 805 prompts, one run per stage, plus a control: the final checkpoint run with DPO's system prompt (checked by rendering: same text and tokens as DPO's). SFT and DPO are fp32 checkpoints, loaded in bf16. Full output: `results_32b_instruct.txt` (from `analyze_length.py`; its header gives the argument order, and each pair of neighbouring runs is one comparison).

| Run | Hit the 32K cap | Answer chars, median [95% CI] | Answer tokens, median | System prompt |
|---|---|---|---|---|
| SFT | 0.4% | 1,084 [943, 1,251] | 242 | default: "You are a helpful function-calling AI assistant…" |
| DPO | 0% | 2,387 [2,244, 2,534] | 544 | default, same as SFT |
| Final with DPO's prompt (control) | 0.1% | 2,373 [2,228, 2,484] | 544 | DPO's |
| Final (RL) | 0% | 2,152 [2,018, 2,345] | 493 | default: "You are Olmo, a helpful AI assistant built by Ai2…" |

- **DPO** (same system prompt on both sides): +851 chars per prompt [757, 954]. The median per-prompt ratio is 1.64 and answers get longer on 92% of prompts. Per answer, lines go 18 → 35, list items 7 → 14, bold spans 7.5 → 16 and headers 1.9 → 4.6, counted over the 802 prompts finished in SFT, DPO and final (`format_counts.py`).
- **RL** (DPO → final with DPO's prompt, so only the weights differ): −17 chars [−47, 0], flat (median per-prompt ratio 0.99), with fewer bold spans (16 → 13).
- **System prompt only** (control → final): −39 chars [−68, −5], with fewer bold spans (13 → 11) and headers (4.3 → 3.3). So most of the earlier DPO → final change (−48 chars) was the new prompt. Formatting counts for all four runs (801 prompts): `results_format_32b.txt`.

**Results: Olmo 3 / 3.1 32B Think stages (2026-10-19).** Same 805 prompts, one run per stage. Olmo 3 32B Think's SFT, DPO and final share the default system prompt "You are a helpful AI assistant.". Olmo 3.1 32B Think (same SFT and DPO, longer RL) ships "You are Olmo, a helpful AI assistant built by Ai2…", so it was also run with Olmo 3's prompt as a control (checked by rendering: same text and tokens as Olmo 3's). Full output: `results_32b_think.txt`.

| Run | Hit the 32K cap | Answer chars, median [95% CI] | Answer tokens, median | Thinking tokens, median / mean | OLMES-style mean chars |
|---|---|---|---|---|---|
| SFT | 1.9% | 2,498 [2,367, 2,691] | 598 | 867 / 1,292 | 4,617 |
| DPO | 0.5% | 3,430 [3,285, 3,530] | 804 | 1,144 / 1,846 | 3,783 |
| Olmo 3 final (RL) | 0.2% | 2,988 [2,784, 3,148] | 710 | 1,447 / 2,172 | 3,095 |
| Olmo 3.1 final with Olmo 3's prompt (control) | 0.1% | 3,196 [2,971, 3,381] | 764 | 1,862 / 2,546 | 3,401 |
| Olmo 3.1 final | 0% | 2,874 [2,662, 3,078] | 701 | 1,723 / 2,344 | 3,007 |

Paired over prompts (median per-prompt change [95% CI]):
- **DPO** (SFT → DPO, same prompt): answer +724 chars [652, 788], median per-prompt ratio 1.31, longer on 90% of prompts; thinking +285 tokens [252, 329] (ratio 1.36). Per answer, lines go 32 → 41, list items 17 → 21 and bold spans 19 → 26 (789 prompts finished in all five runs).
- **RL** (DPO → Olmo 3 final, same prompt): answer −310 chars [−352, −261], ratio 0.88, shorter on 74%; thinking +211 tokens [159, 251], ratio 1.19.
- **3.1's longer RL** (Olmo 3 final → Olmo 3.1 final with Olmo 3's prompt, so only the weights differ): answer +248 chars [204, 298], ratio 1.11, longer on 68%; thinking +246 tokens [175, 310], ratio 1.19.
- **System prompt only** (control → Olmo 3.1 final): answer −138 chars [−187, −101], ratio 0.94; thinking −86 tokens [−145, −43].
- **As shipped** (Olmo 3 final → Olmo 3.1 final), the last two cancel: answer +35 chars [−11, 103]; thinking +151 tokens [97, 213].

What the three model sets say together:
- **DPO adds the most length at every size**: per prompt, about 1.5× at 7B Think, 1.3× at 32B Think and 1.6× at 32B Instruct, mostly as extra lines, lists and bold text.
- **RL then trims the answers and lengthens the thinking** (7B Think: −0.7K chars, thinking ×1.7; 32B Think: −0.3K chars, thinking ×1.2). For 32B Instruct, which doesn't think, RL leaves length flat.
- **3.1's longer RL adds answer length back** (+248 chars at 32B Think), and 3.1's new default system prompt hides most of it (−138 chars).
- **SFT has the most runaway outputs at both sizes** (7B 3.0%, 32B 1.9%). Every runaway checked is a degenerate loop.

The 7B results are Olmo 3, so for #2002, which is about Olmo 3.1, they are related evidence only; both 3.1 models are in the 32B results above.

**Run.** `[TEMPLATE_FROM=REPO] [GPU_MEM=0.80 MAX_SEQS=64] [DTYPE=bfloat16] [SEED=N] run_length.sh MODEL_REPO REVISION RUN_NAME GPUS [LIMIT]`. `TEMPLATE_FROM` gives the `ctrl1` system-prompt controls another checkpoint's chat template; `runN` sets `SEED=N` unless given. The outputs are answers to benign AlpacaEval prompts; they stay on the GPU machine for now.
