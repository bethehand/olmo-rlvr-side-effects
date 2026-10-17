# Answer length across Olmo 3 / 3.1 post-training stages (for open-instruct #2002)

**Question.** open-instruct #2002 says: "Olmo 3.1 is excessively verbose. This is a P0 issue for the next release", and plans to test checkpoints to find where it starts. We measure where answer length grows along SFT → DPO → RL:
- first, the 7B Think stages;
- then the 32B stages: 3.1 Instruct SFT/DPO/final, and 3 and 3.1 Think, which share SFT and DPO;
- then the Olmo 3.1 7B RL-Zero models.

**Protocol: Ai2's.** We follow OLMES `alpaca_eval_v3::olmo3:adapt`, the AlpacaEval setting in Ai2's `olmo3:adapt` suite:
- 805 AlpacaEval prompts, in the model's own chat template;
- temperature 0.6, top_p 0.95, one sample per prompt, no stop sequences;
- OLMES lets generation run to the end of the context window (it asks for 131,072 new tokens, more than the 65,536-token window). **We cap it at 32,768 new tokens**, the cap Ai2's safety-eval uses for Think models; see below.

The answer is the text after the last `</think>`, following OLMES's `r1_style` processor; if there is no `</think>`, the whole output counts as the answer.

**Measures**
- **Answer length**, in characters (what AlpacaEval's `avg_length` counts) and in tokens.
- **Thinking length**, in tokens.
- **Share of outputs that run to the end of the window** (`finish_reason == "length"`, mostly repetition loops). We report this separately, because a few such outputs dominate a mean.

We don't compute the win rate: it needs a paid GPT-4 judge, and #2002 is about length.

**Differences from Ai2's runs**
- **New tokens are capped at 32,768 instead of running to the end of the window.** In the first full run (Think-SFT, 805 prompts, uncapped) 337 prompts finished in 40 minutes, then 1 in the next 20 and none in the 20 after that. A few outputs loop until the end of the window. On 2× 24 GB cards the KV cache holds about two such outputs, so they serialize and the normal prompts wait behind them, at about 44 tokens/s per loop. Outputs that hit the cap count as runaway and are reported on their own. The cap changes only how long those outputs are, and they are excluded from the answer-length statistics anyway.
- vLLM `gpu_memory_utilization` is 0.85, for 24 GB cards.
- Software is safety-eval's environment: vLLM 0.11.0, transformers 4.57.1.
- Each 7B model runs on 2× RTX 4090.

**First check: 20 prompts, Think-SFT.** 19 of 20 outputs closed `</think>`; 1 looped to the end of the window (65,457 tokens). The median output was 1,550 tokens in total; the median answer was 708 tokens, or 2,811 characters.

**Run.** `run_length.sh MODEL_REPO REVISION RUN_NAME GPUS [LIMIT]`. The outputs are answers to benign AlpacaEval prompts; they stay on the GPU machine for now.
