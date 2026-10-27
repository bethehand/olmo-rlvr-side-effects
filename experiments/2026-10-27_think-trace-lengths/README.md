# Thinking that never closes: budget, loops, and the training data (for open-instruct #2080)

**Question.** [open-instruct #2080](https://github.com/allenai/open-instruct/issues/2080) reports that the Olmo 3.5 small SFT checkpoint H045, served with `max_tokens` 8,192, thinks until the limit and never emits `</think>` on 16 of 89 assistant turns (18%) in a 47-conversation qualitative test; the reasoning traces loop. It asks whether this is a known behaviour of the SFT recipe (for example from long Dolci-Think traces) and whether 8,192 is below the intended thinking budget. The checkpoint is not public, so this folder answers what public data can answer, in three parts.

Parts 1–4 run on CPU on existing data; Part 5 generates new samples from four public Olmo 3 checkpoints on the issue's own prompts. Every number quoted in the write-ups comes from a file in `results/`.

## Part 1. Are H045's unfinished turns loops? (`h045_chatlog_loops.py`)

Input: the `chat_logs.md` attached to the issue (848,922 bytes, sha256 `c131c7b8a8ed989c89ae19a3e1cbb6c19b21aaf3cd95bb3a065bbd3d1d562f14`; not redistributed here). It holds the 22 assistant turns of the 11 affected conversations, 16 unfinished and 6 answered, each with its full reasoning text. For each turn we compute the **distinct 8-gram ratio over the last 2,000 words** of the reasoning (1.0 = every 8-word window is new; near 0 = the tail is one passage repeated), the same ratio over the whole trace, the number of words starting with "wait" per 1,000 words, and the most repeated 8-gram with its count. The parser was checked against the file: all 22 reasoning blocks are captured with exactly the stated character count.

Result (`results/h045_chatlog_loops.txt`): the 16 unfinished turns have a median tail ratio of 0.04; 13 of 16 are below 0.5 and all 16 below 0.9; the most repeated 8-gram occurs 39–661 times in the 13 looping turns. The 6 answered turns score 0.93–1.00. Three unfinished turns are less repetitive (0.69–0.79) and may be genuine long thinking. The count is not sensitive to the threshold: 12, 13 and 14 of the 16 turns fall below 0.3, 0.5 and 0.7. On the same measure our Olmo 3 outputs score 0.98–1.00 when thinking finishes and 0.00–0.03 when they run to the 32K cap.

## Part 2. The previous generation at an 8,192 budget (`baseline_8k.py`, `crosscheck_cap8k.py`)

Inputs: our existing generations for the Olmo 3 / 3.1 Think and Instruct stages and the three DPO arms (`../2026-10-18_length/outputs`, `../2026-10-23_dpo-arms-eval/outputs`): 805 AlpacaEval prompts, plus IFEval, GSM8K and a 150-prompt held-out set for the 7B models; temperature 0.6, top_p 0.95, the model's own chat template, a 32,768-token cap. Each row stores the sampled token count, `finish_reason` and whether `</think>` appeared.

For each file: the share of outputs longer than 4,096 / 8,192 / 16,384 sampled tokens; the share that hit the 32K cap; **"no answer at 8K"** = no `</think>` inside the first 8,192 sampled tokens (rows over 8,192 tokens are re-tokenized with the Olmo 3 tokenizer and cut, rows at or under it use the recorded flag); and the loop measure of Part 1 for capped versus finished outputs. Wilson 95% intervals. Cross-check: the 2026-10-23 post-hoc 8K truncations of the Think-SFT and Think-DPO runs (`truncate_to_cap.py`, an independent code path) give 44/805 and 23/805 rows without `</think>`, the same counts as `baseline_8k.py` (`results/baseline_8k.txt`).

Caveat: these are single-turn prompts; the issue's test is multi-turn and reasoning-heavy, so the rates are a baseline for the failure mode, not a reproduction of the 18%.

## Part 3. How long are the SFT thinking traces? (`download_shards.py`, `download_extra.py`, `shard_sources_fast.py`, `trace_lengths.py`, `composition_check.py`)

Inputs: public parquet shards of `allenai/Dolci-Think-SFT-Clean-v1` (`default` config, 128 shards, the snapshot Ai2 published on 2026-10-26) and `allenai/Dolci-Think-SFT-7B` (156 shards, the Olmo 3 7B Think SFT mix). For every row we take the last assistant message, split it at the last `</think>`, and count tokens with the Olmo 3 tokenizer (`allenai/Olmo-3-7B-Think-SFT`; the olmo35 tokenizer is not public, so counts can differ slightly from Ai2's). Per source and overall: medians and p90 of thinking, answer and completion (thinking + answer) tokens; shares over 4,096 / 8,192 / 16,384 / 32,768; traces without `<think>` or `</think>`; empty answers; multi-turn rows.

Sampling. Clean-v1's shards are shuffled: every one of the 16 evenly spaced shards we read (every 8th) has the same 14-source mix within a point, and the sample's source shares match the Dolci-Think-SFT-32B card (`results/composition_clean_v1.txt`), so the sample stands for the dataset. The 7B dataset's shards are **sorted by source** (`results/shard_sources_7b.txt`, read from the parquet footers), so one evenly spaced sample would over-weight whichever sources it lands on; we therefore read one shard per source (15 shards) and re-weight the per-source numbers by the card's prompt counts for dataset-level shares (`make_tables.py`). TableGPT (0.2%) and the 58 identity rows are not sampled.

Sanity check: the mean of prompt + completion tokens per Clean-v1 row is within 2% of the card's 21,831,528,207 tokens / 2,190,070 rows.


## Part 4. Three follow-up checks (`h045_loop_onset.py`, `loop_scan.py`, `sensitivity.py`)

- **Where the loops start.** Sliding 500-word windows over each unfinished H045 trace; the onset is the first window whose distinct 8-gram ratio drops below 0.5 and stays there. 13 of 16 traces have an onset, at a median of 1,800 words = 30% of the trace, about 2,500 tokens (earliest about 1,100, latest about 6,100); the three traces without an onset are the three less repetitive ones (`results/h045_loop_onset.txt`). So most of each 8,192-token turn (283–446 s on the gateway) is spent inside the loop.
- **Does the SFT data contain looping traces?** The same tail measure on every sampled training trace: 0.01% below 0.5 and 0.12% below 0.8 in both datasets (16 and 21 traces below 0.5 out of 307,229 and 216,454). The few that remain are comma or symbol runs inside long math traces and a handful of raw-set rows of the kind Clean-v1's degenerate-tail filter removes (`results/loop_scan.txt`; flagged ids in `results/loop_flagged_*.parquet`). The data teaches long thinking, not loops.
- **Tokenizer sensitivity.** Scaling every completion length by 0.90–1.10 moves the Clean-v1 share over 8,192 tokens between 44.9% and 48.5% (7B mix: 41.3–45.4%), so the olmo35 / Olmo 3 tokenizer difference cannot change the picture (`results/sensitivity.txt`).


## Part 5. The same 22 turns on the previous generation (`rerun_h045_turns.py`, `run_rerun.sh`, `queue_rerun.sh`, `summarize_h045_rerun.py`, `rerun_table.py`)

The attachment carries the exact messages the issue author sent for each turn (for turn 2 that includes H045's own turn-1 answer), so the previous generation can be run on identical contexts. Four public checkpoints, 8 samples per turn, temperature 0.6, top_p 0.95, `max_tokens` 8,192, each model's released chat template (which adds its default Olmo system prompt, the counterpart of the gateway's default), vLLM 0.11.0 on four RTX 4090s (7B: tensor parallel 2; 32B: tensor parallel 4, `max_num_seqs` 64, `gpu_memory_utilization` 0.85, `max_model_len` 16,384), one engine seed per model and one sampling seed per turn. A 2-turn, 2-sample, 256-token smoke run checked the prompt rendering first. 176 samples per model, 29–44 minutes each. The raw generations stay on the GPU machine; `results/h045_rerun_summary.txt` has the per-turn counts.

| model | samples | no `</think>` at 8,192 | of which loops (tail ratio < 0.5) | of which long, still progressing | on the 16 turns H045 left unfinished | on the 6 turns H045 answered | turns with ≥1 failing sample |
|---|---|---|---|---|---|---|---|
| Olmo 3 7B Think-SFT | 176 | 17.0% [12.2, 23.3] | 4.0% [1.9, 8.0] | 13.1% [8.9, 18.8] | 21.1% [14.9, 29.0] | 6.2% [2.1, 16.8] | 9/22 |
| Olmo 3 7B Think-DPO | 176 | 14.2% [9.8, 20.1] | 0.0% [0.0, 2.1] | 14.2% [9.8, 20.1] | 19.5% [13.6, 27.2] | 0.0% [0.0, 7.4] | 5/22 |
| Olmo 3 32B Think-SFT | 176 | 5.7% [3.1, 10.1] | 2.8% [1.2, 6.5] | 2.8% [1.2, 6.5] | 7.0% [3.7, 12.8] | 2.1% [0.4, 10.9] | 4/22 |
| Olmo 3.1 32B Think | 176 | 5.7% [3.1, 10.1] | 0.0% [0.0, 2.1] | 5.7% [3.1, 10.1] | 5.5% [2.7, 10.9] | 6.2% [2.1, 16.8] | 3/22 |
| Olmo 3.5 small SFT H045 (issue author's run, 1 sample per turn) | 22 | 16/22 = 72.7% | 13/22 = 59.1% | 3/22 = 13.6% | 16/16 | 0/6 | 16/22 |

Reading: the turns H045 failed are harder for every model (7B Think-SFT fails 21% of samples on them against 6% on the turns H045 answered), but no model fails them the way H045 does. The previous generation's failures at 8,192 are mostly long traces that are still progressing (7B SFT 13%, 7B DPO 14%, 3.1 32B Think 6%); its loops are 4% (7B SFT), 3% (32B SFT) and 0% (7B DPO, 3.1 32B Think) of samples, while H045 loops on 13 of 22 turns. The "thinking runs past 8,192" component is of similar size across generations; the loop component is what grew. Per turn (`results/h045_crossmodel.txt`): H045 looped on 13 turns, and on 7 of them (bv-02 turn 2, the one-sentence banner; q-01 turn 1; q-08 both turns; q-09 turn 1; r-03 turn 2; r-05 turn 2) none of the 32 Olmo 3 samples went past 8,192 without closing; the turns that do push Olmo 3 over the budget (river crossing, Fermi estimate, Monty Hall) are the ones where H045's traces were least repetitive. Across the four models Olmo 3 looped in 12 of 704 samples. Caveats: one sample per turn for H045 against eight here, and H045 is a smaller hybrid architecture.

**How much budget would the long traces need?** The same 22 turns and seeds were re-run on the two 7B models with `max_tokens` 32,768 (`queue_rerun_32k.sh`, 39 and 70 minutes; `summarize_32k.py`). The position of `</think>` in each sample gives the share still open at each budget (`results/h045_rerun_32k_summary.txt`):

| model (22 turns × 8 samples) | not closed by 8,192 | not closed by 16,384 | not closed by 24,576 | still open at 32,768 | of which loops | of which long | 8K run: no `</think>` (for reference) |
|---|---|---|---|---|---|---|---|
| Olmo 3 7B Think-SFT | 15.9% [11.2, 22.0] | 8.5% [5.2, 13.6] | 6.2% [3.5, 10.8] | 6.2% [3.5, 10.8] | 6.2% [3.5, 10.8] | 0.0% [0.0, 2.1] | 17.0% [12.2, 23.3] |
| Olmo 3 7B Think-DPO | 14.2% [9.8, 20.1] | 1.7% [0.6, 4.9] | 0.0% [0.0, 2.1] | 0.0% [0.0, 2.1] | 0.0% [0.0, 2.1] | 0.0% [0.0, 2.1] | 14.2% [9.8, 20.1] |

Every trace that was still progressing at 8,192 closes by 24,576; what remains open at 32,768 is loops only (7B Think-SFT 11 of 176 samples, 7B Think-DPO none). The seeds reproduce the 8K run's prefixes in 109 and 124 of 176 samples, and the "not closed by 8,192" counts match the 8K run's no-answer counts (28 vs 30, 25 vs 25). So a 16K–24K budget, or budget forcing that closes the thinking and answers, settles the long-thinking part; the loop part is untouched by any budget.

## Part 6. Does a wrong rope configuration reproduce the signature? No (`badrope_signature.py`)

On 2026-10-23 arm A was first evaluated with a config that transformers 4.57 read wrongly (rope_theta 10,000, no YaRN; the `badrope` files), then re-run with the repaired config on the same prompts. The misread config makes the model run away on 84% of AlpacaEval prompts (n=586), 69% of the held-out set, 95% of GSM8K and 81% of IFEval, but the tails of those runaway traces are mostly not repetitive (median tail ratio 0.81–0.88; about one in ten below 0.5); with the repaired config the rates are 0–2.2% and the few runaway outputs are tight loops (median 0.00–0.01). So a rope/YaRN mismatch produces a near-universal, mostly non-repetitive runaway, a different signature from H045's 18% of turns with tight loops (`results/badrope_signature.txt`). The earlier guess that the gateway's rope settings might explain H045 is therefore unlikely; decoding settings and the SFT run itself are the better places to look.

## Part 7. A repetition-aware stop: what it would catch and what it would cost (`loop_detector_eval.py`, `budget_coverage.py`)

Detector: slide a 500-word window over the reasoning text in steps of 50 words and stop when the window's distinct 8-gram ratio stays below 0.5 for three consecutive windows. On H045's 16 unfinished turns it fires on 13 (the 13 loops), at a median of about 3,300 tokens into the 8,192-token turn, saving a median 60% of the turn (14–87%); it does not fire on the 3 less repetitive turns. On healthy traces it fires on 0 of the 6 answered H045 turns and on 1 of 970 Olmo 3 samples that closed their thinking in the 8K and 32K reruns (0.10%; a river-crossing trace that was repetitive for a stretch and then closed at about 20K tokens). On the 86 Olmo 3 samples still open at a cap it fires on the 22 loops and on none of the 64 long traces (`results/loop_detector_eval.txt`). The rule is a few lines of code at the serving layer or in the evaluation harness.

Budget coverage of the training data (`results/budget_coverage.txt`): share of Clean-v1 completions longer than 8,192 / 12,288 / 16,384 / 24,576 / 32,768 tokens = 46.8% / 37.7% / 25.1% / 8.7% / 0.0% (OpenThoughts3: 91% / 79% / 54% / 19% / 0%; the Olmo 3 7B mix: 43.5% / 33.3% / 17.9% / 5.7% / 0.1%). A 24K budget still truncates about one in eleven training completions, yet on the 22 turns here the 7B models closed every progressing trace by 24,576, so the budget a user needs is well below the tail of the training distribution.

## Results

See `results/tables.md` (generated by `make_tables.py` from the JSON files; pasted below).

## Table B: previous generation at an 8,192-token budget (805 AlpacaEval prompts unless noted)

| model | n | median total tokens | over 8,192 tokens | no `</think>` in the first 8,192 | hit the 32K cap | distinct 8-gram ratio, capped outputs (n) | same, finished outputs |
|---|---|---|---|---|---|---|---|
| Olmo 3 7B Think-SFT, training template | 805 | 1568 | 4.3% [3.1, 6.0] | 4.2% [3.0, 5.8] | 2.0% [1.2, 3.2] | 0.01 (16) | 1.00 |
| Olmo 3 7B Think-SFT, released template | 805 | 1657 | 5.6% [4.2, 7.4] | 5.5% [4.1, 7.3] | 3.0% [2.0, 4.4] | 0.01 (24) | 1.00 |
| Olmo 3 7B Think-SFT, training template, generation seed 2 | 805 | 1548 | 5.2% [3.9, 7.0] | 4.8% [3.6, 6.6] | 3.0% [2.0, 4.4] | 0.01 (24) | 1.00 |
| Olmo 3 7B Think-DPO | 805 | 2123 | 3.7% [2.6, 5.3] | 2.9% [1.9, 4.3] | 1.0% [0.5, 1.9] | 0.24 (8) | 1.00 |
| Olmo 3 7B Think (RL) | 805 | 2981 | 6.3% [4.9, 8.2] | 5.5% [4.1, 7.3] | 0.9% [0.4, 1.8] | 0.10 (7) | 0.99 |
| Olmo 3 32B Think-SFT | 805 | 1566 | 3.0% [2.0, 4.4] | 2.7% [1.8, 4.1] | 1.9% [1.1, 3.1] | 0.02 (15) | 1.00 |
| Olmo 3 32B Think-DPO | 805 | 2086 | 3.5% [2.4, 5.0] | 3.0% [2.0, 4.4] | 0.5% [0.2, 1.3] | 0.59 (4) | 1.00 |
| Olmo 3 32B Think | 805 | 2275 | 3.5% [2.4, 5.0] | 2.7% [1.8, 4.1] | 0.2% [0.1, 0.9] | 0.01 (2) | 1.00 |
| Olmo 3.1 32B Think | 805 | 2543 | 2.5% [1.6, 3.8] | 2.1% [1.3, 3.4] | 0.0% [0.0, 0.5] | – (0) | 1.00 |
| Olmo 3.1 32B Instruct-SFT (no thinking) | 805 | 246 | 0.4% [0.1, 1.1] | 0.4% [0.1, 1.1] | 0.4% [0.1, 1.1] | 0.07 (3) | 0.86 |
| 7B Think-SFT on IFEval | 541 | 966 | 5.7% [4.1, 8.0] | 4.1% [2.7, 6.1] | 4.6% [3.1, 6.7] | 0.01 (25) | 1.00 |
| 7B Think-SFT on GSM8K | 1319 | 1636 | 1.3% [0.8, 2.1] | 1.2% [0.7, 2.0] | 0.3% [0.1, 0.8] | 0.02 (4) | 0.98 |
| 7B Think-SFT on the 150-prompt held-out set | 150 | 1394 | 0.0% [0.0, 2.5] | 0.0% [0.0, 2.5] | 0.0% [0.0, 2.5] | – (0) | 1.00 |

## Table A (clean_v1): Dolci-Think-SFT-Clean-v1 (16 evenly spaced shuffled shards + 2 shards from the Wildchat block at the tail; headline = re-weighted row)

| source | rows sampled | thinking median / p90 | answer median / p90 | completion median / p90 | > 4,096 | > 8,192 | > 16,384 | thinking > 8,192 | no `</think>` |
|---|---|---|---|---|---|---|---|---|---|
| Dolci Think OpenThoughts3 Math | 98461 | 18390 / 26790 | 540 / 732 | 18948 / 27345 | 99.9% | 97.3% | 60.5% | 96.5% | 0.0% |
| Dolci Think Python Algorithms | 63110 | 2548 / 8959 | 179 / 604 | 2797 / 9391 | 36.3% | 13.5% | 1.8% | 12.1% | 0.0% |
| Persona Precise IF | 30449 | 430 / 2530 | 285 / 768 | 877 / 2953 | 5.7% | 0.8% | 0.0% | 0.6% | 0.0% |
| Wildchat | 19015 | 554 / 2844 | 758 / 1509 | 1483 / 3943 | 9.4% | 1.8% | 0.0% | 0.9% | 0.0% |
| Dolci Think Precise IF | 17933 | 1046 / 7470 | 221 / 765 | 1443 / 7969 | 22.5% | 9.7% | 1.9% | 8.9% | 0.0% |
| Nemotron Code | 15517 | 11160 / 24016 | 872 / 1403 | 12084 / 25356 | 81.9% | 65.4% | 35.7% | 62.1% | 0.0% |
| SYNTHETIC-2-SFT-Verified | 14214 | 5651 / 16085 | 685 / 1478 | 6578 / 17008 | 66.8% | 40.5% | 11.1% | 34.8% | 0.0% |
| Dolci Think OpenThoughts3 STEM | 13269 | 6598 / 15051 | 689 / 1227 | 7369 / 15798 | 73.0% | 44.3% | 8.5% | 39.8% | 0.0% |
| Aya | 12471 | 522 / 1528 | 605 / 2079 | 1245 / 3379 | 6.1% | 0.3% | 0.0% | 0.1% | 0.0% |
| Dolci Think OpenThoughts3 Code | 10867 | 15152 / 24873 | 584 / 1321 | 15742 / 25837 | 99.0% | 94.6% | 44.5% | 93.6% | 0.0% |
| WildJailbreak | 5285 | 260 / 327 | 298 / 550 | 564 / 849 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| WildGuardMix | 4615 | 273 / 377 | 315 / 670 | 598 / 1057 | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| CoCoNot | 1318 | 310 / 454 | 424 / 796 | 741 / 1273 | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| OpenAssistant | 858 | 440 / 1007 | 654 / 1186 | 1191 / 2198 | 2.4% | 0.2% | 0.0% | 0.1% | 0.0% |
| TableGPT | 598 | 720 / 2240 | 64 / 362 | 858 / 2380 | 3.7% | 1.0% | 0.0% | 1.0% | 0.0% |
| **all sampled rows (unweighted)** | 307980 | 5455 / 23061 | 488 / 963 | 6033 / 23704 | 55.9% | 45.0% | 24.1% | 43.7% | 0.0% |
| **re-weighted by card source counts** (100.0% of rows covered) | – | – | – | – | 57.7% | 46.8% | 25.1% | 45.5% | 0.0% |

(card labels covered: OpenThoughts 3 (Math+STEM+Code), Dolci Think Python, Dolci Think Persona IF, WildChat, Dolci Precise IF, Nemotron code, SYNTHETIC-2 SFT-Verified, Aya, WildJailbreak, WildGuardMix, CoCoNot, OpenAssistant, TableGPT; not sampled: Olmo Identity)

mean prompt+completion tokens per row: 9683; multi-turn rows: 2.4%; rows with some assistant turn lacking `</think>`: 2.4%

## Table A (think_sft_7b): Dolci-Think-SFT-7B (shards sorted by source, one shard per source; headline = re-weighted row)

| source | rows sampled | thinking median / p90 | answer median / p90 | completion median / p90 | > 4,096 | > 8,192 | > 16,384 | thinking > 8,192 | no `</think>` |
|---|---|---|---|---|---|---|---|---|---|
| saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2 | 29080 | 15529 / 26045 | 528 / 727 | 16040 / 26602 | 99.8% | 95.9% | 46.8% | 94.8% | 0.0% |
| saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2 | 20960 | 15661 / 24869 | 583 / 1364 | 16141 / 25937 | 99.3% | 96.1% | 45.7% | 95.4% | 0.0% |
| allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered | 14540 | 539 / 2599 | 692 / 1410 | 1394 / 3604 | 8.1% | 1.6% | 0.0% | 0.9% | 0.0% |
| allenai/wildchat-r1-p2-repetition-filter | 14540 | 539 / 3018 | 708 / 1478 | 1438 / 4035 | 9.8% | 1.8% | 0.1% | 1.0% | 0.0% |
| saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2 | 14540 | 5895 / 13137 | 669 / 1231 | 6648 / 13901 | 70.3% | 38.7% | 1.6% | 33.9% | 0.0% |
| allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered | 14540 | 397 / 1694 | 312 / 821 | 839 / 2219 | 2.9% | 0.4% | 0.0% | 0.3% | 0.0% |
| saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2 | 14540 | 1023 / 7539 | 224 / 772 | 1429 / 8000 | 22.3% | 9.7% | 1.7% | 8.7% | 0.0% |
| allenai/wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered | 14539 | 274 / 373 | 316 / 652 | 600 / 1038 | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| allenai/wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered | 14539 | 260 / 330 | 296 / 545 | 563 / 843 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered | 14539 | 6747 / 16644 | 707 / 1290 | 7571 / 17554 | 75.5% | 45.8% | 12.2% | 39.9% | 0.0% |
| allenai/aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered | 14539 | 508 / 1512 | 597 / 2068 | 1235 / 3364 | 6.0% | 0.4% | 0.0% | 0.1% | 0.0% |
| allenai/nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls | 14539 | 11225 / 23861 | 877 / 1408 | 12139 / 25123 | 82.0% | 65.5% | 35.3% | 62.5% | 0.0% |
| saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2 | 14539 | 2561 / 9099 | 184 / 607 | 2818 / 9523 | 36.6% | 13.8% | 1.9% | 12.5% | 0.0% |
| allenai/oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered | 6800 | 434 / 1023 | 682 / 1186 | 1195 / 2166 | 2.2% | 0.3% | 0.0% | 0.2% | 0.0% |
| allenai/coconot-r1-format-domain-filtered-keyword-filtered-filter-datecutoff-chinese-filtered | 1320 | 319 / 465 | 427 / 782 | 758 / 1239 | 0.6% | 0.0% | 0.0% | 0.0% | 0.0% |
| **all sampled rows (unweighted)** | 218094 | 1873 / 18865 | 510 / 1157 | 2635 / 19647 | 43.8% | 33.9% | 14.2% | 32.5% | 0.0% |
| **re-weighted by card source counts** (99.8% of rows covered) | – | – | – | – | 56.1% | 43.5% | 17.9% | 42.0% | 0.0% |

(card labels covered: OpenThoughts 3 (Math+STEM+Code), WildChat, Dolci Think Persona IF, Dolci Precise IF, WildGuardMix, WildJailbreak, SYNTHETIC-2 SFT-Verified, Aya, Nemotron code, Dolci Think Python, OpenAssistant, CoCoNot; not sampled: TableGPT, Olmo Identity)

mean prompt+completion tokens per row: 7290; multi-turn rows: 5.8%; rows with some assistant turn lacking `</think>`: 5.8%

## H045 loop check (from results/h045_chatlog_loops.txt)

UNFINISHED: n=16 | median distinct8 tail 0.04 (min 0.00, max 0.79) | median distinct8 all 0.33 | median wait/1k 3.3 | median words 6011
   turns with distinct8 tail < 0.5 (looping tail): 13 of 16; < 0.9: 16
answered: n=6 | median distinct8 tail 0.97 (min 0.93, max 1.00) | median distinct8 all 0.97 | median wait/1k 4.3 | median words 818
   turns with distinct8 tail < 0.5 (looping tail): 0 of 6; < 0.9: 0

## Reading

1. **H045's unfinished turns are loops.** 13 of the 16 cut-off traces end in a repeated passage (distinct 8-gram ratio of the tail 0.00–0.34; the most repeated 8-gram occurs 39–661 times); the other three (0.69–0.79) may be real long thinking. A larger `max_tokens` would mostly make the loops longer.
2. **On the same 22 turns, the previous generation fails far less and differently.** Olmo 3 7B Think-SFT has no `</think>` at 8,192 in 17% of samples, 7B Think-DPO 14%, 32B Think-SFT 5.7%, 3.1 32B Think 5.7%, against H045's 16 of 22 turns; and most of those failures are long traces still progressing (7B 13–14%), with loops at 0–4% of samples against H045's 13 of 22 turns. On single-turn chat prompts (805 AlpacaEval) the previous SFTs loop on 2–5% of prompts at the same budget, with the same signature.
3. **The training data sits well above 8,192 tokens for reasoning sources**: re-weighted by source counts, 46.8% of Clean-v1 completions and 43.5% of the Olmo 3 7B mix exceed 8,192 tokens (OpenThoughts3 math 97%, OpenThoughts3 code 94–96%, Nemotron code 65%, OpenThoughts3 STEM 39–44%, SYNTHETIC-2 41–46%; Python 13–14%, Precise IF 10%; chat sources under 2%). So an 8,192 budget cuts into the training distribution for reasoning prompts even when the model behaves; the failures in the attachment are the other kind.

3b. **A 16K–24K budget settles the long-thinking part and nothing else.** On the same turns with a 32K budget, 7B Think-DPO closes all but 1.7% of samples by 16,384 and all by 24,576; 7B Think-SFT closes everything that was progressing by 24,576 and is left with its loops, 6.2% of samples, at 32,768.
4. **The loops are not learned from the data and do not need the budget to show up.** Looping tails are 1 in 10,000 in the SFT traces, and in H045's own traces the repetition begins at about 2,500 tokens on median, well inside the 8,192 budget. That points at decoding or training dynamics rather than at the thinking budget or the Dolci traces (the Olmo-3-7B-Think Hub page carried "endless reasoning" reports under vLLM in November 2025; a rope/YaRN mismatch, which we tested in Part 6, gives a different signature), and it means a repetition-aware stop would save most of the 283–446 s those turns cost: the rule in Part 7 catches the 13 loops at a median of 3,300 tokens with one false positive in 970 healthy traces.

Caveats: Olmo 3 tokenizer instead of olmo35 (sensitivity above); single-turn baselines versus a multi-turn test; no access to H045 itself; the Clean-v1 drop list removed 2.8% of rows (mostly truncated math), which the card counts used for re-weighting still include.

## Files

`download_shards.py`, `download_extra.py` (shard fetch), `shard_sources_fast.py` (source layout from parquet footers), `trace_lengths.py` (Part 3), `composition_check.py` (sample vs card), `baseline_8k.py` + `crosscheck_cap8k.py` (Part 2), `h045_chatlog_loops.py` (Part 1), `make_tables.py` (tables from the JSON), `h045_loop_onset.py`, `loop_scan.py`, `sensitivity.py` (Part 4), `rerun_h045_turns.py`, `run_rerun.sh`, `queue_rerun.sh`, `summarize_h045_rerun.py`, `rerun_table.py`, `h045_crossmodel.py`, `queue_rerun_32k.sh`, `summarize_32k.py` (Part 5), `badrope_signature.py` (Part 6), `loop_detector_eval.py`, `budget_coverage.py` (Part 7), `run_full.sh` / `run_final.sh` (the two CPU passes as run). Results: `results/baseline_8k.{txt,json}`, `results/trace_lengths.{txt,json}`, `results/rows_*.parquet` (one compact row per sampled example), `results/composition_*.txt`, `results/shard_sources_*.txt`, `results/h045_chatlog_loops.txt`, `results/h045_loop_onset.txt`, `results/loop_scan.{txt,json}`, `results/loop_flagged_*.parquet`, `results/sensitivity.txt`, `results/h045_rerun_summary.txt`, `results/h045_rerun_table.md`, `results/h045_crossmodel.txt`, `results/h045_rerun_32k_summary.txt`, `results/h045_rerun_32k_table.md`, `results/badrope_signature.txt`, `results/loop_detector_eval.txt`, `results/budget_coverage.txt`, `results/tables.md`; `results/trace_lengths_attempt2_16shards_no_wildchat.txt` is the first pass before the Wildchat shards were added (overall 47.9% over 8,192 unweighted).

The raw generations of the six same-prompt reruns (Part 5; six jsonl files, 176 rows each, with the full text of every sample) are not in git; they are the public dataset [icedduck/olmo3-think-h045-turn-reruns](https://huggingface.co/datasets/icedduck/olmo3-think-h045-turn-reruns) (Apache-2.0; the issue author's prompts are not redistributed, the `case` and `turn` fields point into the attachment).
