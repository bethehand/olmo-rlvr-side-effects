# Chosen vs rejected in the Dolci DPO data: is DPO taught to be longer? (for open-instruct #2002)

**Question.** In [`../2026-10-18_length`](../2026-10-18_length), DPO was the stage that added most of the answer length: 1.5× per prompt for Olmo 3 7B Think, 1.3× for 32B Think, 1.6× for Olmo 3.1 32B Instruct. The DPO data is public, so we can ask whether the preference pairs themselves prefer longer answers: for every pair, how much longer is the chosen answer than the rejected one, and which formatting marks (lists, bold, headings, tables, links) come with it.

**Data** (all ODC-BY, the exact sets the model cards name):

| set | pairs | used for | pair types | chosen / rejected models (most common) |
|---|---|---|---|---|
| [Dolci-Think-DPO-32B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-32B) | 200,000 | Olmo 3 32B Think DPO (shared by Olmo 3 and 3.1) | delta_learning | qwen3-reasoning-32b / qwen3-reasoning-0.6b |
| [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B) | 150,000 | Olmo 3 7B Think DPO | delta_learning | qwen3-reasoning-32b / qwen3-reasoning-0.6b |
| [Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO) | 259,922 | Olmo 3 7B Instruct DPO and Olmo 3.1 32B Instruct DPO (per the 32B model card) | delta_learning, llm_judged, two multiturn kinds | qwen3-no_reasoning-32b, gpt-4.1 / qwen3-no_reasoning-0.6b, olmo2-1b, phi4-mini, olmo2-7b, yi-9b |

Each pair is a user turn plus one assistant message on each side. In the Think sets every message on both sides has a `<think>…</think>` block followed by the answer; the Instruct set has none. The "Delta Learning" pairs (Geng et al. 2025) take the chosen answer from a strong model and the rejected one from a deliberately weak model answering the same prompt.

**Method** (`analyze_dpo.py`, CPU only, run 2026-10-20 on the full sets). For each side of each pair: the answer (text after the last `</think>`, or the whole message), the thinking block, and counts of list items, bold spans, headings, table rows, URLs, Olmo/Ai2 mentions and emoji in the answer; answer tokens with the Olmo 3 tokenizer. Per set, per preference type and per source: median chosen and rejected length, the median per-pair ratio chosen/rejected with its interquartile range, the share of pairs where chosen is longer, and the same comparisons for each formatting mark. `tabulate_dpo.py` turns the summaries into the tables below; `results/pairs_*.parquet` has one row per pair.

## Results

**The preferred answer is the longer one in most pairs, in all three sets, and by a lot in the Think sets.**

| set | pairs | chosen median | rejected median | per-pair ratio chosen/rejected, median [IQR] | chosen longer | chosen ≥ 2× | rejected longer |
|---|---|---|---|---|---|---|---|
| Dolci-Think-DPO-32B | 200,000 | 2,053 chars (560 tok) | 1,011 chars (258 tok) | 1.76 [1.20, 2.81] | 84% | 42% | 14% |
| Dolci-Think-DPO-7B | 150,000 | 2,012 (527) | 1,003 (254) | 1.77 [1.21, 2.80] | 84% | 42% | 13% |
| Dolci-Instruct-DPO | 259,922 | 1,296 (333) | 1,002 (256) | 1.28 [0.88, 2.06] | 66% | 26% | 32% |
| &nbsp;&nbsp;delta_learning half | 124,942 | 1,063 | 740 | 1.39 [0.99, 2.07] | 72% | | |
| &nbsp;&nbsp;llm_judged half | 124,980 | 1,664 | 1,490 | 1.18 [0.79, 2.11] | 60% | | |
| &nbsp;&nbsp;multiturn (two kinds) | 10,000 | 376–632 | 305–527 | 1.09–1.18 | 58–71% | | |

- In the Think sets every pair is qwen3-reasoning-32b (chosen) against qwen3-reasoning-0.6b (rejected) on the same prompt, and the gap is largest on short prompts (under 200 chars: chosen longer in 91% of pairs, median ratio 2.25; over 1,000 chars: 80%, 1.54). Every source in the mixture shows it (per-source median ratios 1.21–2.18, lowest for tulu-3-sft-personas-math and flan_v2, highest for WildChat regenerations). The chosen side's *thinking* is longer too (median 3,651 vs 2,237 chars; longer in 73% of pairs), so the whole message is 1.57× longer (chosen longer in 78%).
- In the Instruct set the bias is milder and a third of the pairs go the other way. It is stronger in the Delta Learning half (1.39) than in the LLM-judged half (`preference_type` = `llm_judged`; 1.18).
- **Formatting comes with the length.** Per answer, the chosen side has about twice the list items (Think-32B: 9.8 vs 5.2), bold spans (14.1 vs 7.2) and headings (3.3 vs 1.7), and a URL in 3.6% of answers against 1.4%; emoji and dingbat symbols (✅, ⚠️, 🚀 …) appear in 17.5% of chosen answers against 5.3% of rejected ones (the chosen side has more of them in 16% of pairs, the rejected side in 2%). The Instruct set shows the same ordering with smaller gaps (bullets 7.1 vs 5.5, bold 11.0 vs 8.2, URLs 2.8% vs 1.4%). Olmo or Ai2 appear in 19 of the 1.22 million answers (word-bounded `OLMo|Olmo|Ai2|AI2|Allen Institute` plus Ai2-domain URLs, re-scanned over the raw sets: 8 answers in Think-32B, 5 in Think-7B, 6 in Instruct; the pair tables' `c_olmo_ai2`/`r_olmo_ai2` count 18 without the URLs), none of them a self-identification: mostly "Allen Institute" named as a research organization, one allenai.org dataset link, one "run locally with OLMo", and one Instruct pair whose prompt asks about OLMo. None of the 609,922 pairs carries a system turn, so the unprompted self-mentions in [#2001](https://github.com/allenai/open-instruct/issues/2001) don't come from the DPO data.

**Against the measured DPO effect** ([`../2026-10-18_length`](../2026-10-18_length), same AlpacaEval prompts for SFT and DPO): the direction matches everywhere, the size doesn't.

| model | DPO data's median ratio | model's answer change SFT → DPO (per prompt) |
|---|---|---|
| Olmo 3 7B Think | 1.77 | +1,103 chars, ≈1.5× |
| Olmo 3 32B Think | 1.76 | +724 chars, ≈1.3× |
| Olmo 3.1 32B Instruct | 1.28 | +851 chars (1,084 → 2,387 median), ≈1.6× |

The Instruct set has the weakest length preference and the Instruct model made the biggest jump, so how far the model moves also depends on where SFT leaves it (the 32B Instruct SFT answers were the shortest, 1,084 chars median) and on the DPO run itself. What the data does fix is the direction: a DPO stage trained on these pairs is pushed toward longer, more formatted answers on every set.

**Caveats.** Lengths are of the final answer after `</think>` (whole-message numbers given for the Think sets). The DPO prompts are not the AlpacaEval prompts, so the two columns above are not the same distribution. "Longer" is not "worse": many chosen answers are more complete, and in the Delta Learning construction the weaker model simply writes less, so length is a by-product of the quality gap rather than a chosen target; length-controlled pair selection or a length penalty would be the usual ways to keep the quality signal without the length one. Formatting counts are regex counts on markdown. All numbers were recomputed on fresh random samples by a second script (`verify_dpo.py`, `verify_fmt.py`): medians and shares agree within sampling noise.

## Files
- `analyze_dpo.py`: the measurement (run on GS01 against the Hugging Face cache; ~10 min on CPU for the three sets). `tabulate_dpo.py`: the tables in `results/tables.md`.
- `check_system_olmo.py`, `list_olmo_mentions.py` (added 2026-10-21, run on the full sets): none of the 609,922 pairs has a system turn (Think sets: exactly a user and an assistant message per side; Instruct: 2–16 messages, no system role). Olmo/Ai2 is mentioned in 18 of the 1,219,844 final answers (31 assistant messages counting thinking blocks and earlier turns), never as self-identification: the ARC "AI2 Reasoning Challenge", the Allen Institute in lists of labs, a surname, a variable name, and two Instruct chats that ask about OLMo and get wrong answers. Output: `results/system_turns_and_mentions.txt`, `results/olmo_mentions_in_context.txt`.
- `results/summary_*.json`: per set, per preference type, per source and per model pair. `results/pairs_*.parquet`: one row per pair (lengths, tokens, formatting counts, models, source).
- `verify_dpo.py`, `verify_fmt.py`: independent re-checks on random samples.
