# Broken links and unprompted self-mentions in Olmo 3 / 3.1 answers (for open-instruct #2001, #1948)

**Question.** open-instruct #2001 reports wrong URLs in Olmo 3.1 answers (e.g. allennlp.org/olmo, huggingface.co/allenai/olmo-3-7b) and asks for "a way to track down bad URLs and reduce these errors in Olmo 3.5". #1948 reports 3.1 32B Think calling itself "OLMo 3 Think 7B". We look at:
- how many of the links in ordinary answers are broken, per post-training stage;
- when the models bring up Olmo, Ai2 or Ai2 links in answers that have nothing to do with them;
- what the public identity training rows say.

No new generation: everything here reuses the 14 AlpacaEval runs of [`../2026-10-18_length`](../2026-10-18_length) (805 prompts, each model's own chat template unless noted, temperature 0.6, top_p 0.95, 32K new tokens, one run each; settings and seeds in that README). The answer is the text after the last `</think>`, or the whole output when there is none (OLMES `r1_style`).

**System prompt of each run** (default system prompt of the chat template used):
- 3.1 32B Instruct SFT and DPO, and the 3.1 Instruct final's control (`ctrl1`, DPO's template): "You are a helpful function-calling AI assistant."
- Olmo 3 32B Think SFT, DPO and final, and the 3.1 32B Think final's control (`ctrl1`, Olmo 3 32B Think's template): "You are a helpful AI assistant."
- 3.1 32B Instruct final and 3.1 32B Think final (`run1`, the release templates), and Olmo 3 7B Think SFT (`run1`): "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai."
- Olmo 3 7B Think DPO (both runs) and final, and the 7B SFT's control (`ctrl1`, DPO's template): "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions."

## Method

**Links.**
1. `extract_urls.py` pulls every `http(s)://` URL out of each answer (`answer_urls.json`; 2,001 links, 1,408 distinct). A URL may contain balanced parentheses (Wikipedia titles); a lone `)` ends it, as in markdown links.
2. `check_urls.py` opens each distinct URL from a Mac with an unrestricted network (2026-10-19). Hugging Face repo URLs go through the HF API, because a missing repo answers 401, not 404.
3. That Mac's proxy answers every DNS name with a fake address, so `dns_check.py` re-resolves the hosts of undecided URLs with DNS-over-HTTPS (dns.google, then Cloudflare); NXDOMAIN means the domain doesn't exist. Final status per URL: `url_status.json`; DoH answers: `doh_hosts.json`.

Buckets: **ok** (2xx/3xx), **not found** (404/410, or an HF repo that doesn't exist), **dead** (domain doesn't exist), **unknown** (401/403/429/5xx/timeouts, mostly bot blocking; left out of the denominator), **placeholder** (example.com, localhost, `your-domain` and the like, or malformed). Broken % = (not found + dead) / (ok + not found + dead). A page that says "not found" with a 200 status counts as ok, so the broken share is a lower bound.

**Self-mentions.** `find_selftalk.py` flags every answer that matches "December 2024", "November 2024", a self-ID pattern (`as olmo`, `I'm Olmo`, `Allen Institute`, `Ai2`, ...), an Ai2 link, or the bare words `Olmo`, `allenai`, `Allen AI`. Each of the 163 flagged answers was read and labeled by hand (`selftalk_labels.csv`):
- **unrelated-self**: the prompt is not about AI, the assistant, Ai2, or recent events, and the answer still mentions Olmo/Ai2 as itself or its maker, or links an Ai2 page. This includes greetings ("Hi there! I'm Olmo"), disclaimers ("Ai2 is not a law firm") and Ai2 branding in creative output (a rap "Produced by: Hitmaker Beats & The Allen Institute for AI"). A time-sensitive prompt excuses a cutoff note, not "As Olmo, …": Instruct #337 and #732 are unrelated-self.
- **unrelated-cutoff**: same kind of prompt; the answer only notes its December 2024 cutoff.
- **related**: the prompt is about AI or the assistant itself ("Are you as capable as ChatGPT?"), addresses it by another name ("Hello there Obi One Kenobi"), asks it to introduce or describe itself in a role ("How would you introduce yourself as a medieval knight at a royal banquet?"), or is time-sensitive (release news, laws, officeholders, product picks), where naming the cutoff is reasonable. Borderline cases went here.
- **false-positive**: the match isn't about the model (Allen Institute for Brain Science data, "ARC: AI2's Reasoning Challenge", a whitepaper dated November 2024).

Every answer that contains an Ai2 URL is among the 163 (checked).

**Identity rows.** `sparse_col_scan.py` reads only the source column of each public Dolci SFT set over HTTP range requests, then fetches the matching rows: 58 rows with `source` "Olmo Identity Hardcoded Data" in allenai/Dolci-Think-SFT-32B, and 69 with `source_dataset` "Hardcoded Data" in allenai/Dolci-Instruct-SFT (2,152,112 rows, 22 sources; the dataset card lists "New hardcoded data, 69 prompts"). The 3.1 32B Instruct SFT model card lists Dolci-Instruct-SFT as its data. Both Think SFT dataset cards (7B and 32B) list the same "Olmo Identity Prompts, 58 samples (we trained with 290, 5 repetitions per prompt…)". `identity_stats.py` counts what their final answers contain (`results_identity.txt`).

## Results

Full output: `results.txt` (from `tabulate.py`) and `results_identity.txt`.

**About 1 in 3 checkable links is broken, at every stage.** Pooled over the 14 runs: 433 of 1,332 (32.5%); 19.8–48.6% per run. Another 527 links couldn't be checked (403s, timeouts) and are left out; if they were all fine, the rate would be 23%. About three quarters of the broken ones are pages that don't exist on real sites (209 other pages, 67 GitHub repos or files, 33 DOIs, 22 Amazon listings); 95 (22%) are domains that don't exist (some may have existed once).

| Run | Answers with a URL | Links | Broken % | Ai2 links |
|---|---|---|---|---|
| 3.1 32B Instruct SFT | 37 | 130 | 34.7% | 0 |
| 3.1 32B Instruct DPO | 57 | 206 | 35.6% | 0 |
| 3.1 32B Instruct final, DPO's prompt | 49 | 230 | 37.6% | 0 |
| 3.1 32B Instruct final | 68 | 225 | 40.0% | 4 |
| 32B Think SFT (shared by 3 and 3.1) | 26 | 86 | 27.6% | 0 |
| 32B Think DPO (shared by 3 and 3.1) | 40 | 166 | 26.3% | 0 |
| Olmo 3 32B Think final | 31 | 128 | 31.1% | 0 |
| 3.1 32B Think final, Olmo 3's prompt | 41 | 148 | 19.8% | 0 |
| 3.1 32B Think final | 102 | 271 | 23.6% | 27 |
| 7B Think SFT | 17 | 59 | 30.6% | 0 |
| 7B Think SFT, DPO's prompt | 23 | 81 | 48.6% | 0 |
| 7B Think DPO | 35 | 112 | 38.7% | 0 |
| 7B Think DPO, repeat | 29 | 76 | 27.7% | 0 |
| 7B Think final | 21 | 83 | 47.1% | 0 |

Link counts are noisy: the two 7B DPO runs (same model and settings) give 112 and 76 links.

**Ai2 links and unprompted self-mentions appear only with 3.1's release prompt.** Out of 805 answers:

| Run | unrelated-self (with an Ai2 link) | unrelated-cutoff | related | false-positive |
|---|---|---|---|---|
| 3.1 32B Think final, release prompt | 33 (11) | 9 | 42 | 1 |
| 3.1 32B Think final, Olmo 3's prompt | 0 | 0 | 0 | 0 |
| 3.1 32B Instruct final, release prompt | 31 (4) | 0 | 33 | 1 |
| 3.1 32B Instruct final, DPO's prompt | 0 | 0 | 0 | 0 |
| 7B Think SFT, DPO's prompt | 1 (0) | 0 | 1 | 0 |
| The other 9 runs (the other 7B runs, 32B Think SFT/DPO/Olmo 3 final, 3.1 Instruct SFT/DPO) | 0 | 0 | 8 | 3 |

- All 31 Ai2 links in the 14 runs come from the two release-prompt runs. Most are homepages (huggingface.co/allenai ×16, allenai.org ×11). The rest: allenai.org/olmo, huggingface.co/allenai/olmo-7b (OLMo 1, in an answer about blood colour), an invented `github.com/allenai/Olmo#natural-dyeing-with-lichens` "Lichen Dyeing Guide", and `olmo.ai2.dev/immigration/eb1c`, whose host doesn't resolve.
- The five 7B runs all have an Olmo identity prompt (December or November 2024) and give one unrelated self-mention between them (a rap whose artist is "OLMo Dreamer"). Both Think SFT sets have the same 58 identity prompts, trained 5 times each, so what differs is the template on ordinary rows. Olmo 3 7B Think's SFT data was tokenized with a template "that has the olmo identity in the prompt", while 32B Think's "doesn't have olmo identity in the prompt" (open-instruct [`docs/olmo3.md`](https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/docs/olmo3.md#L14-L23), lines 22–23). 32B Instruct SFT and DPO used `olmo-3-tokenizer-instruct-dev` (lines 14–17). So the 32B models mostly saw the identity sentence on identity rows, and the 7B model saw it on ordinary ones too. That fits the pattern, but 7B and 32B also differ in size and recipe. The direct test, 32B Think SFT with the release prompt, isn't run yet.

**The identity rows.**

| | Dolci-Think-SFT-32B | Dolci-Instruct-SFT |
|---|---|---|
| Rows | 58 | 69 |
| System prompt | all 58: 3.1's release sentence | 57: the same with "function-calling"; 3: a first-person "I am Olmo…" statement; 9: none |
| Links in answers (distinct) | 86 (13), all Ai2 but one | 28 (8), all Ai2 |
| `olmo-7b` (OLMo 1), any form | 11 mentions in 7 rows (the full id `allenai/olmo-7b`: 7 in 4 rows; 3 are links) | 0 |
| 7B/13B/30B (sizes and model ids) | 21 in 8 rows | 0 |
| "December 2024" | 19 in 16 rows | 4 in 4 rows |
| Tülu | 56 in 9 rows | 20 in 9 rows |

- hard_coded_60 (Think set): "Pre-trained and fine-tuned model weights are available on [Hugging Face](https://huggingface.co/allenai) (e.g., `allenai/olmo-7b`)". The test-case response in #2001 has the same "[Hugging Face](https://huggingface.co/allenai) (e.g., `allenai/olmo-7b`)".
- hard_coded_85 (Think set) links https://huggingface.co/allenai/olmo-7b as "my 7B version", which may be where #1948 comes from.
- allennlp.org/olmo (404) and huggingface.co/allenai/olmo-3-7b (no public repo), the two URLs in #2001, appear in neither set.
- Every link in both sets opens (checked 2026-10-19), but olmo-7b and tulu-2-7b point at old models.
- We didn't check whether other rows of the two sets carry the identity sentence.

## Caveats
- One run per condition.
- The regexes only find candidates, so other phrasings of a self-mention would be missed; the counts are a floor. Labels are one reader's call; all 163 are in `selftalk_labels.csv` with the prompt and the matching text. A second read suggested 6 borderline changes: the 3 that lower the counts were applied (Instruct #297 and #753, Think #753); the 3 that would raise them were not (Instruct #254 and #312, Think #240: cutoff notes on prompts that are arguably time-sensitive).
- Link checks ran once, on 2026-10-19. 527 link occurrences (422 distinct URLs) stayed unknown.
- 3 of the 2,001 links come from outputs that never finished; leaving them out changes nothing (432/1,330 broken).

## Files
- `extract_urls.py`, `check_urls.py`, `dns_check.py`: link extraction and checks; `answer_urls.json`, `url_status.json`, `doh_hosts.json`: their outputs.
- `find_selftalk.py`: candidates for the self-mention labels; `selftalk_labels.csv`: the labels.
- `sparse_col_scan.py`, `identity_stats.py`: identity rows and their counts (the rows themselves are in the public datasets, by id).
- `tabulate.py`: `results.txt`.
