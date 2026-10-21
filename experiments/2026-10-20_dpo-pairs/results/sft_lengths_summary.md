# Answer length the SFT data teaches (Olmo 3 tokenizer, answer after the last `</think>`)

Random 100-row pages through the Hugging Face datasets-server; the sets are stored by source, so each page is one source and the sample is `pages` clusters. For Dolci-Instruct-SFT the whole-set source shares are known exactly and used as weights.

| set | pages | rows | sources hit | pooled median tokens | pooled IQR | weighted median (Instruct) | thinking median tokens |
|---|---|---|---|---|---|---|---|
| Dolci-Think-SFT-7B | 120 | 12,000 | 14 | 472 | 250–663 | – | 4394 |
| Dolci-Think-SFT-32B | 120 | 12,000 | 15 | 482 | 263–661 | – | 6720.0 |
| Dolci-Instruct-SFT | 120 | 12,000 | 21 | 188 | 62–460 | 196 (q25 64, q75 467) | – |

## Per-source answer medians (tokens)

### Dolci-Think-SFT-7B

| source | sampled rows | median tokens | exact share (Instruct only) |
|---|---|---|---|
| saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2 | 3,400 | 532 | – |
| saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2 | 2,200 | 186 | – |
| allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered | 1,100 | 286 | – |
| allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered | 900 | 586 | – |
| saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2 | 700 | 222 | – |
| allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered | 600 | 729 | – |
| saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2 | 600 | 530 | – |
| allenai/aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered | 600 | 564 | – |
| saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2 | 400 | 684 | – |
| allenai/wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered | 400 | 311 | – |
| allenai/nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls | 400 | 861 | – |
| allenai/wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered | 300 | 277 | – |
| allenai/wildchat-r1-p2-repetition-filter | 200 | 678 | – |
| allenai/coconot-r1-format-domain-filtered-keyword-filtered-filter-datecutoff-chinese-filtered | 200 | 432 | – |

### Dolci-Think-SFT-32B

| source | sampled rows | median tokens | exact share (Instruct only) |
|---|---|---|---|
| Dolci Think OpenThoughts3 Math | 4,074 | 539 | – |
| Dolci Think Python Algorithms | 2,516 | 191 | – |
| Persona Precise IF | 1,173 | 280 | – |
| Dolci Think Precise IF | 775 | 236 | – |
| Nemotron Code | 609 | 880 | – |
| SYNTHETIC-2-SFT-Verified | 590 | 708 | – |
| Dolci Think OpenThoughts3 STEM | 536 | 684 | – |
| Aya | 535 | 608 | 4.6% |
| Dolci Think OpenThoughts3 Code | 508 | 579 | – |
| WildJailbreak | 215 | 316 | 2.3% |
| Wildchat | 200 | 726 | 14.1% |
| WildGuardMix | 178 | 314 | 2.3% |
| CoCoNot | 40 | 371 | 0.5% |
| OpenAssistant | 32 | 586 | 0.3% |
| TableGPT | 19 | 83 | 0.2% |

### Dolci-Instruct-SFT

| source | sampled rows | median tokens | exact share (Instruct only) |
|---|---|---|---|
| Wildchat | 2,400 | 510 | 14.1% |
| Dolci Instruct Tool Use | 2,100 | 65 | 10.6% |
| Verifiable Reasoning | 1,900 | 154 | 14.4% |
| Dolci Instruct Python Algorithms | 786 | 172 | 8.7% |
| Logic Puzzles | 710 | 6 | 7.4% |
| Tulu 3 Persona MATH | 647 | 928 | 7.0% |
| Dolci Instruct Precise IF | 541 | 240 | 6.4% |
| Evol CodeAlpaca | 496 | 350 | 5.0% |
| Dolci Instruct OpenThoughts3+ Science | 405 | 647 | 4.6% |
| Aya | 398 | 75 | 4.6% |
| FLAN | 391 | 20 | 4.2% |
| OpenMathInstruct 2 | 234 | 134 | 2.3% |
| Tulu 3 Persona GSM | 222 | 300 | 2.3% |
| WildJailbreak | 218 | 254 | 2.3% |
| WildGuardMix | 209 | 80 | 2.3% |
| Tulu 3 Persona Python | 156 | 86 | 1.6% |
| Tulu 3 Persona Algebra | 76 | 790 | 0.9% |
| CoCoNot | 49 | 135 | 0.5% |
| OpenAssistant | 30 | 232 | 0.3% |
| TableGPT | 18 | 60 | 0.2% |
| SciRiff | 14 | 71 | 0.2% |

Reference, DPO answer tokens (results/tables.md): Think-DPO-7B chosen 527 / rejected 254; Think-DPO-32B 560 / 258; Instruct-DPO 333 / 256.
