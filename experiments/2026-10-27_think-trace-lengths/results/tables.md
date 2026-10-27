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
