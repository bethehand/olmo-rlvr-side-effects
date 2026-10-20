**Answer length, chosen vs rejected** (answer = text after `</think>`; per-pair ratio = chosen/rejected; median [IQR]).

| data | pairs | chosen median chars | rejected median chars | per-pair ratio | chosen longer | median tokens chosen / rejected |
|---|---|---|---|---|---|---|
| Dolci-Instruct-DPO (3.1 32B Instruct, 7B Instruct) | 259,922 | 1,296 | 1,002 | 1.284 [0.876, 2.058] | 66.1% | 333 / 256 |
| &nbsp;&nbsp;delta_learning | 124,942 | 1,063 | 740 | 1.392 [0.985, 2.069] | 71.8% | 277 / 194 |
| &nbsp;&nbsp;llm_judged | 124,980 | 1,664 | 1,490 | 1.18 [0.785, 2.113] | 60.5% | 434 / 373 |
| &nbsp;&nbsp;multiturn_self_talk | 5,000 | 632 | 527 | 1.18 [0.984, 1.49] | 70.9% | 120 / 97 |
| &nbsp;&nbsp;multiturn_synthetic_context | 5,000 | 376 | 305 | 1.086 [0.863, 1.5] | 58.2% | 88 / 73 |
| Dolci-Think-DPO-32B (Olmo 3 / 3.1 32B Think) | 200,000 | 2,053 | 1,011 | 1.764 [1.204, 2.813] | 84.4% | 560 / 258 |
| Dolci-Think-DPO-7B (Olmo 3 7B Think) | 150,000 | 2,012 | 1,003 | 1.766 [1.207, 2.804] | 84.4% | 527 / 254 |

**Thinking length (Think sets)**: median chars of the `<think>` block, chosen vs rejected; share of pairs where the whole message (thinking + answer) is longer for chosen.

| data | chosen think median | rejected think median | whole-message ratio median | chosen whole message longer |
|---|---|---|---|---|
| Dolci-Instruct-DPO (3.1 32B Instruct, 7B Instruct) | 0 | 0 | 1.284 | 66.1% |
| Dolci-Think-DPO-32B (Olmo 3 / 3.1 32B Think) | 3,651 | 2,237 | 1.572 | 77.5% |
| Dolci-Think-DPO-7B (Olmo 3 7B Think) | 3,472 | 2,205 | 1.556 | 77.6% |

**Formatting marks in the answer** (mean count per answer, chosen → rejected; share of pairs where chosen has more / fewer).

| data | bullets | bold | headers | table_rows | urls | olmo_ai2 | emoji |
|---|---|---|---|---|---|---|---|
| Dolci-Instruct-DPO (3.1 32B Instruct, 7B Instruct) | 7.065 vs 5.454 (37.9% / 18.4%) | 10.954 vs 8.221 (44.4% / 18.4%) | 3.023 vs 1.986 (32.5% / 12.1%) | 0.817 vs 0.231 (7.4% / 0.9%) | 0.121 vs 0.068 (2.4% / 0.7%) | 0.0 vs 0.0 (0.0% / 0.0%) | 0.935 vs 0.4 (17.3% / 3.4%) |
| Dolci-Think-DPO-32B (Olmo 3 / 3.1 32B Think) | 9.78 vs 5.151 (49.9% / 14.9%) | 14.053 vs 7.225 (62.1% / 14.2%) | 3.25 vs 1.67 (40.2% / 8.5%) | 0.353 vs 0.149 (3.8% / 0.4%) | 0.136 vs 0.036 (3.0% / 0.5%) | 0.0 vs 0.0 (0.0% / 0.0%) | 0.854 vs 0.234 (16.1% / 1.7%) |
| Dolci-Think-DPO-7B (Olmo 3 7B Think) | 9.257 vs 4.813 (47.6% / 14.1%) | 13.158 vs 6.651 (59.2% / 13.4%) | 3.067 vs 1.501 (38.4% / 7.9%) | 0.33 vs 0.122 (3.5% / 0.3%) | 0.132 vs 0.054 (2.9% / 0.5%) | 0.0 vs 0.0 (0.0% / 0.0%) | 0.878 vs 0.278 (15.5% / 1.7%) |

**Answers containing a URL / an Olmo or Ai2 mention** (% of answers, chosen vs rejected).

| data | URL chosen | URL rejected | Olmo/Ai2 chosen | Olmo/Ai2 rejected |
|---|---|---|---|---|
| Dolci-Instruct-DPO (3.1 32B Instruct, 7B Instruct) | 2.8% | 1.4% | 0.0% | 0.0% |
| Dolci-Think-DPO-32B (Olmo 3 / 3.1 32B Think) | 3.6% | 1.4% | 0.0% | 0.0% |
| Dolci-Think-DPO-7B (Olmo 3 7B Think) | 3.5% | 1.4% | 0.0% | 0.0% |

**By source (Think sets) / model pair** — top entries.


*Dolci-Instruct-DPO (3.1 32B Instruct, 7B Instruct)*, by source:

| source | pairs | per-pair ratio | chosen longer |
|---|---|---|---|
| None | 259,922 | 1.284 | 66.1% |

model pairs:

| chosen model | rejected model | pairs | per-pair ratio | chosen longer |
|---|---|---|---|---|
| qwen3-no_reasoning-32b | qwen3-no_reasoning-0.6b | 130,986 | 1.383 | 71.7% |
| gpt-4o | gpt-3.5 | 5,000 | 1.109 | 62.7% |
| gpt-120b | olmo2-1b | 1,510 | 1.798 | 73.9% |
| gpt-4.1-2025-04-14 | olmo2-1b | 1,476 | 1.067 | 54.3% |
| gpt-20b | olmo2-1b | 1,356 | 1.157 | 59.3% |
| gpt-120b | qwen3-no_reasoning-0.6b | 1,347 | 2.264 | 81.3% |
| qwen3-no_reasoning-30b-3a | olmo2-1b | 1,346 | 1.574 | 70.5% |
| gpt-4.1-2025-04-14 | qwen3-no_reasoning-0.6b | 1,283 | 1.434 | 75.6% |

*Dolci-Think-DPO-32B (Olmo 3 / 3.1 32B Think)*, by source:

| source | pairs | per-pair ratio | chosen longer |
|---|---|---|---|
| ultrafeedback_cleaned_olmo2_7b | 32,778 | 1.976 | 86.2% |
| flan_v2_converted | 19,660 | 1.421 | 69.1% |
| OpenThoughts3-full-filtered-science-no-cot | 19,023 | 1.813 | 93.0% |
| filtered_wc_sample_500k | 18,106 | 2.105 | 88.4% |
| valpy_if_qwq_reasoning_verified_no_reasoning | 15,836 | 1.864 | 80.4% |
| Wildchat-1M-gpt-4.1-regenerated-english | 15,803 | 2.15 | 93.3% |
| correct-python-sft-187k | 13,236 | 1.588 | 89.7% |
| tulu-3-sft-personas-math | 10,657 | 1.213 | 78.6% |
| evol_codealpaca_heval_decontaminated | 7,634 | 1.89 | 95.9% |
| Wildchat-1m-gpt-4.1-regeneration-not-english | 6,792 | 2.176 | 91.0% |
| tulu_v3.9_wildjailbreak_decontaminated_50k | 5,616 | 2.129 | 84.3% |
| tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 5,338 | 1.847 | 83.2% |

model pairs:

| chosen model | rejected model | pairs | per-pair ratio | chosen longer |
|---|---|---|---|---|
| qwen3-reasoning-32b | qwen3-reasoning-0.6b | 200,000 | 1.764 | 84.4% |

*Dolci-Think-DPO-7B (Olmo 3 7B Think)*, by source:

| source | pairs | per-pair ratio | chosen longer |
|---|---|---|---|
| ultrafeedback_cleaned_olmo2_7b | 23,202 | 1.984 | 85.7% |
| filtered_wc_sample_500k | 17,596 | 1.956 | 86.7% |
| OpenThoughts3-full-filtered-science-no-cot | 14,967 | 1.814 | 93.1% |
| flan_v2_converted | 14,057 | 1.45 | 69.4% |
| Wildchat-1M-gpt-4.1-regenerated-english | 13,955 | 2.06 | 92.6% |
| valpy_if_qwq_reasoning_verified_no_reasoning | 11,101 | 1.847 | 80.2% |
| correct-python-sft-187k | 9,005 | 1.584 | 89.4% |
| tulu-3-sft-personas-math | 7,252 | 1.214 | 79.0% |
| Wildchat-1m-gpt-4.1-regeneration-not-english | 5,220 | 2.117 | 89.6% |
| evol_codealpaca_heval_decontaminated | 5,171 | 1.891 | 95.9% |
| tulu_v3.9_wildjailbreak_decontaminated_50k | 3,885 | 2.15 | 84.0% |
| tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 3,884 | 1.842 | 83.1% |

model pairs:

| chosen model | rejected model | pairs | per-pair ratio | chosen longer |
|---|---|---|---|---|
| qwen3-reasoning-32b | qwen3-reasoning-0.6b | 150,000 | 1.766 | 84.4% |
