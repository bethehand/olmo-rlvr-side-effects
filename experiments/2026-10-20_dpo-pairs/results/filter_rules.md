# Length rules on the Dolci DPO sets (answer tokens, Olmo 3 tokenizer)

Per-pair ratio = (chosen+1)/(rejected+1) answer tokens; p80 = 80th percentile of chosen − rejected tokens.
For the Think sets the rules act on the answer after `</think>`; the thinking ratio column shows what stays of the thinking-length gap.

## Dolci-Think-DPO-32B (200,000 pairs; unfiltered: ratio 1.81, chosen longer 84%, >100 tokens 62%, p80 514 tokens, thinking ratio 1.46)

| rule | pairs kept | median ratio after | chosen longer after | >100 tokens after | p80 after | thinking ratio after |
|---|---|---|---|---|---|---|
| Ai2 §5.3.1: chosen ≤ rejected + 100 tokens | 75,565 (38%) | 1.07 | 58% | 0% | 58 | 1.21 |
| chosen ≤ rejected + 300 tokens | 121,760 (61%) | 1.33 | 74% | 38% | 184 | 1.28 |
| chosen/rejected ≤ 1.25 (tokens) | 53,925 (27%) | 1.00 | 41% | 9% | 43 | 1.14 |
| chosen/rejected ≤ 1.5 (tokens) | 76,334 (38%) | 1.08 | 58% | 26% | 141 | 1.20 |
| chosen/rejected ≤ 2 (tokens) | 112,076 (56%) | 1.27 | 71% | 44% | 286 | 1.29 |

### by source (sources with ≥ 2% of the pairs; 24 sources in all)

| source | pairs | share | ratio | chosen longer | >100 tokens | share of all >100-token pairs |
|---|---|---|---|---|---|---|
| ultrafeedback_cleaned_olmo2_7b | 32,778 | 16% | 2.01 | 86% | 62% | 16% |
| flan_v2_converted | 19,660 | 10% | 1.40 | 68% | 16% | 2% |
| OpenThoughts3-full-filtered-science-no-cot | 19,023 | 10% | 1.92 | 93% | 86% | 13% |
| filtered_wc_sample_500k | 18,106 | 9% | 2.16 | 89% | 74% | 11% |
| valpy_if_qwq_reasoning_verified_no_reasoning | 15,836 | 8% | 1.90 | 80% | 49% | 6% |
| Wildchat-1M-gpt-4.1-regenerated-english | 15,803 | 8% | 2.22 | 94% | 82% | 10% |
| correct-python-sft-187k | 13,236 | 7% | 1.60 | 88% | 73% | 8% |
| tulu-3-sft-personas-math | 10,657 | 5% | 1.20 | 77% | 57% | 5% |
| evol_codealpaca_heval_decontaminated | 7,634 | 4% | 1.93 | 96% | 89% | 5% |
| Wildchat-1m-gpt-4.1-regeneration-not-english | 6,792 | 3% | 2.17 | 92% | 82% | 4% |
| tulu_v3.9_wildjailbreak_decontaminated_50k | 5,616 | 3% | 2.21 | 84% | 74% | 3% |
| tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 5,338 | 3% | 1.91 | 83% | 74% | 3% |
| tulu_v3.9_aya_100k | 4,078 | 2% | 3.71 | 90% | 80% | 3% |

Sources with median ratio ≥ 2 (≥ 500 pairs): ultrafeedback_cleaned_olmo2_7b (32,778, 2.01); filtered_wc_sample_500k (18,106, 2.16); Wildchat-1M-gpt-4.1-regenerated-english (15,803, 2.22); Wildchat-1m-gpt-4.1-regeneration-not-english (6,792, 2.17); tulu_v3.9_wildjailbreak_decontaminated_50k (5,616, 2.21); tulu_v3.9_aya_100k (4,078, 3.71); IF_sft_data_verified_permissive (3,529, 2.06); oasst1_converted (1,762, 2.67); tulu-3-sft-coconot-regenerated (927, 2.86)

## Dolci-Think-DPO-7B (150,000 pairs; unfiltered: ratio 1.79, chosen longer 84%, >100 tokens 61%, p80 499 tokens, thinking ratio 1.44)

| rule | pairs kept | median ratio after | chosen longer after | >100 tokens after | p80 after | thinking ratio after |
|---|---|---|---|---|---|---|
| Ai2 §5.3.1: chosen ≤ rejected + 100 tokens | 58,096 (39%) | 1.08 | 59% | 0% | 59 | 1.21 |
| chosen ≤ rejected + 300 tokens | 93,924 (63%) | 1.35 | 74% | 38% | 183 | 1.27 |
| chosen/rejected ≤ 1.25 (tokens) | 40,604 (27%) | 1.00 | 41% | 9% | 44 | 1.13 |
| chosen/rejected ≤ 1.5 (tokens) | 57,719 (38%) | 1.08 | 58% | 26% | 134 | 1.18 |
| chosen/rejected ≤ 2 (tokens) | 84,905 (57%) | 1.27 | 72% | 44% | 270 | 1.27 |

### by source (sources with ≥ 2% of the pairs; 24 sources in all)

| source | pairs | share | ratio | chosen longer | >100 tokens | share of all >100-token pairs |
|---|---|---|---|---|---|---|
| ultrafeedback_cleaned_olmo2_7b | 23,202 | 15% | 2.03 | 85% | 62% | 16% |
| filtered_wc_sample_500k | 17,596 | 12% | 1.93 | 86% | 66% | 13% |
| OpenThoughts3-full-filtered-science-no-cot | 14,967 | 10% | 1.92 | 93% | 86% | 14% |
| flan_v2_converted | 14,057 | 9% | 1.43 | 68% | 16% | 2% |
| Wildchat-1M-gpt-4.1-regenerated-english | 13,955 | 9% | 2.04 | 92% | 76% | 12% |
| valpy_if_qwq_reasoning_verified_no_reasoning | 11,101 | 7% | 1.88 | 80% | 48% | 6% |
| correct-python-sft-187k | 9,005 | 6% | 1.59 | 87% | 73% | 7% |
| tulu-3-sft-personas-math | 7,252 | 5% | 1.20 | 77% | 58% | 5% |
| Wildchat-1m-gpt-4.1-regeneration-not-english | 5,220 | 3% | 2.11 | 90% | 76% | 4% |
| evol_codealpaca_heval_decontaminated | 5,171 | 3% | 1.93 | 96% | 90% | 5% |
| tulu_v3.9_wildjailbreak_decontaminated_50k | 3,885 | 3% | 2.23 | 84% | 74% | 3% |
| tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 3,884 | 3% | 1.91 | 83% | 72% | 3% |

Sources with median ratio ≥ 2 (≥ 500 pairs): ultrafeedback_cleaned_olmo2_7b (23,202, 2.03); Wildchat-1M-gpt-4.1-regenerated-english (13,955, 2.04); Wildchat-1m-gpt-4.1-regeneration-not-english (5,220, 2.11); tulu_v3.9_wildjailbreak_decontaminated_50k (3,885, 2.23); tulu_v3.9_aya_100k (2,863, 3.67); IF_sft_data_verified_permissive (2,482, 2.00); oasst1_converted (1,259, 2.66); tulu-3-sft-coconot-regenerated (790, 2.93)

## Dolci-Instruct-DPO (259,922 pairs; unfiltered: ratio 1.29, chosen longer 66%, >100 tokens 36%, p80 327 tokens, thinking ratio 1.00)

| rule | pairs kept | median ratio after | chosen longer after | >100 tokens after | p80 after | thinking ratio after |
|---|---|---|---|---|---|---|
| Ai2 §5.3.1: chosen ≤ rejected + 100 tokens | 165,572 (64%) | 1.00 | 47% | 0% | 48 | 1.00 |
| chosen ≤ rejected + 300 tokens | 204,787 (79%) | 1.09 | 57% | 19% | 97 | 1.00 |
| chosen/rejected ≤ 1.25 (tokens) | 125,051 (48%) | 0.87 | 30% | 4% | 15 | 1.00 |
| chosen/rejected ≤ 1.5 (tokens) | 153,238 (59%) | 0.97 | 42% | 11% | 57 | 1.00 |
| chosen/rejected ≤ 2 (tokens) | 190,073 (73%) | 1.05 | 54% | 22% | 120 | 1.00 |

### by preference type (unfiltered, then after Ai2's 100-token rule)

| preference type | pairs | ratio | chosen longer | >100 tokens | p80 | kept by 100-token rule | ratio after |
|---|---|---|---|---|---|---|---|
| delta_learning | 124,942 | 1.38 | 71% | 38% | 317 | 62% | 1.04 |
| llm_judged | 124,980 | 1.20 | 62% | 38% | 380 | 62% | 0.91 |
| multiturn_self_talk | 5,000 | 1.19 | 70% | 0% | 61 | 100% | 1.19 |
| multiturn_synthetic_context | 5,000 | 1.09 | 58% | 0% | 51 | 100% | 1.09 |

### by source (sources with ≥ 2% of the pairs; 0 sources in all)

| source | pairs | share | ratio | chosen longer | >100 tokens | share of all >100-token pairs |
|---|---|---|---|---|---|---|
### Dolci-Instruct-DPO by chosen / rejected model (the release has no source field)

| chosen / rejected | pairs | share | ratio | chosen longer | >100 tokens | share of all >100-token pairs | preference types |
|---|---|---|---|---|---|---|---|
| qwen3-no_reasoning-32b / qwen3-no_reasoning-0.6b | 130,986 | 50% | 1.37 | 71% | 37% | 51% | delta_learning 124,942; multiturn 5,000; llm_judged 1,044 |
| gpt-4o / gpt-3.5 | 5,000 | 2% | 1.13 | 63% | 0% | 0% | multiturn only |
| all other pairs (GPT-judged half: gpt-120b, gpt-4.1, gpt-20b … vs olmo2-1b, qwen3-0.6b, …) | 123,936 | 48% | ≈1.20 | 62% | 38% | 49% | llm_judged |

The two multi-turn subsets (10,000 pairs) are the only part where no pair exceeds the 100-token cap.
