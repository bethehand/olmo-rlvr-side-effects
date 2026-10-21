# Training arms (seed 1, 20000 pairs each; C had 0 cells topped up within the decile)

| arm | pairs | median answer-token ratio | chosen longer | over the 100-token cap | chosen / rejected median tokens | chosen mean | thinking ratio | sources |
|---|---|---|---|---|---|---|---|---|
| pool (150K) | 150,000 | 1.793 | 83.9% | 61.3% | 527 / 254 | 662 | 1.44 | 24 |
| A original | 20,000 | 1.798 | 83.9% | 61.5% | 523 / 253 | 687 | 1.438 | 24 |
| B gap removed (ratio ≤ 1.5, chosen-length matched to A) | 20,000 | 1.13 | 65.0% | 34.5% | 523 / 473 | 610 | 1.186 | 24 |
| C control (B's source × length cells, no rule) | 20,000 | 1.529 | 79.7% | 55.6% | 523 / 289 | 633 | 1.33 | 24 |
| D Ai2 cap (chosen ≤ rejected + 100 tokens) | 20,000 | 1.083 | 58.5% | 0.0% | 107 / 90 | 248 | 1.215 | 24 |

Chosen-length deciles (pool edges in tokens): 0, 48, 113, 239, 397, 527, 663, 802, 961, 1196

| arm | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 2041 | 2011 | 2000 | 2008 | 2006 | 1973 | 1940 | 2000 | 2021 | 2000 |
| B | 2041 | 2011 | 2000 | 2008 | 2006 | 1973 | 1940 | 2000 | 2021 | 2000 |
| C | 2041 | 2011 | 2000 | 2008 | 2006 | 1973 | 1940 | 2000 | 2021 | 2000 |
| D | 5241 | 5070 | 3483 | 2020 | 1229 | 968 | 786 | 522 | 354 | 327 |

Source mix (share of pairs):

| source | pool | A | B | C | D |
|---|---|---|---|---|---|
| ultrafeedback_cleaned_olmo2_7b | 15.5% | 15.3% | 11.1% | 11.1% | 15.6% |
| filtered_wc_sample_500k | 11.7% | 11.5% | 9.8% | 9.8% | 10.3% |
| OpenThoughts3-full-filtered-science-no-cot | 10.0% | 10.1% | 10.4% | 10.4% | 3.7% |
| flan_v2_converted | 9.4% | 9.4% | 8.7% | 8.7% | 20.4% |
| Wildchat-1M-gpt-4.1-regenerated-english | 9.3% | 9.4% | 7.3% | 7.3% | 5.7% |
| valpy_if_qwq_reasoning_verified_no_reasoning | 7.4% | 7.4% | 6.9% | 6.9% | 10.0% |
| correct-python-sft-187k | 6.0% | 5.8% | 7.2% | 7.2% | 4.2% |
| tulu-3-sft-personas-math | 4.8% | 4.9% | 13.3% | 13.3% | 5.3% |
| Wildchat-1m-gpt-4.1-regeneration-not-english | 3.5% | 3.6% | 2.6% | 2.6% | 2.1% |
| evol_codealpaca_heval_decontaminated | 3.4% | 3.4% | 2.8% | 2.8% | 1.0% |
| tulu_v3.9_wildjailbreak_decontaminated_50k | 2.6% | 2.7% | 1.7% | 1.7% | 1.7% |
| tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 2.6% | 2.6% | 2.3% | 2.3% | 1.9% |
| tulu_v3.9_aya_100k | 1.9% | 2.0% | 1.1% | 1.1% | 0.9% |
| tulu_v3.9_open_math_2_gsm8k_50k | 1.7% | 1.8% | 3.2% | 3.2% | 3.5% |
| tulu-3-sft-personas-math-grade | 1.7% | 1.7% | 3.2% | 3.2% | 3.5% |
