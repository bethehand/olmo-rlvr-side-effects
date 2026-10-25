# Arm B' (seed 1, ratio <= 1.5): 19,215 pairs matched on source x chosen-length decile x prompt-length tercile, 469 topped up within source x decile, 316 within decile

| arm | pairs | median answer-token ratio | chosen longer | chosen / rejected median tokens | prompt chars median / mean | thinking ratio | chosen / rejected thinking chars | math sources | shared with A |
|---|---|---|---|---|---|---|---|---|---|
| A | 20,000 | 1.80 | 84% | 523 / 253 | 482 / 1024 | 1.44 | 3463 / 2195 | 8.4% | 20,000 |
| B | 20,000 | 1.13 | 65% | 523 / 473 | 710 / 1248 | 1.19 | 4358 / 3200 | 19.7% | 2,610 |
| B' | 20,000 | 1.15 | 67% | 522 / 467 | 535 / 1142 | 1.26 | 3695 / 2541 | 9.0% | 2,688 |

Source mix (share of pairs), A / B / B':

| source | A | B | B' |
|---|---|---|---|
| ultrafeedback_cleaned_olmo2_7b | 15.3% | 11.1% | 15.3% |
| filtered_wc_sample_500k | 11.5% | 9.8% | 11.6% |
| OpenThoughts3-full-filtered-science-no-cot | 10.1% | 10.4% | 10.4% |
| Wildchat-1M-gpt-4.1-regenerated-english | 9.4% | 7.3% | 9.4% |
| flan_v2_converted | 9.4% | 8.7% | 9.4% |
| valpy_if_qwq_reasoning_verified_no_reasoning | 7.4% | 6.9% | 7.5% |
| correct-python-sft-187k | 5.8% | 7.2% | 5.9% |
| tulu-3-sft-personas-math | 4.9% | 13.3% | 5.5% |
| Wildchat-1m-gpt-4.1-regeneration-not-english | 3.6% | 2.6% | 3.6% |
| evol_codealpaca_heval_decontaminated | 3.4% | 2.8% | 3.5% |
| tulu_v3.9_wildjailbreak_decontaminated_50k | 2.7% | 1.7% | 1.9% |
| tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 2.6% | 2.3% | 2.6% |
| tulu_v3.9_aya_100k | 2.0% | 1.1% | 1.9% |
| tulu_v3.9_open_math_2_gsm8k_50k | 1.8% | 3.2% | 1.8% |
| IF_sft_data_verified_permissive | 1.7% | 1.4% | 1.8% |
