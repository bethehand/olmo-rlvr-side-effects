| run | finished | runaway | answer median tokens [IQR] | mean | thinking median | lines | list items | bold | headings | tables | URLs | emoji | over-answer (short prompts) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-3.1-32B-Instruct-SFT@main_run1 | 802/805 | 3 | 242 [106, 460] | 321 | – | 18.2 | 7.2 | 7.5 | 1.8 | 0.77 | 4.6% | 1.2% | 97/480 (20%) |
| Olmo-3.1-32B-Instruct-SFT@main_rel1 | 801/805 | 4 | 205 [93, 387] | 283 | – | 13.6 | 6.6 | 3.9 | 0.6 | 0.30 | 3.0% | 1.7% | 53/477 (11%) |
| Olmo-3.1-32B-Instruct-DPO@main_run1 | 805/805 | 0 | 544 [252, 816] | 618 | – | 36.8 | 14.0 | 16.5 | 4.7 | 1.41 | 7.1% | 11.7% | 267/480 (56%) |
| Olmo-3.1-32B-Instruct-DPO@main_rel1 | 805/805 | 0 | 517 [221, 813] | 595 | – | 33.3 | 13.3 | 13.5 | 3.9 | 1.13 | 12.0% | 11.6% | 242/480 (50%) |
| Olmo-3.1-32B-Instruct-DPO@main_rob1 | 805/805 | 0 | 557 [250, 825] | 637 | – | 38.5 | 13.8 | 16.8 | 4.7 | 1.34 | 8.3% | 13.8% | 263/480 (55%) |
| Olmo-3.1-32B-Instruct@main_run1 | 805/805 | 0 | 493 [210, 792] | 607 | – | 34.4 | 13.2 | 10.9 | 3.3 | 1.13 | 8.4% | 6.5% | 228/480 (48%) |
| Olmo-3.1-32B-Instruct@main_ctrl1 | 804/805 | 1 | 544 [225, 833] | 634 | – | 38.8 | 13.6 | 13.2 | 4.3 | 1.53 | 6.1% | 7.1% | 249/480 (52%) |

Versus `Olmo-3.1-32B-Instruct-SFT@main_run1.jsonl`, over prompts finished in both (per-prompt ratio of answer tokens; deltas are mean per answer):

| run | prompts | ratio median [IQR] | longer | median delta tokens | lines | list items | bold | headings | URLs | emoji |
|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-3.1-32B-Instruct-SFT@main_run1 | 802 | 1.00 [1.00, 1.00] | 0% | 0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +0.00 |
| Olmo-3.1-32B-Instruct-SFT@main_rel1 | 798 | 0.90 [0.62, 1.23] | 39% | -16 | -4.5 | -0.5 | -3.5 | -1.2 | -0.08 | +0.01 |
| Olmo-3.1-32B-Instruct-DPO@main_run1 | 802 | 1.65 [1.25, 2.74] | 91% | 198 | +16.9 | +6.8 | +8.8 | +2.8 | +0.09 | +0.44 |
| Olmo-3.1-32B-Instruct-DPO@main_rel1 | 802 | 1.57 [1.19, 2.62] | 87% | 168 | +15.0 | +6.1 | +6.0 | +2.1 | +0.29 | +0.34 |
| Olmo-3.1-32B-Instruct-DPO@main_rob1 | 802 | 1.69 [1.24, 2.79] | 91% | 190 | +19.1 | +6.6 | +9.2 | +2.8 | +0.16 | +0.42 |
| Olmo-3.1-32B-Instruct@main_run1 | 802 | 1.52 [1.14, 2.41] | 84% | 146 | +14.4 | +5.8 | +3.5 | +1.4 | +0.12 | +0.21 |
| Olmo-3.1-32B-Instruct@main_ctrl1 | 801 | 1.54 [1.17, 2.61] | 89% | 169 | +17.4 | +6.3 | +5.8 | +2.4 | +0.12 | +0.19 |

Per dataset (median answer length over finished prompts):

| run | helpful_base | koala | oasst | selfinstruct | vicuna |
|---|---|---|---|---|---|
| Olmo-3.1-32B-Instruct-SFT@main_run1 | 121 | 470 | 331 | 152 | 338 |
| Olmo-3.1-32B-Instruct-SFT@main_rel1 | 242 | 348 | 211 | 135 | 294 |
| Olmo-3.1-32B-Instruct-DPO@main_run1 | 647 | 634 | 600 | 290 | 704 |
| Olmo-3.1-32B-Instruct-DPO@main_rel1 | 573 | 614 | 602 | 281 | 713 |
| Olmo-3.1-32B-Instruct-DPO@main_rob1 | 618 | 672 | 621 | 311 | 710 |
| Olmo-3.1-32B-Instruct@main_run1 | 558 | 623 | 552 | 240 | 702 |
| Olmo-3.1-32B-Instruct@main_ctrl1 | 628 | 688 | 618 | 247 | 714 |
