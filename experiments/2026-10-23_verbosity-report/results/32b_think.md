| run | finished | runaway | answer median tokens [IQR] | mean | thinking median | lines | list items | bold | headings | tables | URLs | emoji | over-answer (short prompts) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-3-32B-Think-SFT@main_run1 | 790/805 | 15 | 598 [301, 824] | 595 | 864 | 32.4 | 16.6 | 19.1 | 4.2 | 0.37 | 3.2% | 25.7% | 301/473 (64%) |
| Olmo-3-32B-Think-SFT@main_rel1 | 793/805 | 12 | 580 [312, 826] | 592 | 679 | 32.2 | 16.2 | 18.6 | 3.9 | 0.38 | 9.1% | 41.4% | 293/477 (61%) |
| Olmo-3-32B-Think-SFT@main_rob1 | 798/805 | 7 | 560 [285, 797] | 558 | 808 | 30.6 | 15.5 | 17.9 | 3.8 | 0.29 | 3.3% | 31.5% | 279/476 (59%) |
| Olmo-3-32B-Think-DPO@main_run1 | 801/805 | 4 | 804 [470, 1,037] | 773 | 1,141 | 41.3 | 21.0 | 25.7 | 5.4 | 0.51 | 5.0% | 34.3% | 370/478 (77%) |
| Olmo-3-32B-Think-DPO@main_rel1 | 799/805 | 6 | 818 [486, 1,091] | 805 | 1,061 | 42.6 | 21.5 | 27.0 | 5.4 | 0.69 | 11.0% | 51.4% | 377/477 (79%) |
| Olmo-3-32B-Think@main_run1 | 803/805 | 2 | 710 [344, 957] | 679 | 1,444 | 36.2 | 18.6 | 22.7 | 4.4 | 0.45 | 3.9% | 29.0% | 334/479 (70%) |
| Olmo-3-32B-Think@main_rel1 | 802/805 | 3 | 702 [345, 1,031] | 704 | 1,384 | 36.5 | 18.5 | 22.7 | 4.4 | 0.46 | 10.7% | 43.8% | 321/479 (67%) |
| Olmo-3.1-32B-Think@main_run1 | 805/805 | 0 | 701 [340, 1,049] | 716 | 1,720 | 34.6 | 17.8 | 21.4 | 4.3 | 0.42 | 12.7% | 48.0% | 314/480 (65%) |
| Olmo-3.1-32B-Think@main_ctrl1 | 804/805 | 1 | 764 [402, 1,038] | 745 | 1,860 | 38.6 | 19.7 | 24.4 | 4.8 | 0.48 | 5.1% | 38.8% | 349/480 (73%) |

Versus `Olmo-3-32B-Think-SFT@main_run1.jsonl`, over prompts finished in both (per-prompt ratio of answer tokens; deltas are mean per answer):

| run | prompts | ratio median [IQR] | longer | median delta tokens | lines | list items | bold | headings | URLs | emoji |
|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-3-32B-Think-SFT@main_run1 | 790 | 1.00 [1.00, 1.00] | 0% | 0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +0.00 |
| Olmo-3-32B-Think-SFT@main_rel1 | 782 | 1.00 [0.83, 1.17] | 50% | 0 | -0.6 | -0.5 | -0.7 | -0.3 | +0.17 | +0.20 |
| Olmo-3-32B-Think-SFT@main_rob1 | 786 | 0.94 [0.78, 1.11] | 40% | -22 | -1.9 | -1.1 | -1.3 | -0.4 | -0.01 | +0.10 |
| Olmo-3-32B-Think-DPO@main_run1 | 790 | 1.32 [1.14, 1.60] | 89% | 170 | +8.7 | +4.3 | +6.5 | +1.2 | +0.10 | +0.25 |
| Olmo-3-32B-Think-DPO@main_rel1 | 787 | 1.39 [1.18, 1.67] | 90% | 194 | +10.1 | +4.7 | +7.9 | +1.2 | +0.27 | +0.60 |
| Olmo-3-32B-Think@main_run1 | 789 | 1.15 [0.98, 1.36] | 72% | 70 | +3.6 | +1.9 | +3.6 | +0.2 | +0.05 | +0.20 |
| Olmo-3-32B-Think@main_rel1 | 788 | 1.20 [0.98, 1.42] | 73% | 102 | +4.0 | +1.8 | +3.6 | +0.2 | +0.19 | +0.34 |
| Olmo-3.1-32B-Think@main_run1 | 790 | 1.24 [0.90, 1.53] | 69% | 114 | +2.0 | +1.1 | +2.3 | +0.1 | +0.22 | +0.37 |
| Olmo-3.1-32B-Think@main_ctrl1 | 790 | 1.28 [1.04, 1.54] | 78% | 142 | +6.2 | +3.1 | +5.2 | +0.7 | +0.08 | +0.18 |

Per dataset (median answer length over finished prompts):

| run | helpful_base | koala | oasst | selfinstruct | vicuna |
|---|---|---|---|---|---|
| Olmo-3-32B-Think-SFT@main_run1 | 687 | 702 | 606 | 350 | 779 |
| Olmo-3-32B-Think-SFT@main_rel1 | 674 | 672 | 616 | 355 | 786 |
| Olmo-3-32B-Think-SFT@main_rob1 | 655 | 613 | 598 | 319 | 758 |
| Olmo-3-32B-Think-DPO@main_run1 | 923 | 853 | 830 | 534 | 960 |
| Olmo-3-32B-Think-DPO@main_rel1 | 915 | 934 | 846 | 532 | 994 |
| Olmo-3-32B-Think@main_run1 | 820 | 790 | 732 | 384 | 887 |
| Olmo-3-32B-Think@main_rel1 | 851 | 816 | 776 | 393 | 958 |
| Olmo-3.1-32B-Think@main_run1 | 826 | 887 | 760 | 382 | 998 |
| Olmo-3.1-32B-Think@main_ctrl1 | 897 | 858 | 779 | 451 | 1,003 |
