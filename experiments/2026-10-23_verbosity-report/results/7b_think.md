| run | finished | runaway | answer median tokens [IQR] | mean | thinking median | lines | list items | bold | headings | tables | URLs | emoji | over-answer (short prompts) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-3-7B-Think-SFT@main_run1 | 781/805 | 24 | 589 [309, 817] | 578 | 938 | 31.5 | 16.2 | 19.0 | 4.1 | 0.25 | 2.2% | 26.6% | 295/468 (63%) |
| Olmo-3-7B-Think-SFT@main_ctrl1 | 789/805 | 16 | 567 [277, 793] | 570 | 837 | 30.8 | 15.6 | 18.3 | 3.8 | 0.42 | 2.9% | 28.3% | 287/472 (61%) |
| Olmo-3-7B-Think-DPO@main_run1 | 795/805 | 8 | 874 [546, 1,107] | 847 | 1,160 | 38.6 | 20.5 | 25.8 | 4.6 | 0.52 | 4.4% | 44.8% | 395/473 (84%) |
| Olmo-3-7B-Think-DPO@main_run2 | 798/805 | 7 | 872 [547, 1,095] | 836 | 1,130 | 38.2 | 20.3 | 25.2 | 4.6 | 0.47 | 3.4% | 45.1% | 386/476 (81%) |
| Olmo-3-7B-Think@main_run1 | 798/805 | 7 | 656 [325, 918] | 656 | 2,218 | 33.2 | 17.4 | 19.3 | 4.0 | 0.41 | 2.6% | 28.9% | 319/477 (67%) |

Versus `Olmo-3-7B-Think-SFT@main_run1.jsonl`, over prompts finished in both (per-prompt ratio of answer tokens; deltas are mean per answer):

| run | prompts | ratio median [IQR] | longer | median delta tokens | lines | list items | bold | headings | URLs | emoji |
|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-3-7B-Think-SFT@main_run1 | 781 | 1.00 [1.00, 1.00] | 0% | 0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +0.00 |
| Olmo-3-7B-Think-SFT@main_ctrl1 | 772 | 0.99 [0.84, 1.15] | 48% | -2 | -0.3 | -0.4 | -0.5 | -0.2 | +0.03 | -0.07 |
| Olmo-3-7B-Think-DPO@main_run1 | 773 | 1.50 [1.26, 1.86] | 95% | 250 | +7.3 | +4.5 | +6.9 | +0.6 | +0.07 | +0.46 |
| Olmo-3-7B-Think-DPO@main_run2 | 776 | 1.48 [1.25, 1.81] | 95% | 238 | +7.0 | +4.2 | +6.3 | +0.6 | +0.02 | +0.52 |
| Olmo-3-7B-Think@main_run1 | 775 | 1.13 [0.95, 1.38] | 68% | 64 | +2.2 | +1.4 | +0.6 | +0.0 | +0.03 | -0.03 |

Per dataset (median answer length over finished prompts):

| run | helpful_base | koala | oasst | selfinstruct | vicuna |
|---|---|---|---|---|---|
| Olmo-3-7B-Think-SFT@main_run1 | 670 | 639 | 632 | 346 | 815 |
| Olmo-3-7B-Think-SFT@main_ctrl1 | 676 | 656 | 570 | 331 | 760 |
| Olmo-3-7B-Think-DPO@main_run1 | 972 | 966 | 900 | 594 | 1,020 |
| Olmo-3-7B-Think-DPO@main_run2 | 972 | 972 | 891 | 619 | 1,072 |
| Olmo-3-7B-Think@main_run1 | 759 | 720 | 687 | 350 | 903 |
