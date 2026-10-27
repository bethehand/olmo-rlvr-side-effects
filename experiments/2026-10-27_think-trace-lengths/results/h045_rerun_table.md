<!-- Produced by rerun_table.py on 2026-10-27 14:25 CST -->
| model | samples | no `</think>` at 8,192 | of which loops (tail ratio < 0.5) | of which long, still progressing | on the 16 turns H045 left unfinished | on the 6 turns H045 answered | turns with ≥1 failing sample |
|---|---|---|---|---|---|---|---|
| Olmo 3 7B Think-SFT | 176 | 17.0% [12.2, 23.3] | 4.0% [1.9, 8.0] | 13.1% [8.9, 18.8] | 21.1% [14.9, 29.0] | 6.2% [2.1, 16.8] | 9/22 |
| Olmo 3 7B Think-DPO | 176 | 14.2% [9.8, 20.1] | 0.0% [0.0, 2.1] | 14.2% [9.8, 20.1] | 19.5% [13.6, 27.2] | 0.0% [0.0, 7.4] | 5/22 |
| Olmo 3 32B Think-SFT | 176 | 5.7% [3.1, 10.1] | 2.8% [1.2, 6.5] | 2.8% [1.2, 6.5] | 7.0% [3.7, 12.8] | 2.1% [0.4, 10.9] | 4/22 |
| Olmo 3.1 32B Think | 176 | 5.7% [3.1, 10.1] | 0.0% [0.0, 2.1] | 5.7% [3.1, 10.1] | 5.5% [2.7, 10.9] | 6.2% [2.1, 16.8] | 3/22 |
| Olmo 3.5 small SFT H045 (issue author's run, 1 sample per turn) | 22 | 16/22 = 72.7% | 13/22 = 59.1% | 3/22 = 13.6% | 16/16 | 0/6 | 16/22 |
