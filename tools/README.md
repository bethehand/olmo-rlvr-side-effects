# tools

Small, standalone scripts that came out of the experiments and are meant to be reused on other checkpoints.

- `verbosity_report.py`: how long and how padded a model's answers are, and how that compares with a reference run on the same prompts (answer and thinking length, list items / bold / headings / tables / URLs / emoji per answer, over-answer rate on short prompts, per-prompt ratios against a reference). Input is JSONL with a `response` and an `instruction` field (the format of `experiments/2026-10-18_length/generate_length.py`). Example: `python tools/verbosity_report.py --tokenizer allenai/Olmo-3-7B-Think --reference SFT.jsonl SFT.jsonl DPO.jsonl FINAL.jsonl --out-md report.md`. First use: [`experiments/2026-10-22_verbosity-report`](../experiments/2026-10-22_verbosity-report).
