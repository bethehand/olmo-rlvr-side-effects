"""Independent check of baseline_8k.py's "no answer at 8K" count: the 2026-10-23 post-hoc 8K truncations (truncate_to_cap.py) of the
Think-SFT and Think-DPO AlpacaEval runs already carry has_think_end for the first 8,192 tokens. Compares both routes row by row."""
import json
E = "/mnt/nvme0/looped/oi_work/experiments/2026-10-18_length/outputs/"
for base in ["Olmo-3-7B-Think-SFT@main_run1", "Olmo-3-7B-Think-DPO@main_run1"]:
    full = [json.loads(l) for l in open(E + base + ".jsonl")]
    cap = [json.loads(l) for l in open(E + base + "_cap8k.jsonl")]
    assert [r["index"] for r in full] == [r["index"] for r in cap], "row order differs"
    no_ans_cap8k = sum(1 for r in cap if not r["has_think_end"])
    over = sum(1 for r in full if r["n_tokens"] > 8192)
    print(base, "| cap8k file: rows without </think> =", no_ans_cap8k, "of", len(cap), "(%.1f%%)" % (100 * no_ans_cap8k / len(cap)),
          "| rows over 8192 tokens in the 32K file =", over, "| max n_tokens in cap8k file =", max(r["n_tokens"] for r in cap))
