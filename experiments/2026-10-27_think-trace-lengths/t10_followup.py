"""Follow-up numbers for the temperature-1.0 reruns (queue_rerun_t10.sh) that the summary scripts do not print: exact tests for the
change in loop and no-answer counts between the temperature-0.6 and temperature-1.0 runs of the same model, seeds and budget; what
happened at 1.0 to the (turn, sample) pairs that looped at 0.6; the longest sample of each 32K run. Row-level recount, independent of
rerun_table.py / summarize_32k.py (same tail measure: distinct 8-gram ratio over the last 2,000 words, loop = ratio < 0.5).
Usage: t10_followup.py OUT.txt OUTPUT_DIR   (expects 7b_think_{sft,dpo}{,_32k}.jsonl and the matching *_t10*.jsonl files)"""
import json, sys, time
from math import comb


def tail_ratio(text, n=2000, k=8):
    w = text.split()[-n:]; g = [tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))]
    return len(set(g)) / len(g) if g else float("nan")


def fisher_one_sided(a, n1, b, n2):
    """P(X >= a) under the hypergeometric null: a of n1 in group 1 and b of n2 in group 2 share the same rate."""
    N, K = n1 + n2, a + b
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / comb(N, n1)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def is_loop(r):
    return (not r["has_think_end"]) and tail_ratio(r["text"]) < 0.5


def main():
    out, d = sys.argv[1], sys.argv[2]
    lines = [f"# Produced by t10_followup.py on {time.strftime('%Y-%m-%d %H:%M %Z')}"]
    for model, a, b in [("Olmo 3 7B Think-SFT, 8K", "7b_think_sft", "7b_think_sft_t10"), ("Olmo 3 7B Think-DPO, 8K", "7b_think_dpo", "7b_think_dpo_t10"),
                        ("Olmo 3 7B Think-SFT, 32K", "7b_think_sft_32k", "7b_think_sft_t10_32k"), ("Olmo 3 7B Think-DPO, 32K", "7b_think_dpo_32k", "7b_think_dpo_t10_32k")]:
        A, B = load(f"{d}/{a}.jsonl"), load(f"{d}/{b}.jsonl")
        key = lambda r: (r["case"], r["turn"], r["sample"])
        assert len(A) == len(B) == 176 and all(key(x) == key(y) and x["seed"] == y["seed"] for x, y in zip(A, B))
        ta, tb = {r["temperature"] for r in A}, {r["temperature"] for r in B}
        na, nb = sum(not r["has_think_end"] for r in A), sum(not r["has_think_end"] for r in B)
        la, lb = sum(is_loop(r) for r in A), sum(is_loop(r) for r in B)
        lines.append(f"\n== {model}: temperature {sorted(ta)} -> {sorted(tb)}, 176 samples each, same seeds")
        lines.append(f"no </think>: {na}/176 -> {nb}/176 | Fisher exact one-sided p (fewer at 1.0) = {fisher_one_sided(na, 176, nb, 176):.4f}")
        lines.append(f"loops (tail ratio < 0.5): {la}/176 -> {lb}/176 | Fisher exact one-sided p (fewer at 1.0) = {fisher_one_sided(la, 176, lb, 176):.4f}")
        loop_keys = {key(r) for r in A if is_loop(r)}
        if loop_keys:
            same = [r for r in B if key(r) in loop_keys]
            lines.append(f"the {len(loop_keys)} (turn, sample) pairs that looped at 0.6, at 1.0: closed {sum(r['has_think_end'] for r in same)}, still open {sum(not r['has_think_end'] for r in same)}, loops {sum(is_loop(r) for r in same)}")
        lines.append(f"longest sample (sampled tokens): {max(r['n_tokens'] for r in A)} at 0.6 -> {max(r['n_tokens'] for r in B)} at 1.0 | hit max_tokens: {sum(r['finish_reason'] != 'stop' for r in A)} -> {sum(r['finish_reason'] != 'stop' for r in B)}")
    txt = "\n".join(lines) + "\n"; open(out, "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
