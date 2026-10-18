"""Answer length per post-training stage (for open-instruct #2002), from generate_length.py outputs.
Reports, per stage: the share of outputs that ran to the 32K cap (runaway, Wilson 95% CI); for outputs that finished,
answer length in characters and tokens and thinking length in tokens (median and mean, bootstrap 95% CI); and
OLMES-style mean answer length over all outputs (no </think> -> the whole output counts as the answer).
Between stages, paired over prompts that finished in both: median and mean per-prompt change, bootstrap 95% CI.
Also runs the checks from the pre-posting checklist (counts, indices, cap, answer extraction).
Usage: python analyze_length.py outputs/A.jsonl outputs/B.jsonl ...  (in stage order)"""
import json, math, random, re, statistics as st, sys

def load(p):
    return [json.loads(l) for l in open(p)]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

def boot(xs, f, B=2000, seed=0):
    rnd = random.Random(seed); n = len(xs)
    vals = sorted(f([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]

def answer_of(text):
    # Same result as OLMES's r1_style re.sub("(?ms).*</think>", "", text) (greedy: cut through the last </think>), but
    # linear: that regex is quadratic on long outputs that never close </think>.
    i = text.rfind("</think>")
    a = text[i + len("</think>"):] if i >= 0 else text
    a = re.sub(r"(?ms)^\s*<answer>\s*", "", a); return re.sub(r"(?ms)</answer>\s*$", "", a)

def main():
    paths = sys.argv[1:]
    stages = []
    for p in paths:
        rows = load(p); name = rows[0]["model"].split("/")[-1]
        by = {r["index"]: r for r in rows}
        # checks
        problems = []
        if len(rows) != 805 or len(by) != 805 or set(by) != set(range(805)): problems.append(f"count/indices: {len(rows)} rows, {len(by)} unique")
        caps = {r.get("max_new_tokens") for r in rows}
        if caps != {32768}: problems.append(f"max_new_tokens values {caps}")
        bad_extract = sum(len(answer_of(r["response"])) != r["answer_chars"] for r in rows)
        if bad_extract: problems.append(f"{bad_extract} answers don't match a re-extraction")
        stop_no_think = sum(r["finish_reason"] == "stop" and not r["has_think_end"] for r in rows)
        stages.append((name, by, problems, stop_no_think))
    print(f"{'stage':24} {'runaway (32K cap)':>22} {'answer chars, median [95% CI]':>32} {'mean':>7} {'answer tok med':>15} {'thinking tok med / mean':>24} {'OLMES-style mean chars':>23}")
    for name, by, problems, stop_no_think in stages:
        rows = list(by.values()); n = len(rows)
        run = sum(r["finish_reason"] != "stop" for r in rows); lo, hi = wilson(run, n)
        fin = [r for r in rows if r["finish_reason"] == "stop" and r["has_think_end"]]
        ac = [r["answer_chars"] for r in fin]; at = [r["answer_tokens"] for r in fin]; th = [r["n_tokens"] - r["answer_tokens"] for r in fin]
        m_lo, m_hi = boot(ac, st.median)
        print(f"{name:24} {run:4d} = {100*run/n:4.1f}% [{100*lo:.1f}, {100*hi:.1f}] {st.median(ac):9.0f} [{m_lo:.0f}, {m_hi:.0f}]{'':>9} {st.mean(ac):7.0f} {st.median(at):15.0f} {st.median(th):12.0f} / {st.mean(th):7.0f} {st.mean(r['answer_chars'] for r in rows):23.0f}")
        if problems or stop_no_think:
            print(f"   checks: {problems or 'ok'}; finished without </think>: {stop_no_think}")
    print("\npaired over prompts that finished in both stages: per-prompt change, median [95% CI] | mean [95% CI]")
    for (a, A, _, _), (b, Bm, _, _) in zip(stages, stages[1:]):
        common = [i for i in A if A[i]["finish_reason"] == "stop" and Bm[i]["finish_reason"] == "stop" and A[i]["has_think_end"] and Bm[i]["has_think_end"]]
        for label, key in [("answer chars", lambda r: r["answer_chars"]), ("thinking tokens", lambda r: r["n_tokens"] - r["answer_tokens"])]:
            d = [key(Bm[i]) - key(A[i]) for i in common]
            print(f"  {a} -> {b}, {label:15}: n={len(common)}  median {st.median(d):+7.0f} {tuple(round(x) for x in boot(d, st.median))} | mean {st.mean(d):+7.0f} {tuple(round(x) for x in boot(d, st.mean))}")

if __name__ == "__main__":
    main()
