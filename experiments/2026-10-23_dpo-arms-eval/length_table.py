"""Length table for the 805 AlpacaEval prompts: one row per model, then paired changes against the training-template Think-SFT,
paired arm-vs-arm differences with sign tests, and the robustness checks (characters, trimmed mean, per source dataset).
Answer = text after the last </think> (field answer_tokens); bootstrap 95% intervals (2000 resamples, seed 0).
Usage: length_table.py OUT_DIR [RUN]   OUT_DIR holds <name>.jsonl from ../2026-10-18_length/run_length.sh for the models below;
RUN (default run1) selects the generation seed: run2 files are <model>_run2.jsonl and the control is Olmo-3-7B-Think-SFT@main_tmplA2.
With a second directory as OUT_DIR2 (--compare DIR2), prints per-model paired run-to-run differences as a generation-noise check."""
import json, math, os, random, statistics, sys

RUN = next((a for a in sys.argv[2:] if not a.startswith("--")), "run1")
CMP = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--compare"), None)
sfx = "" if RUN == "run1" else RUN[3:]
FILES = [("Think-SFT (training tmpl)", f"Olmo-3-7B-Think-SFT@main_tmplA{sfx}"),
         ("Think-SFT (released tmpl)", f"Olmo-3-7B-Think-SFT@main_{RUN}"),
         ("Think-DPO", f"Olmo-3-7B-Think-DPO@main_{RUN}"),
         ("arm A", f"icedduck_olmo3-7b-think-dpo-arm_A_seed1@main_{RUN}"),
         ("arm B", f"icedduck_olmo3-7b-think-dpo-arm_B_seed1@main_{RUN}"),
         ("arm B'", f"icedduck_olmo3-7b-think-dpo-arm_Bp_seed1@main_{RUN}")]


def load(path):
    return {json.loads(l)["index"]: json.loads(l) for l in open(path) if l.strip()}


def boot(vals, stat=statistics.median, n=2000):
    random.seed(0)
    bs = sorted(stat([random.choice(vals) for _ in vals]) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def trimmed_mean(v, frac=0.1):
    v = sorted(v); k = int(len(v) * frac)
    return statistics.mean(v[k: len(v) - k])


def sign_test(d):
    neg = sum(x < 0 for x in d); pos = sum(x > 0 for x in d); n = neg + pos
    z = (neg - n / 2) / math.sqrt(n / 4) if n else float("nan")
    return neg, pos, len(d) - n, z


def main():
    D = sys.argv[1]
    data = {k: load(os.path.join(D, v + ".jsonl")) for k, v in FILES if os.path.exists(os.path.join(D, v + ".jsonl"))}
    print(f"[{RUN}] median answer tokens [95% CI] | median total tokens | median thinking tokens (finished) | cap hits | answers > 1000 tokens | median answer chars")
    for k, rows in data.items():
        a = [r["answer_tokens"] for r in rows.values()]; t = [r["n_tokens"] for r in rows.values()]
        th = [r["n_tokens"] - r["answer_tokens"] for r in rows.values() if r["finish_reason"] == "stop"]
        c = [r["answer_chars"] for r in rows.values()]
        lo, hi = boot(a)
        print(f"  {k:26s} n={len(rows)} answer {statistics.median(a):6.0f} [{lo:.0f}, {hi:.0f}] | total {statistics.median(t):6.0f} | "
              f"thinking {statistics.median(th):6.0f} | cap {sum(r['finish_reason'] == 'length' for r in rows.values()):3d} | "
              f">1000 {sum(x > 1000 for x in a) / len(a):.2f} | chars {statistics.median(c):6.0f}")
    base_name = "Think-SFT (training tmpl)"
    if base_name in data:
        base = data[base_name]
        print(f"paired change in answer tokens vs {base_name}: median [95% CI]; share of prompts with a longer answer; thinking tokens change")
        for k in data:
            if k == base_name:
                continue
            idx = [i for i in base if i in data[k]]
            d = [data[k][i]["answer_tokens"] - base[i]["answer_tokens"] for i in idx]
            dt = [(data[k][i]["n_tokens"] - data[k][i]["answer_tokens"]) - (base[i]["n_tokens"] - base[i]["answer_tokens"]) for i in idx
                  if data[k][i]["finish_reason"] == "stop" and base[i]["finish_reason"] == "stop"]
            lo, hi = boot(d); longer = sum(x > 0 for x in d) / len(d)
            print(f"  {k:26s} {statistics.median(d):+6.0f} [{lo:+.0f}, {hi:+.0f}]  n={len(d)}  longer on {longer:.2f} of prompts | thinking {statistics.median(dt):+5.0f}")
    print("arm vs arm, paired on the same prompts: median difference in answer tokens [95% CI]; sign test; robustness (chars, 10% trimmed mean)")
    for x, y in (("arm B", "arm A"), ("arm B'", "arm A"), ("arm B'", "arm B")):
        if x not in data or y not in data:
            continue
        idx = [i for i in data[y] if i in data[x]]
        d = [data[x][i]["answer_tokens"] - data[y][i]["answer_tokens"] for i in idx]
        dc = [data[x][i]["answer_chars"] - data[y][i]["answer_chars"] for i in idx]
        lo, hi = boot(d); neg, pos, ties, z = sign_test(d)
        tm = trimmed_mean([data[x][i]["answer_tokens"] for i in idx]) - trimmed_mean([data[y][i]["answer_tokens"] for i in idx])
        print(f"  {x} − {y}: {statistics.median(d):+5.0f} [{lo:+.0f}, {hi:+.0f}] tokens, n={len(d)}; {x} shorter on {neg}, longer on {pos}, ties {ties}, z = {z:+.2f}; "
              f"chars {statistics.median(dc):+5.0f} [{boot(dc)[0]:+.0f}, {boot(dc)[1]:+.0f}]; trimmed-mean diff {tm:+.0f} tokens")
        by = {}
        for i in idx:
            by.setdefault(data[x][i].get("dataset", "?"), []).append(data[x][i]["answer_tokens"] - data[y][i]["answer_tokens"])
        print("     per source: " + "; ".join(f"{s} {statistics.median(v):+.0f} (n={len(v)})" for s, v in sorted(by.items())))
    if CMP:
        other_run = "run1" if RUN != "run1" else "run2"; osfx = "" if other_run == "run1" else other_run[3:]
        print(f"generation-noise check, same model {RUN} vs {other_run} on the same prompts: median paired difference [95% CI]; median answer tokens in each run")
        for k, v in FILES:
            v2 = v.replace(f"_tmplA{sfx}", f"_tmplA{osfx}").replace(f"_{RUN}", f"_{other_run}")
            p2 = os.path.join(CMP, v2 + ".jsonl")
            if k in data and os.path.exists(p2):
                o = load(p2); idx = [i for i in data[k] if i in o]
                d = [data[k][i]["answer_tokens"] - o[i]["answer_tokens"] for i in idx]; lo, hi = boot(d)
                print(f"  {k:26s} {RUN} − {other_run}: {statistics.median(d):+5.0f} [{lo:+.0f}, {hi:+.0f}]  n={len(d)}  medians {statistics.median([data[k][i]['answer_tokens'] for i in idx]):.0f} vs {statistics.median([o[i]['answer_tokens'] for i in idx]):.0f}")


if __name__ == "__main__":
    main()
