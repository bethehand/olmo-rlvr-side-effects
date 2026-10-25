"""Tables for the held-out set (150 prompts, three categories) across models.
  lengths      per model and category: median answer tokens [bootstrap 95% CI], share of outputs that hit the 32K cap
  short        per model: share correct (a gold string appears in the answer, case-insensitive), share over-answered
               (answer longer than --short_cap tokens, default 60), with Wilson 95% intervals; lists the misses for a manual look
  paired       per category: per-prompt change in answer tokens from the first file to each other file, median [95% CI]
  judgments    if --judgments is given: pairwise win/tie/loss counts per pair and category (a model wins a prompt only when
               it wins in both orders), and rubric coverage per model (mean share of points covered, bootstrap 95% CI)
Usage: analyze_heldout.py OUT.jsonl [OUT2.jsonl ...] [--judgments judgments.jsonl] [--short_cap 60]"""
import argparse, json, math, random, re, statistics


def load(path):
    return {json.loads(l)["index"]: json.loads(l) for l in open(path) if l.strip()}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def boot(vals, stat, n=2000):
    random.seed(0)
    bs = sorted(stat([random.choice(vals) for _ in vals]) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def short_name(path):
    return re.sub(r"\.jsonl$", "", path.split("/")[-1]).replace("@main", "")


def correct(row):
    a = row["answer"].lower()
    return any(g.lower() in a for g in (row.get("gold") or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judgments", default=None)
    ap.add_argument("--short_cap", type=int, default=60)
    ap.add_argument("--prompts", default=None, help="heldout_prompts.jsonl: take gold strings from here instead of the output rows")
    args = ap.parse_args()
    data = {f: load(f) for f in args.files}
    if args.prompts:
        gold = {i: r.get("gold") for i, r in enumerate(json.loads(l) for l in open(args.prompts) if l.strip())}
        for rows in data.values():
            for i, r in rows.items():
                r["gold"] = gold.get(i)
    cats = ("short", "detail", "open")
    print("== answer length (tokens after </think>): median [95% CI], n, hit cap")
    for f, rows in data.items():
        line = [f"{short_name(f):42s}"]
        for c in cats:
            v = [r["answer_tokens"] for r in rows.values() if r["category"] == c]
            if not v:
                line.append(f"{c}: -"); continue
            lo, hi = boot(v, statistics.median)
            cap = sum(r["finish_reason"] == "length" for r in rows.values() if r["category"] == c)
            line.append(f"{c} {statistics.median(v):6.0f} [{lo:.0f}, {hi:.0f}] n={len(v)} cap={cap}")
        print("  " + " | ".join(line))
    print(f"\n== short questions: correct (gold string in answer), over-answered (> {args.short_cap} tokens)")
    for f, rows in data.items():
        s = [r for r in rows.values() if r["category"] == "short"]
        if not s:
            continue
        k = sum(correct(r) for r in s); o = sum(r["answer_tokens"] > args.short_cap for r in s); n = len(s)
        print(f"  {short_name(f):42s} correct {k}/{n} = {k/n:.2f} [{wilson(k, n)[0]:.2f}, {wilson(k, n)[1]:.2f}]"
              f"   over-answered {o}/{n} = {o/n:.2f} [{wilson(o, n)[0]:.2f}, {wilson(o, n)[1]:.2f}]")
        miss = [r["id"] + ":" + r["answer"].strip().replace("\n", " ")[:60] for r in s if not correct(r)]
        if miss:
            print("     misses: " + " || ".join(miss))
    if len(args.files) > 1:
        base = data[args.files[0]]
        print(f"\n== paired change in answer tokens from {short_name(args.files[0])}: median [95% CI]")
        for f in args.files[1:]:
            line = [f"{short_name(f):42s}"]
            for c in cats:
                idx = [i for i in base if i in data[f] and base[i]["category"] == c]
                d = [data[f][i]["answer_tokens"] - base[i]["answer_tokens"] for i in idx]
                if not d:
                    continue
                lo, hi = boot(d, statistics.median)
                line.append(f"{c} {statistics.median(d):+6.0f} [{lo:+.0f}, {hi:+.0f}] n={len(d)}")
            print("  " + " | ".join(line))
    if args.judgments:
        J = [json.loads(l) for l in open(args.judgments) if l.strip()]
        pairs = {}
        for j in J:
            if j["kind"] != "pairwise":
                continue
            pairs.setdefault((j["x"], j["y"]), {}).setdefault(j["index"], {})[j["order"]] = (j["winner"], j["category"])
        print("\n== pairwise (Qwen3-32B judge, both orders; a win needs both orders)")
        for (x, y), per in pairs.items():
            print(f"  {short_name(x)}  vs  {short_name(y)}")
            for c in cats + ("all",):
                w = l = t = 0
                for i, o in per.items():
                    if len(o) < 2:
                        continue
                    if c != "all" and o["xy"][1] != c:
                        continue
                    a, b = o["xy"][0], o["yx"][0]
                    if a == b == "x":
                        w += 1
                    elif a == b == "y":
                        l += 1
                    else:
                        t += 1
                n = w + l + t
                if n:
                    agree = (w + l) / n
                    print(f"     {c:6s} n={n:3d}  {short_name(x)[:18]} wins {w} ({w/n:.2f})  ties/disagree {t} ({t/n:.2f})  {short_name(y)[:18]} wins {l} ({l/n:.2f})  orders agree {agree:.2f}")
        cov = {}
        for j in J:
            if j["kind"] == "rubric" and j.get("covered") is not None:
                cov.setdefault(j["file"], []).append(sum(bool(v) for v in j["covered"]) / len(j["covered"]))
        if cov:
            print("\n== rubric coverage on detail questions (share of points covered): mean [95% CI]")
            for f, v in cov.items():
                lo, hi = boot(v, lambda z: sum(z) / len(z))
                print(f"  {short_name(f):42s} {sum(v)/len(v):.3f} [{lo:.3f}, {hi:.3f}] n={len(v)}")


if __name__ == "__main__":
    main()
