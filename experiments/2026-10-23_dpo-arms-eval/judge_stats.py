"""Extra statistics on the pairwise judgments (judgments/heldout_run1.jsonl): per pair the win-share difference with a bootstrap
95% interval (2000 resamples, seed 0) and a sign test over decided prompts; how often the longer answer won among decided
prompts; and the decided prompts split by which side wrote the longer answer, a post-hoc check of whether the judge's choices
follow length rather than the model, plus the decided prompts whose two answers are within 20 % of each other in length. A prompt is decided only when the judge picks the same model in both orders (as in
analyze_heldout.py); everything else is a tie. Usage: judge_stats.py JUDGMENTS.jsonl HELDOUT_OUTPUT_DIR"""
import collections, json, math, os, random, sys

J, H = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(J) if l.strip()]
pw = [r for r in rows if r["kind"] == "pairwise"]


def short(p):
    b = os.path.basename(p).replace("@main", "").replace(".jsonl", "").replace("icedduck_olmo3-7b-think-dpo-", "").replace("Olmo-3-7B-Think-", "")
    return b.replace("_seed1_run1", "").replace("_run1", "")


L = {f: {json.loads(l)["index"]: json.loads(l)["answer_tokens"] for l in open(os.path.join(H, f)) if l.strip()} for f in os.listdir(H) if f.endswith(".jsonl")}
pairs = collections.OrderedDict()
for r in pw:
    pairs.setdefault((r["x"], r["y"]), {}).setdefault(r["index"], {})[r["order"]] = r["winner"]
print("pair: n, wins (x / ties / y), y − x win share [95% CI], sign test z over decided prompts, share of decided prompts won by the longer answer")
pooled = collections.Counter()
for (x, y), d in pairs.items():
    res = []; strat = collections.Counter(); longer = 0
    for i, o in d.items():
        if "xy" not in o or "yx" not in o:
            continue
        v = 0
        if o["xy"] == o["yx"] and o["xy"] in ("x", "y"):
            v = 1 if o["xy"] == "y" else -1
        res.append(v)
        if v:
            lx, ly = L[os.path.basename(x)][i], L[os.path.basename(y)][i]
            side_longer = "x" if lx > ly else "y"
            strat[(side_longer, "x" if v == -1 else "y")] += 1
            longer += (side_longer == ("y" if v == 1 else "x"))
    n = len(res); wy = sum(v == 1 for v in res); wx = sum(v == -1 for v in res); dec = wx + wy
    random.seed(0); bs = sorted(sum(random.choice(res) for _ in range(n)) / n for _ in range(2000))
    z = (wy - wx) / math.sqrt(dec) if dec else float("nan")
    print(f"  {short(x):>9s} vs {short(y):<7s} n={n}  {wx:3d} / {n - dec:3d} / {wy:3d}  {(wy - wx) / n:+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]  z={z:+.2f}  longer wins {longer}/{dec} = {longer / dec if dec else float('nan'):.2f}")
    print(f"             when {short(x)} is longer: {short(x)} wins {strat[('x', 'x')]}, {short(y)} wins {strat[('x', 'y')]};  when {short(y)} is longer: {short(x)} wins {strat[('y', 'x')]}, {short(y)} wins {strat[('y', 'y')]}")
    # near-equal lengths: decided prompts whose answer lengths are within 20 % of each other (ratio between 0.8 and 1.25)
    near = collections.Counter()
    for i, o in d.items():
        if "xy" in o and "yx" in o and o["xy"] == o["yx"] and o["xy"] in ("x", "y"):
            lx, ly = L[os.path.basename(x)][i], L[os.path.basename(y)][i]
            if lx and ly and 0.8 <= ly / lx <= 1.25:
                near[o["xy"]] += 1
    print(f"             lengths within 20 % of each other: {short(x)} wins {near['x']}, {short(y)} wins {near['y']} (of {near['x'] + near['y']} decided)")
    if short(x).startswith("arm") and short(y).startswith("arm"):
        pooled["longer"] += longer; pooled["decided"] += dec
print(f"arm-vs-arm pairs pooled: the longer answer won {pooled['longer']} of {pooled['decided']} decided prompts = {pooled['longer'] / pooled['decided']:.2f}")
