"""Step 1 of the #2002 follow-up: what would length rules do to the three Dolci DPO sets?

Reads results/pairs_*.parquet (one row per pair; c_/r_ = chosen/rejected; *_tokens and *_chars are the answer after
the last </think>, *_think_chars the thinking block). Rules are applied to the answer part, so for the Think sets a
rule leaves the thinking gap untouched. Writes results/filter_rules.md and prints it."""
import pandas as pd

SETS = [("Dolci-Think-DPO-32B", "think32b"), ("Dolci-Think-DPO-7B", "think7b"), ("Dolci-Instruct-DPO", "instruct")]
RULES = [
    ("Ai2 §5.3.1: chosen ≤ rejected + 100 tokens", lambda d: (d.c_tokens - d.r_tokens) <= 100),
    ("chosen ≤ rejected + 300 tokens", lambda d: (d.c_tokens - d.r_tokens) <= 300),
    ("chosen/rejected ≤ 1.25 (tokens)", lambda d: d.c_tokens <= 1.25 * d.r_tokens),
    ("chosen/rejected ≤ 1.5 (tokens)", lambda d: d.c_tokens <= 1.5 * d.r_tokens),
    ("chosen/rejected ≤ 2 (tokens)", lambda d: d.c_tokens <= 2.0 * d.r_tokens),
]


def describe(g):
    diff = g.c_tokens - g.r_tokens
    ratio = (g.c_tokens + 1) / (g.r_tokens + 1)
    out = dict(n=len(g), ratio_med=ratio.median(), chosen_longer=(g.c_tokens > g.r_tokens).mean() * 100,
               over100=(diff > 100).mean() * 100, p80=diff.quantile(0.8), c_med=g.c_tokens.median(), r_med=g.r_tokens.median())
    if g.c_think_chars.gt(0).any():
        out["think_ratio_med"] = ((g.c_think_chars + 1) / (g.r_think_chars + 1)).median()
    return out


lines = ["# Length rules on the Dolci DPO sets (answer tokens, Olmo 3 tokenizer)", ""]
lines += ["Per-pair ratio = (chosen+1)/(rejected+1) answer tokens; p80 = 80th percentile of chosen − rejected tokens.",
          "For the Think sets the rules act on the answer after `</think>`; the thinking ratio column shows what stays of the thinking-length gap.", ""]
summary_rows = []
for name, key in SETS:
    d = pd.read_parquet(f"results/pairs_{key}.parquet")
    base = describe(d)
    lines += [f"## {name} ({base['n']:,} pairs; unfiltered: ratio {base['ratio_med']:.2f}, chosen longer {base['chosen_longer']:.0f}%, "
              f">100 tokens {base['over100']:.0f}%, p80 {base['p80']:.0f} tokens" + (f", thinking ratio {base['think_ratio_med']:.2f}" if "think_ratio_med" in base else "") + ")", ""]
    lines += ["| rule | pairs kept | median ratio after | chosen longer after | >100 tokens after | p80 after | thinking ratio after |", "|---|---|---|---|---|---|---|"]
    for rname, fn in RULES:
        k = describe(d[fn(d)])
        lines.append(f"| {rname} | {k['n']:,} ({k['n'] / base['n'] * 100:.0f}%) | {k['ratio_med']:.2f} | {k['chosen_longer']:.0f}% | {k['over100']:.0f}% | {k['p80']:.0f} | "
                     + (f"{k['think_ratio_med']:.2f}" if "think_ratio_med" in k else "–") + " |")
        summary_rows.append((name, rname, k["n"] / base["n"] * 100, k["ratio_med"]))
    lines.append("")
    if key == "instruct":
        lines += ["### by preference type (unfiltered, then after Ai2's 100-token rule)", "",
                  "| preference type | pairs | ratio | chosen longer | >100 tokens | p80 | kept by 100-token rule | ratio after |", "|---|---|---|---|---|---|---|---|"]
        for t, g in d.groupby("preference_type"):
            b = describe(g); k = describe(g[RULES[0][1](g)])
            lines.append(f"| {t} | {b['n']:,} | {b['ratio_med']:.2f} | {b['chosen_longer']:.0f}% | {b['over100']:.0f}% | {b['p80']:.0f} | {k['n'] / b['n'] * 100:.0f}% | {k['ratio_med']:.2f} |")
        lines.append("")
    # by source
    src = d.groupby("source")
    rows = []
    for s, g in src:
        b = describe(g); rows.append((s, b, (g.c_tokens - g.r_tokens > 100).sum()))
    tot_over = (d.c_tokens - d.r_tokens > 100).sum()
    rows.sort(key=lambda r: -r[1]["n"])
    lines += [f"### by source (sources with ≥ 2% of the pairs; {len(rows)} sources in all)", "",
              "| source | pairs | share | ratio | chosen longer | >100 tokens | share of all >100-token pairs |", "|---|---|---|---|---|---|---|"]
    for s, b, n_over in rows:
        if b["n"] / base["n"] < 0.02:
            continue
        lines.append(f"| {s} | {b['n']:,} | {b['n'] / base['n'] * 100:.0f}% | {b['ratio_med']:.2f} | {b['chosen_longer']:.0f}% | {b['over100']:.0f}% | {n_over / tot_over * 100:.0f}% |")
    heavy = [r for r in rows if r[1]["ratio_med"] >= 2.0 and r[1]["n"] >= 500]
    if heavy:
        lines += ["", "Sources with median ratio ≥ 2 (≥ 500 pairs): " + "; ".join(f"{s} ({b['n']:,}, {b['ratio_med']:.2f})" for s, b, _ in heavy)]
    lines.append("")
text = "\n".join(lines)
open("results/filter_rules.md", "w").write(text)
print(text)
