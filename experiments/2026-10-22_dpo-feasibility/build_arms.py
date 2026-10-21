"""Build the training arms for the #2002 DPO test from Dolci-Think-DPO-7B (answer tokens = text after </think>).

A  original:   N random pairs.
B  gap removed: pairs with chosen/rejected answer-token ratio <= 1.5, sampled to match A's chosen-length decile
               distribution. The ratio rule, unlike Ai2's 100-token cap, keeps answer length (pool: chosen median
               346 vs 329 tokens) and only removes the "chosen is longer" signal.
C  control:    pairs from the whole pool matched to B cell by cell on (source, chosen-length decile), so B vs C differs
               only in the rejected side (the length gap) and A vs C only in the source / length mix.
D  Ai2 cap:    pairs with chosen <= rejected + 100 tokens (Ai2's §5.3.1 rule), random N. Both answers end up short
               (median 109 / 92 tokens), so D tests the rule as Ai2 applies it, not the gap alone.

Inputs: pairs_think7b.parquet next to this script (id, source, c_tokens, r_tokens, c_think_chars, r_think_chars) and the
dataset from the Hugging Face cache. Writes arm_{A,B,C,D}.jsonl (original columns), arm_*_ids.json, arms_summary.md.
Usage: python build_arms.py OUT_DIR [N=20000] [SEED=1]"""
import json, os, sys
import numpy as np, pandas as pd
from datasets import load_dataset

out, N, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000, int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(seed)
pairs = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_think7b.parquet"))
pairs["diff"] = pairs.c_tokens - pairs.r_tokens
pairs["ratio"] = (pairs.c_tokens + 1) / (pairs.r_tokens + 1)
edges = np.quantile(pairs.c_tokens, np.linspace(0, 1, 11)); edges[-1] += 1
pairs["decile"] = pd.cut(pairs.c_tokens, edges, include_lowest=True, labels=False).astype(int)

def sample(df, n):
    return df.sample(n=min(n, len(df)), random_state=rng)

A = sample(pairs, N)
# B: ratio <= 1.5, matched to A's decile counts
pool_b = pairs[pairs.ratio <= 1.5]
B = pd.concat([sample(pool_b[pool_b.decile == dec], k) for dec, k in A.decile.value_counts().items()])
# C: whole pool, matched to B on (source, decile); cells without enough pairs are topped up within the same decile
cells = B.groupby(["source", "decile"]).size()
parts, short = [], 0
for (src, dec), k in cells.items():
    cell = pairs[(pairs.source == src) & (pairs.decile == dec)]
    got = sample(cell, k); parts.append(got); short += k - len(got)
C = pd.concat(parts)
if len(C) < N:
    need = B.decile.value_counts() - C.decile.value_counts().reindex(B.decile.unique(), fill_value=0)
    rest = pairs.drop(C.index)
    parts = [C] + [sample(rest[rest.decile == dec], int(k)) for dec, k in need.items() if k > 0]
    C = pd.concat(parts)
D = sample(pairs[pairs["diff"] <= 100], N)

def describe(d, name):
    return dict(arm=name, pairs=len(d), ratio_median=round(d.ratio.median(), 3), chosen_longer=round((d.c_tokens > d.r_tokens).mean() * 100, 1),
                over_cap=round((d["diff"] > 100).mean() * 100, 1), chosen_med=int(d.c_tokens.median()), rejected_med=int(d.r_tokens.median()),
                chosen_mean=int(d.c_tokens.mean()), think_ratio=round(((d.c_think_chars + 1) / (d.r_think_chars + 1)).median(), 3), sources=len(d.source.unique()))

rows = [describe(pairs, "pool (150K)"), describe(A, "A original"), describe(B, "B gap removed (ratio ≤ 1.5, chosen-length matched to A)"),
        describe(C, "C control (B's source × length cells, no rule)"), describe(D, "D Ai2 cap (chosen ≤ rejected + 100 tokens)")]
ds = load_dataset("allenai/Dolci-Think-DPO-7B", split="train")
by_id = {x: i for i, x in enumerate(ds["id"])}
for name, d in (("A", A), ("B", B), ("C", C), ("D", D)):
    sub = ds.select([by_id[i] for i in d.id])
    sub.to_json(os.path.join(out, f"arm_{name}.jsonl"), lines=True, force_ascii=False)
    json.dump(sorted(d.id.tolist()), open(os.path.join(out, f"arm_{name}_ids.json"), "w"))
lines = [f"# Training arms (seed {seed}, {N} pairs each; C had {short} cells topped up within the decile)", "",
         "| arm | pairs | median answer-token ratio | chosen longer | over the 100-token cap | chosen / rejected median tokens | chosen mean | thinking ratio | sources |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['arm']} | {r['pairs']:,} | {r['ratio_median']} | {r['chosen_longer']}% | {r['over_cap']}% | {r['chosen_med']} / {r['rejected_med']} | {r['chosen_mean']} | {r['think_ratio']} | {r['sources']} |")
lines += ["", "Chosen-length deciles (pool edges in tokens): " + ", ".join(str(int(e)) for e in edges[:-1]), "", "| arm | " + " | ".join(f"d{i}" for i in range(10)) + " |", "|---|" + "---|" * 10]
for name, d in (("A", A), ("B", B), ("C", C), ("D", D)):
    vc = d.decile.value_counts().reindex(range(10), fill_value=0)
    lines.append(f"| {name} | " + " | ".join(f"{int(v)}" for v in vc) + " |")
lines += ["", "Source mix (share of pairs):", "", "| source | pool | A | B | C | D |", "|---|---|---|---|---|---|"]
for src in pairs.source.value_counts().index[:15]:
    f = lambda d: f"{(d.source == src).mean() * 100:.1f}%"
    lines.append(f"| {src} | {f(pairs)} | {f(A)} | {f(B)} | {f(C)} | {f(D)} |")
open(os.path.join(out, "arms_summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
