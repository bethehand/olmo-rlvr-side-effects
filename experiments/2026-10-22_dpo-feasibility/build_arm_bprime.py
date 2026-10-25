"""Build arm B' (written 2026-10-23, after A and B had trained; see ../2026-10-23_dpo-arms-eval/README.md for why).

B' is B done properly: the small-gap pairs (chosen/rejected answer-token ratio <= 1.5) sampled to match A *cell by cell* on
source x chosen-length decile x prompt-length tercile, so that only the length gap (and what is tied to it: rejected length,
prompt length within a tercile, thinking-length ratio) differs from A. Cells the small-gap subset cannot fill are topped up
within the same source x decile, then within the decile. B matched A on chosen-length decile only, which let the source mix drift
(math 19.7% vs 8.4%) and the prompts get longer (median 710 vs 482 chars).

Inputs: the per-pair table pairs_think7b.parquet (../2026-10-20_dpo-pairs/results) and arms/arm_A_ids.json (and arm_B_ids.json for
the comparison table). Writes arms/arm_Bp_ids.json and arms/arm_Bp_summary.md. Sampling seed 1. The JSONL is rebuilt on the
training node by rented/rebuild_arms_from_ids.py OUT Bp.
Usage: python build_arm_bprime.py PAIRS_PARQUET ARMS_DIR [SEED=1] [RATIO=1.5]"""
import json, os, sys
import numpy as np, pandas as pd

pq, arms_dir = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ratio_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
p = pd.read_parquet(pq)
p["ratio"] = (p.c_tokens + 1) / (p.r_tokens + 1)
edges = np.quantile(p.c_tokens, np.linspace(0, 1, 11)); edges[-1] += 1
p["decile"] = pd.cut(p.c_tokens, edges, include_lowest=True, labels=False).astype(int)
pe = np.quantile(p.prompt_chars, [0, 1 / 3, 2 / 3, 1]); pe[-1] += 1
p["ptile"] = pd.cut(p.prompt_chars, pe, include_lowest=True, labels=False).astype(int)
A = p[p.id.isin(set(json.load(open(os.path.join(arms_dir, "arm_A_ids.json")))))]
B = p[p.id.isin(set(json.load(open(os.path.join(arms_dir, "arm_B_ids.json")))))] if os.path.exists(os.path.join(arms_dir, "arm_B_ids.json")) else None
sub = p[p.ratio <= ratio_max]
rng = np.random.RandomState(seed)
parts, filled3 = [], 0
for (s, d, t), k in A.groupby(["source", "decile", "ptile"]).size().items():
    cell = sub[(sub.source == s) & (sub.decile == d) & (sub.ptile == t)]
    got = cell.sample(n=min(k, len(cell)), random_state=rng); parts.append(got); filled3 += len(got)
Bp = pd.concat(parts); rest = sub.drop(Bp.index)
quota2 = A.groupby(["source", "decile"]).size()
need2 = quota2 - Bp.groupby(["source", "decile"]).size().reindex(quota2.index, fill_value=0)
filled2 = 0
for (s, d), k in need2[need2 > 0].items():
    cell = rest[(rest.source == s) & (rest.decile == d)]
    got = cell.sample(n=min(int(k), len(cell)), random_state=rng); parts.append(got); rest = rest.drop(got.index); filled2 += len(got)
Bp = pd.concat(parts)
need1 = A.groupby("decile").size() - Bp.groupby("decile").size().reindex(range(10), fill_value=0)
filled1 = 0
for d, k in need1[need1 > 0].items():
    cell = rest[rest.decile == d]
    got = cell.sample(n=min(int(k), len(cell)), random_state=rng); parts.append(got); rest = rest.drop(got.index); filled1 += len(got)
Bp = pd.concat(parts)
assert len(Bp) == len(A) and Bp.id.is_unique, (len(Bp), len(A))
json.dump(sorted(Bp.id.tolist()), open(os.path.join(arms_dir, "arm_Bp_ids.json"), "w"))


def row(d, name):
    think = ((d.c_think_chars + 1) / (d.r_think_chars + 1)).median()
    return (f"| {name} | {len(d):,} | {d.ratio.median():.2f} | {(d.c_tokens > d.r_tokens).mean() * 100:.0f}% | {int(d.c_tokens.median())} / {int(d.r_tokens.median())} "
            f"| {int(d.prompt_chars.median())} / {int(d.prompt_chars.mean())} | {think:.2f} | {int(d.c_think_chars.median())} / {int(d.r_think_chars.median())} "
            f"| {d.source.str.contains('math').mean() * 100:.1f}% | {len(set(d.id) & set(A.id)):,} |")


lines = [f"# Arm B' (seed {seed}, ratio <= {ratio_max}): {filled3:,} pairs matched on source x chosen-length decile x prompt-length tercile, "
         f"{filled2:,} topped up within source x decile, {filled1:,} within decile", "",
         "| arm | pairs | median answer-token ratio | chosen longer | chosen / rejected median tokens | prompt chars median / mean | thinking ratio | chosen / rejected thinking chars | math sources | shared with A |",
         "|---|---|---|---|---|---|---|---|---|---|", row(A, "A")]
if B is not None:
    lines.append(row(B, "B"))
lines.append(row(Bp, "B'"))
lines += ["", "Source mix (share of pairs), A / B / B':", "", "| source | A | B | B' |", "|---|---|---|---|"]
for s in A.source.value_counts().index[:15]:
    f = lambda d: f"{(d.source == s).mean() * 100:.1f}%" if d is not None else "-"
    lines.append(f"| {s} | {f(A)} | {f(B)} | {f(Bp)} |")
open(os.path.join(arms_dir, "arm_Bp_summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
