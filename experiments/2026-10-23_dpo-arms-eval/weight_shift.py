"""How far each DPO arm moved from the start model (Olmo-3-7B-Think-SFT), read directly from the safetensors files on CPU:
relative L2 norm of the parameter delta ||theta_arm - theta_start|| / ||theta_start||, overall and by parameter group, the share of
the squared delta per group, and the cosine similarity between the arms' deltas (overall and per group). Same steps, learning rate
and batch for every arm, so a smaller delta means smaller gradients, i.e. smaller preference margins to learn from; similar deltas
with a different behaviour point at what changed in the data, not how much was learned.
Usage: weight_shift.py START_SNAPSHOT_DIR LABEL=ARM_SNAPSHOT_DIR [LABEL=DIR ...] [--limit=N] [--threads=T]
  (snapshot dirs hold model*.safetensors; --limit=N reads only the first N tensors, for a smoke run; default 8 threads)"""
import glob, math, os, sys

import torch
from safetensors import safe_open

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
limit = int(opt.get("--limit", 0)); torch.set_num_threads(int(opt.get("--threads", 8)))
start_dir, arms = args[0], [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args[1:]]


def handles(d):
    hs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = safe_open(f, "pt")
        for k in h.keys():
            hs[k] = h
    return hs


def group(name):
    if "embed" in name or "lm_head" in name:
        return "embeddings/lm_head"
    if "norm" in name:
        return "norms"
    if any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(x in name for x in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    return "other"


start = handles(start_dir); armh = {lab: handles(d) for lab, d in arms}
names = sorted(start)
for lab, h in armh.items():
    missing = [n for n in names if n not in h]; extra = [n for n in h if n not in start]
    print(f"{lab}: {len(h)} tensors; missing vs start {len(missing)} {missing[:3]}; extra {len(extra)} {extra[:3]}", flush=True)
if limit:
    names = names[:limit]
labs = [lab for lab, _ in arms]
groups = ["attention", "mlp", "embeddings/lm_head", "norms", "other", "ALL"]
ss_start = {g: 0.0 for g in groups}; ss_delta = {(l, g): 0.0 for l in labs for g in groups}
dot = {(a, b, g): 0.0 for i, a in enumerate(labs) for b in labs[i + 1:] for g in groups}
n_params = 0
for i, n in enumerate(names):
    g = group(n); s = start[n].get_tensor(n).float(); n_params += s.numel()
    d = {lab: armh[lab][n].get_tensor(n).float() - s for lab in labs}
    for gg in (g, "ALL"):
        ss_start[gg] += float((s * s).sum())
        for lab in labs:
            ss_delta[(lab, gg)] += float((d[lab] * d[lab]).sum())
        for j, a in enumerate(labs):
            for b in labs[j + 1:]:
                dot[(a, b, gg)] += float((d[a] * d[b]).sum())
    if i % 50 == 0 or i == len(names) - 1:
        print(f"  {i + 1}/{len(names)} {n} {tuple(s.shape)}", flush=True)
print(f"\n{len(names)} tensors, {n_params / 1e9:.3f} B parameters, start model {start_dir}")
print("relative L2 delta ||arm - start|| / ||start|| (x 1e-3), overall and by group; in brackets the group's share of the arm's squared delta")
print("  group               " + "".join(f"{l:>26s}" for l in labs))
for g in groups:
    if ss_start[g] == 0:
        continue
    row = f"  {g:20s}"
    for l in labs:
        rel = math.sqrt(ss_delta[(l, g)] / ss_start[g]) * 1e3; share = ss_delta[(l, g)] / ss_delta[(l, "ALL")] if g != "ALL" else 1.0
        row += f"{rel:14.3f} [{share:5.1%}]     "
    print(row)
print("delta norm ratios: " + ", ".join(f"{b}/{a} = {math.sqrt(ss_delta[(b, 'ALL')] / ss_delta[(a, 'ALL')]):.3f}" for i, a in enumerate(labs) for b in labs[i + 1:]))
print("cosine similarity between the arms' deltas, overall and by group")
for g in groups:
    if ss_start[g] == 0:
        continue
    print(f"  {g:20s}" + "".join(f"  {a}~{b} {dot[(a, b, g)] / math.sqrt(ss_delta[(a, g)] * ss_delta[(b, g)]):+.3f}" for i, a in enumerate(labs) for b in labs[i + 1:]))
