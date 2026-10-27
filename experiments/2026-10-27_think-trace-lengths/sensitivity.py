"""Tokenizer sensitivity of the "> 8,192 tokens" shares: the olmo35 tokenizer is not public, so the counts use the Olmo 3 tokenizer.
Recomputes the dataset-level share (re-weighted by card source counts, as in make_tables.py) with every completion length scaled
by 0.90, 0.95, 1.00, 1.05 and 1.10. Usage: sensitivity.py RESULTS_DIR"""
import os, sys, time
import pyarrow.parquet as pq
from composition_check import CARDS, match

R = sys.argv[1]
lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by sensitivity.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
for ds, card in (("clean_v1", "32b"), ("think_sft_7b", "7b")):
    t = pq.read_table(os.path.join(R, f"rows_{ds}.parquet"), columns=["source", "completion_tokens"])
    src, comp = t.column("source").to_pylist(), t.column("completion_tokens").to_pylist()
    total, labels = CARDS[card]
    groups = {}
    for s, c in zip(src, comp):
        k = match(s)
        if k in labels:
            groups.setdefault(k, []).append(c)
    covered = sum(labels[k][1] for k in groups)
    lines.append(f"\n== {ds} ({len(comp)} rows; {100*covered/total:.1f}% of card rows covered)")
    lines.append(f"{'scale':>6s} {'> 4,096':>9s} {'> 8,192':>9s} {'> 16,384':>9s}")
    for f in (0.90, 0.95, 1.00, 1.05, 1.10):
        shares = []
        for thr in (4096, 8192, 16384):
            shares.append(sum((labels[k][1] / covered) * sum(c * f > thr for c in g) / len(g) for k, g in groups.items()))
        lines.append(f"{f:6.2f} {100*shares[0]:8.1f}% {100*shares[1]:8.1f}% {100*shares[2]:8.1f}%")
txt = "\n".join(lines) + "\n"
open(os.path.join(R, "sensitivity.txt"), "w").write(txt); print(txt)
