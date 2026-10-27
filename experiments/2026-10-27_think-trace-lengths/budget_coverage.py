"""Share of SFT completions longer than each candidate thinking budget (8,192 / 12,288 / 16,384 / 24,576 / 32,768 tokens), re-weighted by
the dataset card's source counts as in make_tables.py, for Clean-v1 and the Olmo 3 7B mix, with the reasoning sources listed separately.
Usage: budget_coverage.py RESULTS_DIR OUT.txt"""
import os, sys, time
import pyarrow.parquet as pq
from composition_check import CARDS, match

R, out = sys.argv[1], sys.argv[2]
BUDGETS = (8192, 12288, 16384, 24576, 32768)
lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by budget_coverage.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
for ds, card in (("clean_v1", "32b"), ("think_sft_7b", "7b")):
    t = pq.read_table(os.path.join(R, f"rows_{ds}.parquet"), columns=["source", "completion_tokens"])
    src, comp = t.column("source").to_pylist(), t.column("completion_tokens").to_pylist()
    total, labels = CARDS[card]
    groups = {}
    for s, c in zip(src, comp):
        k = match(s)
        if k in labels: groups.setdefault(k, []).append(c)
    covered = sum(labels[k][1] for k in groups)
    lines.append(f"\n== {ds}: share of completions longer than the budget (dataset level, re-weighted; {100*covered/total:.1f}% of card rows covered)")
    lines.append("  " + " ".join(f"{'>'+format(b, ','):>10s}" for b in BUDGETS))
    lines.append("  " + " ".join(f"{100*sum((labels[k][1]/covered) * sum(c > b for c in g)/len(g) for k, g in groups.items()):9.1f}%" for b in BUDGETS))
    lines.append("  by source:")
    for k, g in sorted(groups.items(), key=lambda kv: -labels[kv[0]][1]):
        lines.append(f"  {labels[k][0][:34]:34s} " + " ".join(f"{100*sum(c > b for c in g)/len(g):9.1f}%" for b in BUDGETS))
txt = "\n".join(lines) + "\n"; open(out, "w").write(txt); print(txt)
