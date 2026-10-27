"""Markdown table for the same-prompt rerun (rerun_h045_turns.py outputs): per model, share of samples with no </think> at the
8,192 budget, split into loops (tail distinct 8-gram ratio < 0.5) and long-but-progressing (>= 0.5), the same share on the turns
H045 left unfinished versus answered, and how many of the 22 turns had at least one failing sample. Wilson 95% intervals.
Usage: rerun_table.py OUT.md LABEL=FILE.jsonl [LABEL=FILE.jsonl ...]"""
import json, math, sys, time
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pc(k, n):
    lo, hi = wilson(k, n); return f"{100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def tail_ratio(text, n=2000, k=8):
    w = text.split()[-n:]; g = [tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))]
    return len(set(g)) / len(g) if g else float("nan")


lines = ["| model | samples | no `</think>` at 8,192 | of which loops (tail ratio < 0.5) | of which long, still progressing | on the 16 turns H045 left unfinished | on the 6 turns H045 answered | turns with ≥1 failing sample |",
         "|---|---|---|---|---|---|---|---|"]
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows); noans = [r for r in rows if not r["has_think_end"]]
    loops = sum(tail_ratio(r["text"]) < 0.5 for r in noans); long_ = len(noans) - loops
    unf = [r for r in rows if r["h045_status"] == "UNFINISHED"]; ans = [r for r in rows if r["h045_status"] == "answered"]
    byturn = defaultdict(list)
    for r in rows: byturn[(r["case"], r["turn"])].append(r)
    t_any = sum(any(not r["has_think_end"] for r in g) for g in byturn.values())
    lines.append(f"| {label} | {n} | {pc(len(noans), n)} | {pc(loops, n)} | {pc(long_, n)} | {pc(sum(not r['has_think_end'] for r in unf), len(unf))} | {pc(sum(not r['has_think_end'] for r in ans), len(ans))} | {t_any}/{len(byturn)} |")
lines.append("| Olmo 3.5 small SFT H045 (issue author's run, 1 sample per turn) | 22 | 16/22 = 72.7% | 13/22 = 59.1% | 3/22 = 13.6% | 16/16 | 0/6 | 16/22 |")
out = "\n".join(lines) + "\n"
open(sys.argv[1], "w").write("<!-- Produced by rerun_table.py on " + time.strftime("%Y-%m-%d %H:%M %Z") + " -->\n" + out); print(out)
