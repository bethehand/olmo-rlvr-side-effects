"""Does a wrong rope configuration reproduce the loop signature? On 2026-10-23 arm A was first evaluated with a config that transformers
4.57 read wrongly (rope_theta 10,000, no YaRN; the "badrope" files), then re-run with the repaired config. Same prompts, same sampling,
32K cap. For each pair: share of outputs with no </think>, share that hit the cap, and the tail distinct 8-gram ratio of the no-answer outputs.
Usage: badrope_signature.py OUT.txt BAD.jsonl GOOD.jsonl [BAD.jsonl GOOD.jsonl ...]"""
import json, math, statistics, sys, time


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pc(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if n else "n/a"


def tail_ratio(text, n=2000, k=8):
    w = text.split()[-n:]; g = [tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))]
    return len(set(g)) / len(g) if g else float("nan")


lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by badrope_signature.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
args = sys.argv[2:]
for bad, good in zip(args[::2], args[1::2]):
    for label, path in (("wrong rope (badrope)", bad), ("repaired config", good)):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        n = len(rows); noans = [r for r in rows if not r.get("has_think_end")]
        cap = sum(r["finish_reason"] != "stop" for r in rows)
        ratios = sorted(tail_ratio(r["response"]) for r in noans)
        med = statistics.median(ratios) if ratios else float("nan")
        lines.append(f"{path.split('/')[-2]+'/'+path.split('/')[-1]:75s} {label:22s} n={n:4d} | no </think> {pc(len(noans), n)} | hit cap {pc(cap, n)} | no-answer tail ratio median {med:.2f}, <0.5: {sum(r < 0.5 for r in ratios)}")
    lines.append("")
txt = "\n".join(lines) + "\n"; open(sys.argv[1], "w").write(txt); print(txt)
