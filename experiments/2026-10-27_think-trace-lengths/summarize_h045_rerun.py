"""Summarize rerun_h045_turns.py outputs: per model, the share of samples with no </think> (no final answer) at the 8,192 budget,
overall and split by whether H045 answered that turn; per-turn counts; the loop measure (distinct 8-gram ratio over the last
2,000 words) for the no-answer samples versus the finished ones; median tokens of finished samples. Wilson 95% intervals.
Usage: summarize_h045_rerun.py OUT.txt FILE.jsonl [FILE.jsonl ...]"""
import json, math, statistics, sys, time
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pc(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if n else "n/a"


def tail_ratio(text, n=2000, k=8):
    w = text.split()[-n:]; g = [tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))]
    return len(set(g)) / len(g) if g else float("nan")


lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by summarize_h045_rerun.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
for path in sys.argv[2:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows: continue
    model = rows[0]["model"]; n = len(rows)
    noans = [r for r in rows if not r["has_think_end"]]; fin = [r for r in rows if r["has_think_end"]]
    cap = sum(r["finish_reason"] != "stop" for r in rows)
    byturn = defaultdict(list)
    for r in rows: byturn[(r["case"], r["turn"], r["h045_status"])].append(r)
    unf = [r for r in rows if r["h045_status"] == "UNFINISHED"]; ans = [r for r in rows if r["h045_status"] == "answered"]
    lines.append(f"\n== {model} ({path}): {len(byturn)} turns x {n // max(1, len(byturn))} samples = {n} samples; max_tokens {rows[0]['max_tokens']}")
    lines.append(f"no </think> (no final answer): {pc(len(noans), n)} | hit max_tokens: {pc(cap, n)}")
    lines.append(f"  on the {len({k for k in byturn if k[2]=='UNFINISHED'})} turns H045 left unfinished: {pc(sum(not r['has_think_end'] for r in unf), len(unf))}"
                 f" | on the {len({k for k in byturn if k[2]=='answered'})} turns H045 answered: {pc(sum(not r['has_think_end'] for r in ans), len(ans))}")
    turns_any = sum(any(not r["has_think_end"] for r in g) for g in byturn.values()); turns_all = sum(all(not r["has_think_end"] for r in g) for g in byturn.values())
    lines.append(f"  turns with at least one no-answer sample: {turns_any}/{len(byturn)}; turns where every sample failed: {turns_all}/{len(byturn)}")
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    ratios = [tail_ratio(r["text"]) for r in noans]
    loops = sum(r < 0.5 for r in ratios); long_ = len(noans) - loops
    lines.append(f"  loop measure, tail distinct 8-gram ratio: no-answer samples median {med(ratios):.2f} (n={len(noans)}); finished samples median {med([tail_ratio(r['text']) for r in fin]):.2f} (n={len(fin)})")
    lines.append(f"  no-answer samples split: loops (tail ratio < 0.5) {pc(loops, n)} of all samples | long but progressing (>= 0.5) {pc(long_, n)} of all samples"
                 f"   [H045 in the same 22 turns, 1 sample each: 13 loops + 3 long = 16 unfinished]")
    lines.append(f"  finished samples: median total tokens {med([r['n_tokens'] for r in fin]):.0f}; prompt tokens median {med([r['prompt_tokens'] for r in rows]):.0f}")
    lines.append(f"  {'case':32s} {'turn':>4s} {'H045':10s} {'no answer':>9s} {'median tokens':>13s}")
    for (case, turn, st), g in sorted(byturn.items()):
        lines.append(f"  {case[:32]:32s} {turn:4d} {st:10s} {sum(not r['has_think_end'] for r in g):4d}/{len(g):<4d} {med([r['n_tokens'] for r in g]):13.0f}")
out = "\n".join(lines) + "\n"; open(sys.argv[1], "w").write(out); print(out)
