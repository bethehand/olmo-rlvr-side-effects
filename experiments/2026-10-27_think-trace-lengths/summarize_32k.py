"""Summarize the 32K-budget reruns of the 22 turns (rerun_h045_turns.py with max_tokens 32768) and compare them with the 8K runs.
For every sample: the sampled-token position of </think> (tokens of the text up to and including the tag, Olmo 3 tokenizer), so the
share of samples that close by 8,192 / 16,384 / 24,576 / 32,768; samples still open at 32,768 are split into loops (tail distinct
8-gram ratio < 0.5) and long traces. Consistency with the 8K run of the same model and seeds: share of (turn, sample) pairs whose
first 1,500 characters are identical, and the 8K run's no-answer count next to the 32K run's "not closed by 8,192" count.
Usage: summarize_32k.py OUT.txt TABLE.md LABEL=RUN32K.jsonl:RUN8K.jsonl [...]"""
import json, math, statistics, sys, time
from collections import defaultdict

TOK_NAME = "allenai/Olmo-3-7B-Think-SFT"


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pc(k, n):
    lo, hi = wilson(k, n); return f"{100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if n else "n/a"


def tail_ratio(text, n=2000, k=8):
    w = text.split()[-n:]; g = [tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))]
    return len(set(g)) / len(g) if g else float("nan")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_NAME)
    lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by summarize_32k.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
    table = ["| model (22 turns × 8 samples) | not closed by 8,192 | not closed by 16,384 | not closed by 24,576 | still open at 32,768 | of which loops | of which long | 8K run: no `</think>` (for reference) |",
             "|---|---|---|---|---|---|---|---|"]
    for spec in sys.argv[3:]:
        label, paths = spec.split("=", 1); p32, p8 = paths.split(":")
        r32 = [json.loads(l) for l in open(p32) if l.strip()]; r8 = [json.loads(l) for l in open(p8) if l.strip()]
        n = len(r32)
        close_pos = []
        for r in r32:
            if r["has_think_end"]:
                idx = r["text"].find("</think>")
                close_pos.append(len(tok(r["text"][:idx + len("</think>")], add_special_tokens=False)["input_ids"]))
            else:
                close_pos.append(None)
        not_by = {b: sum(1 for c in close_pos if c is None or c > b) for b in (8192, 16384, 24576)}
        open32 = [r for r in r32 if not r["has_think_end"]]
        loops = sum(tail_ratio(r["text"]) < 0.5 for r in open32); long_ = len(open32) - loops
        k8 = {(r["case"], r["turn"], r["sample"]): r for r in r8}
        same = sum(1 for r in r32 if (r["case"], r["turn"], r["sample"]) in k8 and k8[(r["case"], r["turn"], r["sample"])]["text"][:1500] == r["text"][:1500])
        noans8 = sum(not r["has_think_end"] for r in r8)
        lines.append(f"\n== {label}: {n} samples at max_tokens {r32[0]['max_tokens']}")
        lines.append(f"not closed by 8,192: {pc(not_by[8192], n)} | by 16,384: {pc(not_by[16384], n)} | by 24,576: {pc(not_by[24576], n)} | still open at 32,768: {pc(len(open32), n)} (loops {loops}, long {long_})")
        lines.append(f"consistency with the 8K run (same seeds): identical first 1,500 chars in {same}/{n} samples; 8K run no-answer {noans8}/{len(r8)} vs 32K run not closed by 8,192 {not_by[8192]}/{n}")
        closed = [c for c in close_pos if c is not None]
        lines.append(f"close position of finished samples: median {statistics.median(closed):.0f} tokens, p90 {sorted(closed)[int(0.9*(len(closed)-1))]:.0f}, max {max(closed)}")
        byturn = defaultdict(list)
        for r, c in zip(r32, close_pos): byturn[(r["case"], r["turn"], r["h045_status"])].append(c)
        lines.append(f"  {'case':32s} {'turn':>4s} {'H045':10s} {'open@8K':>7s} {'open@16K':>8s} {'open@32K':>8s} {'median close':>12s}")
        for (case, turn, st), cs in sorted(byturn.items()):
            o8 = sum(1 for c in cs if c is None or c > 8192); o16 = sum(1 for c in cs if c is None or c > 16384); o32 = sum(1 for c in cs if c is None)
            med = statistics.median([c for c in cs if c is not None]) if any(c is not None for c in cs) else float("nan")
            lines.append(f"  {case[:32]:32s} {turn:4d} {st:10s} {o8:4d}/8  {o16:5d}/8  {o32:5d}/8  {med:12.0f}")
        table.append(f"| {label} | {pc(not_by[8192], n)} | {pc(not_by[16384], n)} | {pc(not_by[24576], n)} | {pc(len(open32), n)} | {pc(loops, n)} | {pc(long_, n)} | {pc(noans8, len(r8))} |")
    txt = "\n".join(lines) + "\n"; open(sys.argv[1], "w").write(txt); print(txt)
    open(sys.argv[2], "w").write("<!-- Produced by summarize_32k.py on " + time.strftime("%Y-%m-%d %H:%M %Z") + " -->\n" + "\n".join(table) + "\n"); print("\n".join(table))


if __name__ == "__main__":
    main()
