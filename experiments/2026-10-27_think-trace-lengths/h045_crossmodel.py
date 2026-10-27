"""Per-turn view across the four same-prompt reruns: for each of the 22 turns, how many of the 32 Olmo 3 samples (4 models x 8)
had no </think> at 8,192, how many of those were loops (tail distinct 8-gram ratio < 0.5), next to H045's outcome on that turn
(status and tail ratio from h045_chatlog_loops.txt). Then: on how many of the turns where H045 looped did no Olmo 3 sample fail at all?
Also prints the tails of a few Olmo 3 7B Think-SFT no-answer samples that are *not* loops, to check that they are still progressing.
Usage: h045_crossmodel.py OUT.txt H045_LOOPS.txt LABEL=FILE.jsonl ..."""
import json, re, sys, time
from collections import defaultdict


def tail_ratio(text, n=2000, k=8):
    w = text.split()[-n:]; g = [tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))]
    return len(set(g)) / len(g) if g else float("nan")


out, loops_file, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
h045 = {}
for l in open(loops_file):
    m = re.match(r"^(\S+)\s+(\d+)\s+(UNFINISHED|answered)\s+(\d+)/(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", l)
    if m: h045[(m.group(1), int(m.group(2)))] = (m.group(3), float(m.group(7)))
per_turn = defaultdict(lambda: {"n": 0, "noans": 0, "loops": 0, "by_model": {}})
sft_long = []
for spec in specs:
    label, path = spec.split("=", 1)
    for l in open(path):
        r = json.loads(l); key = (r["case"][:32], r["turn"]); t = per_turn[key]   # h045_chatlog_loops.txt prints case ids cut at 32 chars
        t["n"] += 1; t["by_model"].setdefault(label, [0, 0]); t["by_model"][label][1] += 1
        if not r["has_think_end"]:
            t["noans"] += 1; t["by_model"][label][0] += 1
            rr = tail_ratio(r["text"])
            if rr < 0.5: t["loops"] += 1
            elif label.startswith("Olmo 3 7B Think-SFT") and len(sft_long) < 5: sft_long.append((key, rr, r["text"][-350:]))
labels = [s.split("=", 1)[0] for s in specs]
lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by h045_crossmodel.py on " + time.strftime("%Y-%m-%d %H:%M %Z"), "",
         f"{'case':32s} {'turn':>4s} | {'H045':10s} {'H045 tail':>9s} | " + " | ".join(f"{l[:14]:>14s}" for l in labels) + f" | {'all 32: no answer':>17s} {'of which loops':>14s}"]
zero_on_loop_turns = 0; loop_turns = 0; zero_rows = []
for key in sorted(per_turn):
    t = per_turn[key]; st, hr = h045.get(key, ("?", float("nan")))
    cells = " | ".join(f"{t['by_model'][l][0]:>3d}/{t['by_model'][l][1]:<3d}     " if l in t["by_model"] else f"{'-':>14s}" for l in labels)
    lines.append(f"{key[0][:32]:32s} {key[1]:4d} | {st:10s} {hr:9.2f} | {cells} | {t['noans']:>7d}/{t['n']:<9d} {t['loops']:>14d}")
    if st == "UNFINISHED" and hr < 0.5:
        loop_turns += 1
        if t["noans"] == 0: zero_on_loop_turns += 1; zero_rows.append(key)
lines.append(f"\nH045 looped (tail ratio < 0.5) on {loop_turns} turns; on {zero_on_loop_turns} of them no Olmo 3 sample (0 of 32) exceeded 8,192 without closing: " + ", ".join(f"{c} t{t}" for c, t in zero_rows))
lines.append(f"turns where at least one Olmo 3 sample failed: {sum(1 for t in per_turn.values() if t['noans'])} of {len(per_turn)}; Olmo 3 loops in total: {sum(t['loops'] for t in per_turn.values())} of {sum(t['n'] for t in per_turn.values())} samples")
lines.append("\nTails (last 350 chars) of Olmo 3 7B Think-SFT no-answer samples that are not loops:")
for key, rr, tail in sft_long:
    lines.append(f"--- {key[0]} turn {key[1]} (tail ratio {rr:.2f}): ..." + tail.replace("\n", " / "))
txt = "\n".join(lines) + "\n"; open(out, "w").write(txt); print(txt)
