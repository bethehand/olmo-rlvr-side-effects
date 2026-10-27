"""Loop signature of the Olmo 3.5 SFT H045 turns in the chat_logs.md attached to open-instruct #2080 (exported from Ai2's qualeval,
max_tokens 8192). For every assistant turn: status (answered / UNFINISHED), completion tokens, reasoning length, and two repetition
measures computed on the reasoning text: the distinct 8-gram ratio over the last 2,000 words (1.0 = no repeated 8-word windows,
near 0 = the tail is one repeated passage) and the number of words starting with "wait" per 1,000 words; plus the most repeated
8-gram and how many times it occurs. The same measures on our Olmo 3 outputs: finished thinking 0.98-1.00, 32K-cap loops 0.00-0.03.
Usage: h045_chatlog_loops.py chat_logs.md OUT.txt"""
import re, sys, statistics, time
from collections import Counter

text = open(sys.argv[1], encoding="utf-8").read()
convs = re.split(r"\n## (?=\S)", text)
rows = []
for conv in convs[1:]:
    case = conv.split("\n", 1)[0].split(" — ")[0].strip()
    for m in re.finditer(r"### Assistant turn (\d+): (answered|UNFINISHED[^\n]*)\n\nfinish_reason `(\w+)` · completion tokens (\d+)/(\d+)[^\n]*think block closed: (\w+)\n\n<details><summary>Reasoning \((\d+) chars\)</summary>\n\n````text\n(.*?)\n````\n</details>", conv, re.S):
        turn, status, finish, ctoks, budget, closed, rchars, reasoning = m.groups()
        words = reasoning.split()
        tail = words[-2000:]
        grams = [tuple(tail[i:i + 8]) for i in range(max(0, len(tail) - 7))]
        distinct_tail = len(set(grams)) / len(grams) if grams else float("nan")
        allgrams = Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
        distinct_all = len(allgrams) / sum(allgrams.values()) if allgrams else float("nan")
        top, topn = (allgrams.most_common(1)[0] if allgrams else ((), 0))
        waits = sum(1 for w in words if w.lower().startswith("wait")) / max(1, len(words)) * 1000
        rows.append(dict(case=case, turn=int(turn), status="UNFINISHED" if status.startswith("UNFINISHED") else "answered", finish=finish,
                         completion_tokens=int(ctoks), budget=int(budget), closed=closed, reasoning_chars=int(rchars), words=len(words),
                         distinct8_tail=distinct_tail, distinct8_all=distinct_all, wait_per_1k=waits, top_gram=" ".join(top), top_count=topn))

lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by h045_chatlog_loops.py on " + time.strftime("%Y-%m-%d %H:%M %Z"),
         f"# turns parsed: {len(rows)} (UNFINISHED {sum(r['status']=='UNFINISHED' for r in rows)}, answered {sum(r['status']=='answered' for r in rows)})", "",
         f"{'case':32s} {'turn':>4s} {'status':10s} {'tokens':>9s} {'words':>6s} {'distinct8 tail':>14s} {'distinct8 all':>13s} {'wait/1k':>7s} {'top 8-gram count':>16s}  most repeated 8-gram"]
for r in rows:
    lines.append(f"{r['case'][:32]:32s} {r['turn']:4d} {r['status']:10s} {r['completion_tokens']:4d}/{r['budget']:4d} {r['words']:6d} {r['distinct8_tail']:14.2f} {r['distinct8_all']:13.2f} {r['wait_per_1k']:7.1f} {r['top_count']:16d}  {r['top_gram'][:70]}")
for st in ("UNFINISHED", "answered"):
    g = [r for r in rows if r["status"] == st]
    if g:
        med = lambda k: statistics.median(r[k] for r in g)
        lines.append(f"\n{st}: n={len(g)} | median distinct8 tail {med('distinct8_tail'):.2f} (min {min(r['distinct8_tail'] for r in g):.2f}, max {max(r['distinct8_tail'] for r in g):.2f}) | median distinct8 all {med('distinct8_all'):.2f} | median wait/1k {med('wait_per_1k'):.1f} | median words {med('words'):.0f}")
        lines.append(f"   turns with distinct8 tail < 0.5 (looping tail): {sum(r['distinct8_tail'] < 0.5 for r in g)} of {len(g)}; < 0.9: {sum(r['distinct8_tail'] < 0.9 for r in g)}")
out = "\n".join(lines) + "\n"
open(sys.argv[2], "w").write(out); print(out)
