"""Where in each unfinished H045 trace does the repetition start? Sliding window of 500 words, step 50, over the reasoning text;
the onset is the first window whose distinct 8-gram ratio drops below 0.5 and stays below it for the rest of the trace (allowing one
recovery window). Reports onset in words, as a share of the trace, and in tokens estimated as onset_words * 8192 / total_words
(the trace is exactly 8,192 tokens long). Usage: h045_loop_onset.py chat_logs.md OUT.txt"""
import re, sys, time, statistics

text = open(sys.argv[1], encoding="utf-8").read()
pat = re.compile(r"## (\S+) — [^\n]*\n(.*?)(?=\n## \S+ — |\Z)", re.S)
turn_pat = re.compile(r"### Assistant turn (\d+): (answered|UNFINISHED[^\n]*)\n\nfinish_reason `(\w+)` · completion tokens (\d+)/(\d+)[^\n]*\n\n<details><summary>Reasoning \((\d+) chars\)</summary>\n\n````text\n(.*?)\n````\n</details>", re.S)


def ratio(ws, k=8):
    grams = [tuple(ws[i:i + k]) for i in range(max(0, len(ws) - k + 1))]
    return len(set(grams)) / len(grams) if grams else 1.0


rows = []
for case, body in pat.findall(text):
    for turn, status, finish, ctoks, budget, rchars, reasoning in turn_pat.findall(body):
        if not status.startswith("UNFINISHED"):
            continue
        words = reasoning.split(); n = len(words)
        win, step = 500, 50
        ratios = [(i, ratio(words[i:i + win])) for i in range(0, max(1, n - win + 1), step)]
        onset = None
        for j, (i, r) in enumerate(ratios):
            if r < 0.5:
                rest = [rr for _, rr in ratios[j:]]
                if sum(rr >= 0.5 for rr in rest) <= 1:
                    onset = i; break
        rows.append((case, int(turn), n, onset, ratio(words[-2000:])))
lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by h045_loop_onset.py on " + time.strftime("%Y-%m-%d %H:%M %Z"), "",
         f"{'case':32s} {'turn':>4s} {'words':>6s} {'onset word':>10s} {'onset share':>11s} {'onset ~tokens':>13s} {'tail ratio':>10s}"]
onsets = []
for case, turn, n, onset, tr in rows:
    if onset is None:
        lines.append(f"{case[:32]:32s} {turn:4d} {n:6d} {'none':>10s} {'-':>11s} {'-':>13s} {tr:10.2f}")
    else:
        est = onset * 8192 / n; onsets.append((onset, onset / n, est))
        lines.append(f"{case[:32]:32s} {turn:4d} {n:6d} {onset:10d} {100*onset/n:10.0f}% {est:13.0f} {tr:10.2f}")
if onsets:
    lines.append(f"\nturns with a detected onset: {len(onsets)} of {len(rows)}; median onset {statistics.median(o for o,_,_ in onsets):.0f} words = {100*statistics.median(s for _,s,_ in onsets):.0f}% of the trace ≈ {statistics.median(e for _,_,e in onsets):.0f} tokens; "
                 f"earliest ≈ {min(e for _,_,e in onsets):.0f} tokens, latest ≈ {max(e for _,_,e in onsets):.0f} tokens")
out = "\n".join(lines) + "\n"; open(sys.argv[2], "w").write(out); print(out)
