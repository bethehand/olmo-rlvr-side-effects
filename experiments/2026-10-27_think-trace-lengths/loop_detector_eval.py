"""Would a repetition-aware stop have caught H045's loops, and what would it cost on healthy traces?
Detector: slide a 500-word window over the reasoning text in steps of 50 words; fire at the first position where the distinct 8-gram
ratio of the window is below 0.5 for three consecutive windows. Evaluated on: (a) the 16 unfinished H045 turns (recall and where it fires,
in words and in estimated tokens = words * total_tokens / total_words); (b) healthy traces: the 6 answered H045 turns and every Olmo 3
sample that closed its thinking in the 8K and 32K reruns (false positives, with the position relative to the trace); (c) Olmo 3 samples
still open at the cap (recall on the previous generation's loops). Also reports the share of the 8,192-token H045 turns that would be saved.
Usage: loop_detector_eval.py OUT.txt chat_logs.md FILE.jsonl [FILE.jsonl ...]"""
import json, re, statistics, sys, time

WIN, STEP, THR, CONSEC = 500, 50, 0.5, 3


def ratio(ws, k=8):
    g = [tuple(ws[i:i + k]) for i in range(max(0, len(ws) - k + 1))]
    return len(set(g)) / len(g) if g else 1.0


def fire_position(words):
    """First window start where the ratio stays below THR for CONSEC consecutive windows; None if it never fires."""
    run = 0
    for i in range(0, max(1, len(words) - WIN + 1), STEP):
        if ratio(words[i:i + WIN]) < THR:
            run += 1
            if run >= CONSEC:
                return i + WIN  # the detector has seen the full third window at this point
        else:
            run = 0
    return None


out, logs, files = sys.argv[1], sys.argv[2], sys.argv[3:]
lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by loop_detector_eval.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
text = open(logs, encoding="utf-8").read()
turn_pat = re.compile(r"### Assistant turn (\d+): (answered|UNFINISHED[^\n]*)\n\nfinish_reason `(\w+)` · completion tokens (\d+)/(\d+)[^\n]*\n\n<details><summary>Reasoning \((\d+) chars\)</summary>\n\n````text\n(.*?)\n````\n</details>", re.S)
unf, ans = [], []
for case, body in re.findall(r"\n## (\S+) — [^\n]*\n(.*?)(?=\n## \S+ — |\Z)", text, re.S):
    for turn, status, finish, ctoks, budget, rchars, reasoning in turn_pat.findall(body):
        words = reasoning.split(); pos = fire_position(words)
        rec = (case, int(turn), len(words), int(ctoks), pos)
        (unf if status.startswith("UNFINISHED") else ans).append(rec)
caught = [r for r in unf if r[4] is not None]
lines.append(f"\n== H045 unfinished turns: detector fires on {len(caught)}/{len(unf)}")
saved = []
for case, turn, nw, nt, pos in unf:
    if pos is None:
        lines.append(f"  {case[:32]:32s} t{turn}  no fire  ({nw} words)")
    else:
        est = pos * nt / nw; saved.append(1 - est / nt)
        lines.append(f"  {case[:32]:32s} t{turn}  fires at word {pos:5d} of {nw} = ~{est:5.0f} tokens; would save {100*(1-est/nt):.0f}% of the turn")
if saved:
    lines.append(f"  median fire position ~{statistics.median(pos * nt / nw for _, _, nw, nt, pos in caught):.0f} tokens; median saving {100*statistics.median(saved):.0f}% of the 8,192-token turn")
fp_ans = [r for r in ans if r[4] is not None]
lines.append(f"\n== H045 answered turns: false positives {len(fp_ans)}/{len(ans)}")
tot_h, tot_fp, tot_open, tot_open_caught = 0, 0, 0, 0
for path in files:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    healthy = [r for r in rows if r["has_think_end"]]; open_ = [r for r in rows if not r["has_think_end"]]
    fps = []
    for r in healthy:
        think = r["text"].split("</think>")[0].split(); pos = fire_position(think)
        if pos is not None: fps.append((r["case"], r["turn"], r["sample"], pos, len(think)))
    oc = sum(fire_position(r["text"].split()) is not None for r in open_)
    tot_h += len(healthy); tot_fp += len(fps); tot_open += len(open_); tot_open_caught += oc
    lines.append(f"\n== {path.split('/')[-1]}: healthy (closed) samples {len(healthy)}, false positives {len(fps)} | open at cap {len(open_)}, detector fires on {oc}")
    for c, t, smp, pos, nw in fps[:6]:
        lines.append(f"    FP: {c[:30]} t{t} s{smp} fires at word {pos} of {nw}")
lines.append(f"\nTOTAL over the Olmo 3 reruns: false positives {tot_fp}/{tot_h} closed traces ({100*tot_fp/max(1,tot_h):.2f}%); fires on {tot_open_caught}/{tot_open} traces that were still open at the cap")
txt = "\n".join(lines) + "\n"; open(out, "w").write(txt); print(txt)
