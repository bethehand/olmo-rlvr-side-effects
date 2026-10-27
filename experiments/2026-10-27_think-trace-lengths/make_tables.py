"""Markdown tables for the README and the #2080 comment, built from the results files so no number is typed by hand.
Usage: make_tables.py RESULTS_DIR   (reads baseline_8k.json, trace_lengths.json, h045_chatlog_loops.txt)
Both datasets get a dataset-level row re-weighted by the dataset card's per-source prompt counts: the Dolci-Think-SFT-7B shards are
sorted by source (one sampled shard per source), and the Clean-v1 sample adds two shards from the Wildchat block that sits at the tail
of the shuffled file, which would otherwise over-weight Wildchat. The unweighted row is kept for reference."""
import json, math, os, sys
from composition_check import CARDS, match

R = sys.argv[1]
base = {(r["set"], r["file"]): r for r in json.load(open(os.path.join(R, "baseline_8k.json")))}
tl = json.load(open(os.path.join(R, "trace_lengths.json")))


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pc(k, n):
    lo, hi = wilson(k, n); return f"{100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


print("## Table B: previous generation at an 8,192-token budget (805 AlpacaEval prompts unless noted)\n")
print("| model | n | median total tokens | over 8,192 tokens | no `</think>` in the first 8,192 | hit the 32K cap | distinct 8-gram ratio, capped outputs (n) | same, finished outputs |")
print("|---|---|---|---|---|---|---|---|")
want = [("alpaca805", "Olmo-3-7B-Think-SFT@main_tmplA", "Olmo 3 7B Think-SFT, training template"), ("alpaca805", "Olmo-3-7B-Think-SFT@main_run1", "Olmo 3 7B Think-SFT, released template"),
        ("alpaca805", "Olmo-3-7B-Think-SFT@main_tmplA2", "Olmo 3 7B Think-SFT, training template, generation seed 2"),
        ("alpaca805", "Olmo-3-7B-Think-DPO@main_run1", "Olmo 3 7B Think-DPO"), ("alpaca805", "Olmo-3-7B-Think@main_run1", "Olmo 3 7B Think (RL)"),
        ("alpaca805", "Olmo-3-32B-Think-SFT@main_run1", "Olmo 3 32B Think-SFT"), ("alpaca805", "Olmo-3-32B-Think-DPO@main_run1", "Olmo 3 32B Think-DPO"),
        ("alpaca805", "Olmo-3-32B-Think@main_run1", "Olmo 3 32B Think"), ("alpaca805", "Olmo-3.1-32B-Think@main_run1", "Olmo 3.1 32B Think"),
        ("alpaca805", "Olmo-3.1-32B-Instruct-SFT@main_run1", "Olmo 3.1 32B Instruct-SFT (no thinking)"),
        ("ifeval", "Olmo-3-7B-Think-SFT@main_tmplA", "7B Think-SFT on IFEval"), ("gsm8k", "Olmo-3-7B-Think-SFT@main_tmplA", "7B Think-SFT on GSM8K"),
        ("heldout", "Olmo-3-7B-Think-SFT@main_tmplA", "7B Think-SFT on the 150-prompt held-out set")]
for s, f, lab in want:
    r = base[(s, f)]; n = r["n"]; lc, lf = r["loop_capped"], r["loop_finished"]
    dc = f"{lc['distinct_8gram_median']:.2f} ({lc['n']})" if lc["n"] else "– (0)"
    print(f"| {lab} | {n} | {r['median_total_tokens']:.0f} | {pc(r['over_8192'], n)} | {pc(r['no_answer_at_8k'], n)} | {pc(r['cap_hit_32k'], n)} | {dc} | {lf['distinct_8gram_median']:.2f} |")

for ds, title, card in (("clean_v1", "Dolci-Think-SFT-Clean-v1 (16 evenly spaced shuffled shards + 2 shards from the Wildchat block at the tail; headline = re-weighted row)", "32b"), ("think_sft_7b", "Dolci-Think-SFT-7B (shards sorted by source, one shard per source; headline = re-weighted row)", "7b")):
    if ds not in tl: continue
    o, srcs = tl[ds]["overall"], tl[ds]["by_source"]
    print(f"\n## Table A ({ds}): {title}\n")
    print("| source | rows sampled | thinking median / p90 | answer median / p90 | completion median / p90 | > 4,096 | > 8,192 | > 16,384 | thinking > 8,192 | no `</think>` |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for s in sorted(srcs, key=lambda x: -x["n"]):
        n = s["n"]; rows.append(s)
        print(f"| {s['label'].strip()} | {n} | {s['think_median']:.0f} / {s['think_p90']:.0f} | {s['answer_median']:.0f} / {s['answer_p90']:.0f} | {s['completion_median']:.0f} / {s['completion_p90']:.0f} | {100*s['over_4096']/n:.1f}% | {100*s['over_8192']/n:.1f}% | {100*s['over_16384']/n:.1f}% | {100*s['think_over_8192']/n:.1f}% | {100*s['no_think_close']/n:.1f}% |")
    n = o["n"]
    print(f"| **all sampled rows (unweighted)** | {n} | {o['think_median']:.0f} / {o['think_p90']:.0f} | {o['answer_median']:.0f} / {o['answer_p90']:.0f} | {o['completion_median']:.0f} / {o['completion_p90']:.0f} | {100*o['over_4096']/n:.1f}% | {100*o['over_8192']/n:.1f}% | {100*o['over_16384']/n:.1f}% | {100*o['think_over_8192']/n:.1f}% | {100*o['no_think_close']/n:.1f}% |")
    # re-weighted dataset-level shares by card counts (sources we sampled only)
    total, labels = CARDS[card]
    # group sampled sources under their card label (e.g. OpenThoughts3 math / STEM / code -> one card count); inside a group the
    # sub-sources are weighted by their sampled rows, across groups by the card's prompt counts
    groups = {}
    for s in rows:
        k = match(s["label"].strip())
        if k and k in labels: groups.setdefault(k, []).append(s)
    covered = sum(labels[k][1] for k in groups)
    def rw(key):
        out = 0.0
        for k, subs in groups.items():
            n_k = sum(x["n"] for x in subs)
            share_k = sum((x["n"] / n_k) * x[key] / x["n"] for x in subs)
            out += (labels[k][1] / covered) * share_k
        return out
    print(f"| **re-weighted by card source counts** ({100*covered/total:.1f}% of rows covered) | – | – | – | – | {100*rw('over_4096'):.1f}% | {100*rw('over_8192'):.1f}% | {100*rw('over_16384'):.1f}% | {100*rw('think_over_8192'):.1f}% | {100*rw('no_think_close'):.1f}% |")
    print(f"\n(card labels covered: {', '.join(labels[k][0] for k in groups)}; not sampled: {', '.join(labels[k][0] for k in labels if k not in groups) or 'none'})")
    print(f"\nmean prompt+completion tokens per row: {o['mean_prompt_plus_completion_tokens']:.0f}; multi-turn rows: {100*o['multi_assistant']/n:.1f}%; rows with some assistant turn lacking `</think>`: {100*o['any_assistant_without_think_close']/n:.1f}%")

print("\n## H045 loop check (from results/h045_chatlog_loops.txt)\n")
for l in open(os.path.join(R, "h045_chatlog_loops.txt")):
    if l.startswith(("UNFINISHED:", "answered:", "   turns with")): print(l.rstrip())
