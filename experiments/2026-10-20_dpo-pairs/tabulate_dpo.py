"""Markdown tables from analyze_dpo.py's summary_*.json. Usage: tabulate_dpo.py results/"""
import json, sys, glob, os
out = sys.argv[1] if len(sys.argv) > 1 else "results"
S = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{out}/summary_*.json")) if "summary_all" not in f}
NAMES = {"think32b": "Dolci-Think-DPO-32B (Olmo 3 / 3.1 32B Think)", "think7b": "Dolci-Think-DPO-7B (Olmo 3 7B Think)", "instruct": "Dolci-Instruct-DPO (3.1 32B Instruct, 7B Instruct)"}
def row(name, a):
    return (f"| {name} | {a['n']:,} | {a['chosen_chars_median']:,.0f} | {a['rejected_chars_median']:,.0f} | {a['pair_ratio_median']} [{a['pair_ratio_q25']}, {a['pair_ratio_q75']}] | {a['chosen_longer_pct']}% |"
            + (f" {a['chosen_tokens_median']:,.0f} / {a['rejected_tokens_median']:,.0f} |" if "chosen_tokens_median" in a else " |"))
print("**Answer length, chosen vs rejected** (answer = text after `</think>`; per-pair ratio = chosen/rejected; median [IQR]).\n")
print("| data | pairs | chosen median chars | rejected median chars | per-pair ratio | chosen longer | median tokens chosen / rejected |\n|---|---|---|---|---|---|---|")
for k, s in S.items():
    print(row(NAMES.get(k, k), s["all"]))
    for pt, a in s["by_preference_type"].items():
        if len(s["by_preference_type"]) > 1 and a: print(row(f"&nbsp;&nbsp;{pt}", a))
print("\n**Thinking length (Think sets)**: median chars of the `<think>` block, chosen vs rejected; share of pairs where the whole message (thinking + answer) is longer for chosen.\n")
print("| data | chosen think median | rejected think median | whole-message ratio median | chosen whole message longer |\n|---|---|---|---|---|")
for k, s in S.items():
    a = s["all"]
    if "think_chars_median_chosen" in a:
        print(f"| {NAMES.get(k, k)} | {a['think_chars_median_chosen']:,.0f} | {a['think_chars_median_rejected']:,.0f} | {a['full_pair_ratio_median']} | {a['chosen_longer_full_pct']}% |")
print("\n**Formatting marks in the answer** (mean count per answer, chosen → rejected; share of pairs where chosen has more / fewer).\n")
marks = ["bullets", "bold", "headers", "table_rows", "urls", "olmo_ai2", "emoji"]
print("| data | " + " | ".join(marks) + " |\n|---|" + "---|" * len(marks))
for k, s in S.items():
    a = s["all"]
    print(f"| {NAMES.get(k, k)} | " + " | ".join(f"{a[m]['chosen_mean']} vs {a[m]['rejected_mean']} ({a[m]['chosen_more_pct']}% / {a[m]['rejected_more_pct']}%)" for m in marks) + " |")
print("\n**Answers containing a URL / an Olmo or Ai2 mention** (% of answers, chosen vs rejected).\n")
print("| data | URL chosen | URL rejected | Olmo/Ai2 chosen | Olmo/Ai2 rejected |\n|---|---|---|---|---|")
for k, s in S.items():
    a = s["all"]
    print(f"| {NAMES.get(k, k)} | {a['urls']['chosen_any_pct']}% | {a['urls']['rejected_any_pct']}% | {a['olmo_ai2']['chosen_any_pct']}% | {a['olmo_ai2']['rejected_any_pct']}% |")
print("\n**By source (Think sets) / model pair** — top entries.\n")
for k, s in S.items():
    print(f"\n*{NAMES.get(k, k)}*, by source:\n\n| source | pairs | per-pair ratio | chosen longer |\n|---|---|---|---|")
    for src, a in s["by_source"].items():
        if a: print(f"| {src} | {a['n']:,} | {a['pair_ratio_median']} | {a['chosen_longer_pct']}% |")
    print(f"\nmodel pairs:\n\n| chosen model | rejected model | pairs | per-pair ratio | chosen longer |\n|---|---|---|---|---|")
    for m in s["model_pairs_top"][:8]:
        print(f"| {m['chosen_model']} | {m['rejected_model']} | {m['n']:,} | {m['pair_ratio_median']} | {m['chosen_longer_pct']}% |")
