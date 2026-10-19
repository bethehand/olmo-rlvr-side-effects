"""Counts quoted in the README for the identity rows of the public Dolci SFT sets, from the rows that
sparse_col_scan.py saves (Dolci-Think-SFT-32B: source "Olmo Identity Hardcoded Data"; Dolci-Instruct-SFT:
source_dataset "Hardcoded Data"). Counts are over the final answers (text after the last </think>).
Usage: identity_stats.py think32b_identity.json instruct_identity.json"""
import collections, re, sys
import json
URL = re.compile(r"https?://[^\s)\]>\"'`*]+")
PATS = [("olmo-7b, any form", r"olmo-7b"), ("allenai/olmo-7b, full id", r"allenai/olmo-7b"), ("7B/13B/30B (sizes and model ids)", r"\b(7|13|30)B\b"), ("December 2024", r"December 2024"),
        ("Tulu / Tülu", r"T[üu]lu"), ("allennlp", r"allennlp")]
for path in sys.argv[1:]:
    rows = json.load(open(path))
    sysp = collections.Counter(next((m["content"][:90] for m in r["messages"] if m["role"] == "system"), "(no system turn)") for r in rows)
    ans = [next(m["content"] for m in r["messages"] if m["role"] == "assistant").split("</think>")[-1] for r in rows]
    print(f"== {path}: {len(rows)} rows")
    for s, n in sysp.most_common(): print(f"   system prompt x{n}: {s}")
    us = collections.Counter(u.rstrip(".,;:!?") for a in ans for u in URL.findall(a))
    print(f"   URLs in answers: {sum(us.values())} ({len(us)} distinct)")
    for u, n in us.most_common(): print(f"      {n:3d} {u}")
    for label, p in PATS:
        per = [len(re.findall(p, a, re.I)) for a in ans]
        print(f"   {label}: {sum(per)} times in {sum(1 for x in per if x)} rows")
