"""Independent re-check of analyze_dpo.py's headline numbers: a fresh random sample of raw pairs, lengths computed by a
different route (str.split on </think>), plus a look at edge cases and a few pairs read in full."""
import glob, os, random, statistics, re, json
import pyarrow.parquet as pq
random.seed(7)
cache = os.environ["HF_HUB_CACHE"]
olmo_re = re.compile(r"\bOLMo\b|\bOlmo\b|\bAi2\b|\bAI2\b|Allen Institute")
assert olmo_re.search("I am Olmo, built by Ai2.") and not olmo_re.search("holmography")
for repo in ["Dolci-Think-DPO-32B", "Dolci-Think-DPO-7B", "Dolci-Instruct-DPO"]:
    files = sorted(glob.glob(cache + "/datasets--allenai--" + repo + "/snapshots/*/**/*.parquet", recursive=True))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["chosen", "rejected"])
        idx = sorted(random.sample(range(t.num_rows), 1200))
        rows += t.take(idx).to_pylist()
    def ans(msgs):
        c = [m for m in msgs if m["role"] == "assistant"][-1]["content"] or ""
        return c.split("</think>")[-1].strip() if "</think>" in c else c.strip()
    ca = [ans(r["chosen"]) for r in rows]; ra = [ans(r["rejected"]) for r in rows]
    ratios = sorted((len(a) + 1) / (len(b) + 1) for a, b in zip(ca, ra))
    print(f"== {repo}: sample {len(rows)} pairs")
    print(f"   chosen median {statistics.median(map(len, ca)):.0f} chars, rejected {statistics.median(map(len, ra)):.0f}; per-pair ratio median {statistics.median(ratios):.3f} "
          f"[{ratios[len(ratios)//4]:.3f}, {ratios[3*len(ratios)//4]:.3f}]; chosen longer {100*sum(len(a) > len(b) for a, b in zip(ca, ra))/len(rows):.1f}%")
    print(f"   empty answers: chosen {sum(len(a)==0 for a in ca)}, rejected {sum(len(b)==0 for b in ra)}; "
          f"chosen with </think>: {sum('</think>' in ([m for m in r['chosen'] if m['role']=='assistant'][-1]['content'] or '') for r in rows)}; "
          f"multiple assistant turns: {sum(sum(m['role']=='assistant' for m in r['chosen'])>1 for r in rows)}; "
          f"Olmo/Ai2 in any answer: {sum(bool(olmo_re.search(a)) for a in ca+ra)}")
    print(f"   bullets mean chosen {statistics.mean(len(re.findall(r'(?m)^\\s*(?:[-*•]|\\d+[.)])\\s+\\S', a)) for a in ca):.2f} vs rejected {statistics.mean(len(re.findall(r'(?m)^\\s*(?:[-*•]|\\d+[.)])\\s+\\S', b)) for b in ra):.2f}; "
          f"bold mean {statistics.mean(len(re.findall(r'\\*\\*[^*\\n]+\\*\\*', a)) for a in ca):.2f} vs {statistics.mean(len(re.findall(r'\\*\\*[^*\\n]+\\*\\*', b)) for b in ra):.2f}")
    # two pairs in full-ish, to see what "longer" looks like
    for k in random.sample(range(len(rows)), 2):
        u = [m for m in rows[k]["chosen"] if m["role"] == "user"][0]["content"]
        print(f"   --- pair {k}: user: {u[:140]!r}")
        print(f"       chosen ({len(ca[k])} chars): {ca[k][:260]!r}")
        print(f"       rejected ({len(ra[k])} chars): {ra[k][:260]!r}")
