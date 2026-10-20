"""Clean re-check of the formatting means on a random sample (the inline regexes in verify_dpo.py were mangled by shell escaping)."""
import glob, os, random, statistics, re
import pyarrow.parquet as pq
random.seed(11)
cache = os.environ["HF_HUB_CACHE"]
bullets = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S"); bold = re.compile(r"\*\*[^*\n]+\*\*"); headers = re.compile(r"(?m)^\s*#{1,6}\s+\S")
emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿]"); urls = re.compile(r"https?://(?:[^\s()\[\]<>\"'`*]|\([^\s()<>\"'`*]*\))+")
for repo in ["Dolci-Think-DPO-32B", "Dolci-Instruct-DPO"]:
    files = sorted(glob.glob(cache + "/datasets--allenai--" + repo + "/snapshots/*/**/*.parquet", recursive=True))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["chosen", "rejected"]); rows += t.take(sorted(random.sample(range(t.num_rows), 1000))).to_pylist()
    def ans(msgs):
        c = [m for m in msgs if m["role"] == "assistant"][-1]["content"] or ""
        return c.split("</think>")[-1].strip() if "</think>" in c else c.strip()
    ca = [ans(r["chosen"]) for r in rows]; ra = [ans(r["rejected"]) for r in rows]
    print(f"== {repo}: {len(rows)} pairs")
    for name, pat in [("bullets", bullets), ("bold", bold), ("headers", headers), ("emoji", emoji), ("urls", urls)]:
        c = [len(pat.findall(a)) for a in ca]; r = [len(pat.findall(b)) for b in ra]
        print(f"   {name:8} mean chosen {statistics.mean(c):.2f} vs rejected {statistics.mean(r):.2f}; any: {100*sum(x>0 for x in c)/len(c):.1f}% vs {100*sum(x>0 for x in r)/len(r):.1f}%")
    # what an emoji-bearing chosen answer looks like
    k = next((i for i, a in enumerate(ca) if emoji.search(a)), None)
    if k is not None: print("   example chosen answer with emoji:", repr(ca[k][:200]))
