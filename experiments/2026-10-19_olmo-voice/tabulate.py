"""Tables for the README: broken links per run (answer_urls.json + url_status.json) and the hand-labeled
self-talk counts (selftalk_labels.csv). Usage: python tabulate.py > results.txt"""
import collections, csv, json, urllib.parse
urls = json.load(open("answer_urls.json"))
st = json.load(open("url_status.json"))
ORDER = ["Olmo-3.1-32B-Instruct-SFT@main_run1", "Olmo-3.1-32B-Instruct-DPO@main_run1", "Olmo-3.1-32B-Instruct@main_ctrl1", "Olmo-3.1-32B-Instruct@main_run1",
         "Olmo-3-32B-Think-SFT@main_run1", "Olmo-3-32B-Think-DPO@main_run1", "Olmo-3-32B-Think@main_run1", "Olmo-3.1-32B-Think@main_ctrl1", "Olmo-3.1-32B-Think@main_run1",
         "Olmo-3-7B-Think-SFT@main_run1", "Olmo-3-7B-Think-SFT@main_ctrl1", "Olmo-3-7B-Think-DPO@main_run1", "Olmo-3-7B-Think-DPO@main_run2", "Olmo-3-7B-Think@main_run1"]

def bucket(s):
    if s in ("placeholder", "malformed"): return "placeholder"
    if s == "dns": return "dead"
    if s in ("404", "410", "hf-missing"): return "notfound"
    if s[:1] in "23": return "ok"
    return "unknown"

def is_ai2(u):
    p = urllib.parse.urlparse(u); h = (p.hostname or "").lower()
    return h.endswith(("allenai.org", "allennlp.org", "ai2.dev")) or (h in ("huggingface.co", "github.com") and p.path.lower().startswith("/allenai"))

def kind(u):
    p = urllib.parse.urlparse(u); h = (p.hostname or "").lower().removeprefix("www."); path = [x for x in p.path.split("/") if x]
    if bucket(st[u]) == "dead": return "domain does not exist"
    if h == "github.com": return "GitHub repo or file"
    if h in ("doi.org", "dx.doi.org"): return "DOI"
    if h.startswith("amazon.") or h == "amzn.to": return "Amazon listing"
    return "other page on a real site" if path else "site root"

print("== Broken links per run (counts are link occurrences in answers; broken % = (not found + dead) / (ok + not found + dead))")
print(f"{'run':38} {'answers w/ URL':>14} {'URLs':>5} {'ok':>4} {'notfound':>8} {'dead':>4} {'unknown':>7} {'placeholder':>11} {'broken%':>8} {'Ai2 URLs':>8}")
tot = collections.Counter()
for run in ORDER:
    rows = urls[run]
    us = [u for _, _, x in rows for u in x]
    b = collections.Counter(bucket(st[u]) for u in us); tot.update(b)
    chk = b["ok"] + b["notfound"] + b["dead"]
    print(f"{run:38} {sum(1 for _, _, x in rows if x):14d} {len(us):5d} {b['ok']:4d} {b['notfound']:8d} {b['dead']:4d} {b['unknown']:7d} {b['placeholder']:11d} "
          f"{100 * (b['notfound'] + b['dead']) / max(chk, 1):7.1f}% {sum(map(is_ai2, us)):8d}")
chk = tot["ok"] + tot["notfound"] + tot["dead"]
print(f"pooled: {sum(tot.values())} links, {dict(tot)}; broken {tot['notfound'] + tot['dead']}/{chk} = {100 * (tot['notfound'] + tot['dead']) / chk:.1f}%")
print("distinct URLs:", len(st), dict(collections.Counter(bucket(s) for s in st.values())))
broken = [u for r in urls.values() for _, _, x in r for u in x if bucket(st[u]) in ("notfound", "dead")]
print("broken links by type:", {k: f"{n} ({100 * n / len(broken):.0f}%)" for k, n in collections.Counter(map(kind, broken)).most_common()})
ai2 = collections.Counter((run, u, bucket(st[u])) for run in ORDER for _, _, x in urls[run] for u in x if is_ai2(u))
print("Ai2 URLs:", sum(ai2.values()))
for (run, u, b), n in sorted(ai2.items()):
    print(f"   {run:34} {n:3d} {b:9} {u}")

print("\n== Hand-labeled answers that mention Olmo, Ai2, an Ai2 link or a 2024 cutoff month (selftalk_labels.csv)")
lab = list(csv.DictReader(open("selftalk_labels.csv")))
c = collections.defaultdict(collections.Counter)
for r in lab:
    c[r["run"]][r["label"]] += 1
    if r["label"] == "unrelated-self" and r["ai2_urls"]: c[r["run"]]["unrelated-self with Ai2 URL"] += 1
cols = ["unrelated-self", "unrelated-self with Ai2 URL", "unrelated-cutoff", "related", "false-positive"]
print(f"{'run':38} " + " ".join(f"{k:>27}" for k in cols))
for run in ORDER:
    print(f"{run:38} " + " ".join(f"{c[run][k]:27d}" for k in cols))
