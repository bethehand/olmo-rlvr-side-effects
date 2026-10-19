"""Find answers that mention Olmo, Ai2, an Ai2 link or the cutoff date, as candidates for hand labeling.
The regexes only find candidates: every hit was then read and labeled by hand (selftalk_labels.csv).
Usage: find_selftalk.py OUTPUTS_DIR OUT.json"""
import collections, glob, json, os, re, sys
PATS = {
    "dec2024": re.compile(r"December 2024", re.I),
    "nov2024": re.compile(r"November 2024", re.I),
    "selfid": re.compile(r"\b(as olmo|i am olmo|i.m olmo|olmo,? (an?|the) |allen institute|\bai2\b)", re.I),
    "ai2link": re.compile(r"huggingface\.co/allenai|allenai\.org|allennlp\.org", re.I),
    "olmo": re.compile(r"\bolmo\b|allenai|allen ai", re.I),
    "cutoff": re.compile(r"(knowledge|date|training)[ -]?(data )?cut-?off|my (training|knowledge)[^.]{0,40}(up to|until|through)", re.I),
}
src, out_path = sys.argv[1], sys.argv[2]
out = {"hits": [], "meta": {}}
for f in sorted(glob.glob(os.path.join(src, "*@main_run*.jsonl")) + glob.glob(os.path.join(src, "*@main_ctrl1.jsonl"))):
    run = os.path.basename(f)[:-6]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    meta = collections.Counter((str(r.get("template_from")), str(r.get("seed"))) for r in rows)
    out["meta"][run] = {"n": len(rows), "model": rows[0]["model"], "template_from, seed, rows": [list(k) + [v] for k, v in meta.items()]}
    for r in rows:
        resp = r["response"]
        ans = resp.rpartition("</think>")[2] if "</think>" in resp else resp
        flags = {k: [m.span() for m in p.finditer(ans)] for k, p in PATS.items()}
        if any(flags[k] for k in ("dec2024", "nov2024", "selfid", "ai2link", "olmo")):
            ctx = [[k, ans[max(0, s - 160):e + 120].replace("\n", " ")] for k, spans in flags.items() for s, e in spans[:3]]
            out["hits"].append({"run": run, "index": r["index"], "instruction": r["instruction"][:300],
                                "flags": {k: len(v) for k, v in flags.items()}, "ctx": ctx})
json.dump(out, open(out_path, "w"))
print(len(out["meta"]), "files;", len(out["hits"]), "candidate answers")
for run, m in out["meta"].items():
    print(run, m)
