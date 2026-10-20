"""Pull every URL out of the probe answers (outputs/*_p1.jsonl) and the rel1 AlpacaEval runs (outputs/*_rel1.jsonl),
in the format ../2026-10-19_olmo-voice/check_urls.py reads: {run: [[key, finished, [urls]], ...]}.
Usage: extract_urls_probe.py OUTPUTS_DIR OUT.json"""
import glob, json, os, re, sys
url_re = re.compile(r"https?://(?:[^\s()\[\]<>\"'`*]|\([^\s()<>\"'`*]*\))+")
src, out = sys.argv[1], sys.argv[2]
res = {}
for f in sorted(glob.glob(os.path.join(src, "*_p1.jsonl")) + glob.glob(os.path.join(src, "*_rel1.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows = []
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        resp = r["response"]
        ans = resp.rpartition("</think>")[2] if "</think>" in resp else resp
        fin = r["finish_reason"] == "stop" and (r.get("has_think_end") or "Think" not in name)
        key = f"{r['id']}/{r['condition']}/s{r['sample']}" if "id" in r else r["index"]
        rows.append([key, fin, [u.rstrip(".,;:!?") for u in url_re.findall(ans)]])
    res[name] = rows
json.dump(res, open(out, "w"))
print(len(res), "files", sum(len(v) for v in res.values()), "rows", len({u for v in res.values() for _, _, us in v for u in us}), "unique URLs")
