"""Pull every URL out of the answers in the AlpacaEval length outputs (../2026-10-18_length).
The answer is the text after the last </think> (the whole output when there is none), as in OLMES r1_style.
Usage: extract_urls.py OUTPUTS_DIR OUT.json   ->   {run: [[index, finished, [urls]], ...]}"""
import glob, json, os, re, sys
# a URL may contain balanced parentheses (Wikipedia titles, Javadoc anchors); a lone ")" ends it, as in markdown links
url_re = re.compile(r"https?://(?:[^\s()\[\]<>\"'`*]|\([^\s()<>\"'`*]*\))+")
src, out = sys.argv[1], sys.argv[2]
res = {}
for f in sorted(glob.glob(os.path.join(src, "*@main_run*.jsonl")) + glob.glob(os.path.join(src, "*@main_ctrl1.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows = []
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        resp = r["response"]
        fin = r["finish_reason"] == "stop" and (r["has_think_end"] or "Think" not in name)
        ans = resp.rpartition("</think>")[2] if "</think>" in resp else resp
        rows.append([r["index"], fin, [u.rstrip(".,;:!?") for u in url_re.findall(ans)]])
    res[name] = rows
json.dump(res, open(out, "w"))
print(len(res), "files", sum(len(v) for v in res.values()), "rows")
