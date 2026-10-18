"""Judge spot-check for #1964: the benign prompts judged as refusals (XSTest safe types, WJ benign), grouped by prompt
over the 9 runs, with the start of one answer each, so a person can check WildGuard's labels."""
import json, collections
ST = ["Olmo-3-7B-Think-SFT", "Olmo-3-7B-Think-DPO", "Olmo-3-7B-Think"]; RU = ["run1", "run2", "run3"]
X, W = "xstest:wildguard_reasoning_answer", "wildjailbreak:benign_wildguard_reasoning_answer"
def rows(s, r, t): return next(iter(json.load(open(f"outputs/{s}/{r}/all.json.{t}")).values()))
cnt = collections.Counter(); ex = {}
for s in ST:
    for r in RU:
        for x in rows(s, r, X):
            if "contrast" in x["type"]: continue
            a = x.get("response") or ""
            if x.get("response_refusal") != "compliance" or a == "":
                k = (x["id"], x["type"], x["prompt"]); cnt[k] += 1; ex.setdefault(k, []).append((s, r, a))
print(f"XSTest safe prompts refused at least once: {len(cnt)} of 250; refusals over the 9 runs: {sum(cnt.values())}")
print("by type:", dict(collections.Counter(k[1] for k in cnt.elements())))
for k, c in cnt.most_common():
    s, r, a = ex[k][0]
    print(f"\n[{c}/9] {k[1]} | {k[2][:110]!r}\n    e.g. {s} {r} ({len(a)} chars): {a.strip()[:300]!r}")
print("\n== WJ benign: misses by prompt over the 9 runs")
cw = collections.Counter(); why = {}
for s in ST:
    for r in RU:
        for x in rows(s, r, W):
            a = x.get("response") or ""
            if a == "" or x.get("response_refusal") != "compliance":
                cw[x["id"]] += 1; why.setdefault(x["id"], []).append(f"{s.split('-')[-1]}/{r}:{'empty' if a == '' else x.get('response_refusal')}")
for i, c in cw.most_common():
    print(f"  {i}: {c}/9 | {why[i]}")
