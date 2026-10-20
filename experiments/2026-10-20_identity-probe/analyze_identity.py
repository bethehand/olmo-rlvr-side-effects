"""Tabulate the identity probe outputs: per model x system-prompt condition, how often the answer (text after </think>)
names a model size, a model name/version, a maker, a cutoff date, a URL, or (for the unrelated prompts) Olmo/Ai2 at all.
Regex counts for a first look; the README's headline numbers are hand-checked from these candidates.
Usage: analyze_identity.py outputs/*.jsonl [--show CATEGORY]"""
import argparse, collections, json, re, sys

PATS = {
    "size":     re.compile(r"\b(\d{1,3}(?:\.\d)?)\s?-?\s?[Bb](?:illion)?\b(?:\s*param)?|\b\d{1,3}\s+billion\s+param", re.I),
    "olmo":     re.compile(r"\bOLMo\b|\bOlmo\b|\bolmo\b"),
    "ai2":      re.compile(r"\bAi2\b|\bAI2\b|Allen Institute", re.I),
    "robot":    re.compile(r"\bRobot\b|\bAcme\b"),
    "cutoff":   re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d\d\b|\bcut-?off\b|training data (?:is|goes|ends|extends)", re.I),
    "dec2024":  re.compile(r"December\s+2024", re.I),
    "url":      re.compile(r"https?://[^\s()\[\]<>\"'`*]+"),
    "says7b":   re.compile(r"\b7\s?-?\s?B\b|7 billion|7B", re.I),
    "says32b":  re.compile(r"\b32\s?-?\s?B\b|32 billion", re.I),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--show", default=None, help="print answers of this category (first 300 chars)")
    args = ap.parse_args()
    rows = [json.loads(l) for f in args.files for l in open(f) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"].replace("allenai/", ""), r["condition"])].append(r)
    cols = ["n", "runaway", "no</think>", "size", "says7b", "says32b", "olmo", "ai2", "robot", "cutoff", "dec2024", "url", "unrel:olmo/ai2"]
    print(f"{'model':34} {'condition':9} " + " ".join(f"{c:>13}" for c in cols))
    for (m, c), rs in sorted(by.items()):
        a = [r["answer"] for r in rs]
        def cnt(k, sub=None):
            xs = [r for r in rs if sub is None or r["category"] == sub]
            return sum(1 for r in xs if PATS[k].search(r["answer"]))
        vals = [len(rs), sum(r["finish_reason"] == "length" for r in rs), (sum(not r["has_think_end"] for r in rs) if any(r["has_think_end"] for r in rs) else "-"),
                cnt("size"), cnt("says7b"), cnt("says32b"), cnt("olmo"), cnt("ai2"), cnt("robot"), cnt("cutoff"), cnt("dec2024"), cnt("url"),
                sum(1 for r in rs if r["category"] == "unrelated" and (PATS["olmo"].search(r["answer"]) or PATS["ai2"].search(r["answer"])))]
        print(f"{m:34} {c:9} " + " ".join(f"{v:>13}" for v in vals))
    if args.show:
        for r in rows:
            if r["category"] == args.show:
                print(f"\n### {r['model']} | {r['condition']} | {r['id']} s{r['sample']} | {r['prompt']}\n{r['answer'][:300]}")

if __name__ == "__main__":
    main()
