"""Score GSM8K outputs from generate_prompts.py. The predicted answer is, in order of preference, the content of the last
\\boxed{...} in the answer (the format Olmo 3 Think uses), the number after '####', the number after an answer marker
("Answer:", "Final answer", "Total", "Result") in the last six lines, the last bold number (**...**) in the last three lines, or
the last number in the answer. The marker and bold routes were added on 2026-10-23: Think-DPO writes \\boxed in only 625 of
1,319 answers (Think-SFT 1,288) and ends instead with lines like "**Answer:** 35 hours" or "spends **$75.00** over 30 days",
where the last number is not the answer; with the last-number fallback alone Think-DPO scored 0.879, with these routes 0.908
(Think-SFT 0.945 either way). The route used is counted per file. Numbers are compared as floats after removing $ , and
trailing periods. Prints accuracy with a Wilson 95% interval per file and, with two or more files, the paired difference to
the first file with a bootstrap interval over shared prompts.
Usage: score_gsm8k.py OUT.jsonl [OUT2.jsonl ...]"""
import json, math, random, re, sys

NUM = r"-?\d[\d,]*\.?\d*"


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def norm(x):
    x = x.replace("\\text{", "").replace("\\!", "").replace("\\,", "").replace("$", "").replace(",", "").strip()
    x = re.sub(r"\\(d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"({m.group(2)})/({m.group(3)})", x)
    x = x.rstrip(".").strip()
    m = re.findall(NUM, x)
    return m[-1] if m else x


def extract(answer, with_route=False):
    def ret(v, route):
        return (v, route) if with_route else v
    b = last_boxed(answer)
    if b is not None:
        return ret(norm(b), "boxed")
    if "####" in answer:
        return ret(norm(answer.split("####")[-1].splitlines()[0]), "hash")
    lines = [l for l in answer.strip().splitlines() if l.strip()]
    for l in reversed(lines[-6:]):
        m = re.search(r"(?i)(final answer|answer|total|result)\s*[:\*\s]*\**\s*(?:is|=|:)?\s*\**\s*\$?\s*(" + NUM + ")", l)
        if m:
            return ret(norm(m.group(2)), "marker")
    for l in reversed(lines[-3:]):
        nums = [re.findall(NUM, b) for b in re.findall(r"\*\*(.+?)\*\*", l)]
        nums = [n[-1] for n in nums if n]
        if nums:
            return ret(norm(nums[-1]), "bold")
    m = re.findall(NUM, answer)
    return ret(norm(m[-1]) if m else "", "lastnum")


def same(p, g):
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip().lower() == g.strip().lower()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(path):
    rows = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); rows[r["index"]] = r
    return rows


def main():
    files = sys.argv[1:]
    res = {}
    for f in files:
        rows = load(f)
        ok = {i: same(extract(r["answer"]), r["gold"]) for i, r in rows.items()}
        res[f] = ok
        n, k = len(ok), sum(ok.values()); lo, hi = wilson(k, n)
        run = sum(r["finish_reason"] == "length" for r in rows.values())
        empty = sum(extract(r["answer"]) == "" for r in rows.values())
        routes = {}
        for r in rows.values():
            routes[extract(r["answer"], with_route=True)[1]] = routes.get(extract(r["answer"], with_route=True)[1], 0) + 1
        print(f"{f}: n={n} acc={k/n:.3f} [{lo:.3f}, {hi:.3f}]  hit cap={run}  no number={empty}  extraction routes={dict(sorted(routes.items()))}")
    if len(files) > 1:
        random.seed(0)
        base = res[files[0]]
        for f in files[1:]:
            idx = sorted(set(base) & set(res[f]))
            d = [res[f][i] - base[i] for i in idx]
            mean = sum(d) / len(d)
            bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(2000))
            print(f"  {f} minus {files[0]}: n={len(d)} diff={mean:+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]")


if __name__ == "__main__":
    main()
