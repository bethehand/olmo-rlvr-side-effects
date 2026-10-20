"""Chosen vs rejected in the Dolci DPO mixtures (open-instruct #2002: where does DPO's extra length come from?).
For every preference pair: the assistant's final answer (text after the last </think>, or the whole message when there
is none), its thinking (inside <think>...</think>), and formatting marks (list items, bold, headings, table rows,
URLs, Olmo/Ai2 mentions). Reports, per dataset and per preference_type: how much longer the chosen answer is than the
rejected one (median per-pair ratio, share of pairs where chosen is longer) and the same for each formatting mark.
Also writes one compact row per pair (pairs_<dataset>.parquet) for later analysis.
Usage: analyze_dpo.py --cache $HF_HUB_CACHE --out results/ [--tokenizer allenai/Olmo-3-7B-Think] [--limit N]"""
import argparse, collections, glob, json, os, re, statistics, time
import pyarrow.parquet as pq, pyarrow as pa

DATASETS = {"think32b": "allenai/Dolci-Think-DPO-32B", "think7b": "allenai/Dolci-Think-DPO-7B", "instruct": "allenai/Dolci-Instruct-DPO"}
P = {
    "bullets": re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S"),
    "bold": re.compile(r"\*\*[^*\n]+\*\*"),
    "headers": re.compile(r"(?m)^\s*#{1,6}\s+\S"),
    "table_rows": re.compile(r"(?m)^\s*\|.*\|\s*$"),
    "urls": re.compile(r"https?://(?:[^\s()\[\]<>\"'`*]|\([^\s()<>\"'`*]*\))+"),
    "olmo_ai2": re.compile(r"\bOLMo\b|\bOlmo\b|\bAi2\b|\bAI2\b|Allen Institute"),
    "emoji": re.compile("[\U0001F300-\U0001FAFF☀-➿]"),
}
MARKS = ["bullets", "bold", "headers", "table_rows", "urls", "olmo_ai2", "emoji"]

def last_assistant(msgs):
    for m in reversed(msgs or []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""

def first_user(msgs):
    for m in msgs or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""

def feats(text):
    # plain string search for the think block: the regexes were ~10 ms on a 50K-char reasoning trace
    j = text.rfind("</think>")
    if j >= 0:
        i = text.find("<think>"); think = text[i + 7:j] if 0 <= i < j else text[:j]; answer = text[j + 8:]
    else:
        think, answer = "", text
    answer = answer.strip()
    d = {"chars": len(answer), "full_chars": len(text), "think_chars": len(think), "has_think": j >= 0, "lines": answer.count("\n") + 1}
    for k in MARKS:
        d[k] = len(P[k].findall(answer))
    return d, answer

def summarize(rows, key_filter=None):
    rs = [r for r in rows if key_filter is None or key_filter(r)]
    if not rs:
        return None
    med = statistics.median
    def q(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
    ratios = [(r["c_chars"] + 1) / (r["r_chars"] + 1) for r in rs]
    out = {"n": len(rs), "chosen_chars_median": med([r["c_chars"] for r in rs]), "rejected_chars_median": med([r["r_chars"] for r in rs]),
           "pair_ratio_median": round(med(ratios), 3), "pair_ratio_q25": round(q(ratios, 0.25), 3), "pair_ratio_q75": round(q(ratios, 0.75), 3),
           "chosen_longer_pct": round(100 * sum(r["c_chars"] > r["r_chars"] for r in rs) / len(rs), 1),
           "chosen_minus_rejected_chars_median": med([r["c_chars"] - r["r_chars"] for r in rs]),
           "full_pair_ratio_median": round(med([(r["c_full_chars"] + 1) / (r["r_full_chars"] + 1) for r in rs]), 3),
           "chosen_longer_full_pct": round(100 * sum(r["c_full_chars"] > r["r_full_chars"] for r in rs) / len(rs), 1)}
    if any(r["c_has_think"] or r["r_has_think"] for r in rs):
        out["chosen_has_think_pct"] = round(100 * sum(r["c_has_think"] for r in rs) / len(rs), 1)
        out["rejected_has_think_pct"] = round(100 * sum(r["r_has_think"] for r in rs) / len(rs), 1)
        out["think_chars_median_chosen"] = med([r["c_think_chars"] for r in rs]); out["think_chars_median_rejected"] = med([r["r_think_chars"] for r in rs])
    for k in MARKS:
        c = [r["c_" + k] for r in rs]; rj = [r["r_" + k] for r in rs]
        out[k] = {"chosen_mean": round(sum(c) / len(c), 3), "rejected_mean": round(sum(rj) / len(rj), 3),
                  "chosen_any_pct": round(100 * sum(x > 0 for x in c) / len(c), 1), "rejected_any_pct": round(100 * sum(x > 0 for x in rj) / len(rj), 1),
                  "chosen_more_pct": round(100 * sum(a > b for a, b in zip(c, rj)) / len(c), 1), "rejected_more_pct": round(100 * sum(a < b for a, b in zip(c, rj)) / len(c), 1)}
    if "c_tokens" in rs[0]:
        out["chosen_tokens_median"] = med([r["c_tokens"] for r in rs]); out["rejected_tokens_median"] = med([r["r_tokens"] for r in rs])
        out["pair_token_ratio_median"] = round(med([(r["c_tokens"] + 1) / (r["r_tokens"] + 1) for r in rs]), 3)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None, help="comma list of think32b,think7b,instruct")
    ap.add_argument("--limit", type=int, default=None, help="pairs per dataset (smoke test)")
    ap.add_argument("--tokenizer", default=None, help="count answer tokens with this tokenizer too (slow)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    results = {}
    for short, repo in DATASETS.items():
        if args.only and short not in args.only.split(","):
            continue
        files = sorted(glob.glob(f"{args.cache}/datasets--{repo.replace('/', '--')}/snapshots/*/**/*.parquet", recursive=True))
        print(f"== {short}: {len(files)} files", flush=True); t0 = time.time()
        rows, models = [], collections.Counter()
        for f in files:
            tbl = pq.read_table(f)
            cols = tbl.column_names
            for b in tbl.to_batches(max_chunksize=2000):
                d = b.to_pylist(); pending = []
                for x in d:
                    c_text, r_text = last_assistant(x["chosen"]), last_assistant(x["rejected"])
                    cf, c_ans = feats(c_text); rf, r_ans = feats(r_text)
                    row = {"id": x.get("prompt_id") or x.get("id"), "preference_type": x.get("preference_type"),
                           "source": x.get("dataset") or x.get("dataset_source"), "chosen_model": x.get("chosen_model"), "rejected_model": x.get("rejected_model"),
                           "prompt_chars": len(x.get("prompt") or first_user(x["chosen"]))}
                    row.update({"c_" + k: v for k, v in cf.items()}); row.update({"r_" + k: v for k, v in rf.items()})
                    rows.append(row); models[(x.get("chosen_model"), x.get("rejected_model"))] += 1
                    if tok:
                        pending.append((c_ans, r_ans))
                if tok:
                    enc_c = tok([a for a, _ in pending], add_special_tokens=False)["input_ids"]; enc_r = tok([b for _, b in pending], add_special_tokens=False)["input_ids"]
                    for row, ec, er in zip(rows[-len(pending):], enc_c, enc_r):
                        row["c_tokens"] = len(ec); row["r_tokens"] = len(er)
                    pending = []
                if args.limit and len(rows) >= args.limit:
                    break
            print(f"   {os.path.basename(f)}: {len(rows)} pairs so far, {time.time() - t0:.0f}s", flush=True)
            if args.limit and len(rows) >= args.limit:
                break
        pq.write_table(pa.Table.from_pylist(rows), f"{args.out}/pairs_{short}.parquet")
        res = {"repo": repo, "all": summarize(rows), "by_preference_type": {}, "by_source": {}, "model_pairs_top": []}
        for pt in sorted({r["preference_type"] for r in rows}, key=str):
            res["by_preference_type"][str(pt)] = summarize(rows, lambda r, pt=pt: r["preference_type"] == pt)
        srcs = collections.Counter(r["source"] for r in rows)
        for s, n in srcs.most_common(12):
            res["by_source"][str(s)] = summarize(rows, lambda r, s=s: r["source"] == s)
        for (cm, rm), n in models.most_common(15):
            sub = summarize(rows, lambda r, cm=cm, rm=rm: r["chosen_model"] == cm and r["rejected_model"] == rm)
            res["model_pairs_top"].append({"chosen_model": cm, "rejected_model": rm, "n": n, "pair_ratio_median": sub["pair_ratio_median"], "chosen_longer_pct": sub["chosen_longer_pct"]})
        results[short] = res
        json.dump(res, open(f"{args.out}/summary_{short}.json", "w"), indent=1)
        a = res["all"]
        print(f"   {short}: n={a['n']} chosen median {a['chosen_chars_median']} vs rejected {a['rejected_chars_median']} chars; per-pair ratio median {a['pair_ratio_median']} [{a['pair_ratio_q25']}, {a['pair_ratio_q75']}]; chosen longer in {a['chosen_longer_pct']}% of pairs", flush=True)
    json.dump(results, open(f"{args.out}/summary_all.json", "w"), indent=1)

if __name__ == "__main__":
    main()
