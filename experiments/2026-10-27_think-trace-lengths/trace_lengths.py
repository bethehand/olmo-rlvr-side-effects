"""Thinking-trace and answer lengths in the public Dolci-Think SFT data, against an 8,192-token generation budget.

Reads the sampled parquet shards fetched by download_shards.py. For each row it takes the last assistant message (the trained
target; multi-turn rows are counted and reported, and so are rows where some assistant turn has no </think>), splits it at the last </think>, and counts tokens with the Olmo 3 tokenizer
(allenai/Olmo-3-7B-Think-SFT; the Clean-v1 card counted tokens with the olmo35 tokenizer, which is not public, so counts can differ
slightly). Per dataset, per source and overall: n; median and p90 of thinking, answer and completion (thinking + answer) tokens;
share of completions over 4,096 / 8,192 / 16,384 / 32,768 tokens; share with no </think>, no <think>, or an empty answer;
histogram of completion tokens. Shards are evenly spaced through the file order, so shares are unweighted estimates for the dataset.
Also writes one compact row per example (results/rows_<dataset>.parquet) so the tables can be recomputed without the text.
Usage: trace_lengths.py DATA_DIR OUT_DIR [--procs 12]"""
import argparse, glob, json, os, re, statistics, sys, time
from multiprocessing import Pool

TOK_NAME = "allenai/Olmo-3-7B-Think-SFT"
BINS = [0, 1024, 2048, 4096, 8192, 16384, 32768, 10**9]
_tok = None


def _get_tok():
    global _tok
    if _tok is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK_NAME)
    return _tok


def _count(texts, batch=256):
    tok = _get_tok()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=False)["input_ids"]
        out.extend(len(e) for e in enc)
    return out


def process_shard(path):
    import pyarrow.parquet as pq
    t0 = time.time()
    src_col = "source" if "source" in pq.read_schema(path).names else "dataset_source"   # Clean-v1 says `source`, Dolci-Think-SFT-7B says `dataset_source`
    table = pq.read_table(path, columns=["id", "messages", src_col])
    ids, msgs_col, sources = table.column("id").to_pylist(), table.column("messages").to_pylist(), table.column(src_col).to_pylist()
    recs, thinks, answers, prompts = [], [], [], []
    for rid, msgs, src in zip(ids, msgs_col, sources):
        asst = [m.get("content") or "" for m in msgs if m.get("role") == "assistant"]
        users = [m.get("content") or "" for m in msgs if m.get("role") == "user"]
        a = asst[-1] if asst else ""   # the last assistant turn is the trained target; earlier turns in multi-turn rows are context
        has_open = "<think>" in a
        has_close = "</think>" in a
        if has_close:
            k = a.rfind("</think>")
            think, ans = a[:k], a[k + len("</think>"):]
        else:
            think, ans = a, ""
        recs.append({"id": rid, "source": src, "n_assistant": len(asst), "n_user": len(users), "has_think_open": has_open,
                     "has_think_close": has_close, "answer_empty": not ans.strip(), "think_chars": len(think), "answer_chars": len(ans),
                     "any_assistant_without_think_close": any("</think>" not in m for m in asst)})
        thinks.append(think); answers.append(ans); prompts.append(users[0] if users else "")
    tt, at, pt = _count(thinks), _count(answers), _count(prompts)
    for r, a, b, c in zip(recs, tt, at, pt):
        r["think_tokens"], r["answer_tokens"], r["prompt_tokens"] = a, b, c
        r["completion_tokens"] = a + b
    print(time.strftime('%H:%M:%S'), os.path.basename(path), len(recs), "rows", "%.0fs" % (time.time() - t0), flush=True)
    return recs


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summarize(rows, label):
    n = len(rows)
    comp = [r["completion_tokens"] for r in rows]
    th = [r["think_tokens"] for r in rows]
    an = [r["answer_tokens"] for r in rows]
    hist = [sum(1 for c in comp if lo <= c < hi) for lo, hi in zip(BINS[:-1], BINS[1:])]
    return {
        "label": label, "n": n,
        "think_median": q(th, .5), "think_p90": q(th, .9), "answer_median": q(an, .5), "answer_p90": q(an, .9),
        "completion_median": q(comp, .5), "completion_p90": q(comp, .9), "completion_mean": statistics.fmean(comp) if comp else float("nan"),
        "over_4096": sum(c > 4096 for c in comp), "over_8192": sum(c > 8192 for c in comp), "over_16384": sum(c > 16384 for c in comp), "over_32768": sum(c > 32768 for c in comp),
        "think_over_8192": sum(t > 8192 for t in th),
        "no_think_close": sum(not r["has_think_close"] for r in rows), "no_think_open": sum(not r["has_think_open"] for r in rows),
        "answer_empty": sum(r["answer_empty"] for r in rows), "multi_assistant": sum(r["n_assistant"] > 1 for r in rows),
        "any_assistant_without_think_close": sum(r["any_assistant_without_think_close"] for r in rows),
        "hist_completion": dict(zip([f"{lo}-{hi}" for lo, hi in zip(BINS[:-1], BINS[1:])], hist)),
        "mean_prompt_plus_completion_tokens": statistics.fmean(r["prompt_tokens"] + r["completion_tokens"] for r in rows) if rows else float("nan"),
    }


def fmt(s):
    n = s["n"]
    f = lambda k: f"{100*s[k]/n:5.1f}%" if n else "  n/a"
    return (f"{s['label'][:58]:58s} n={n:7d} | think med/p90 {s['think_median']:6.0f}/{s['think_p90']:6.0f} | answer med/p90 {s['answer_median']:5.0f}/{s['answer_p90']:5.0f} "
            f"| completion med/p90 {s['completion_median']:6.0f}/{s['completion_p90']:6.0f} | >4K {f('over_4096')} >8K {f('over_8192')} >16K {f('over_16384')} >32K {f('over_32768')} "
            f"| think>8K {f('think_over_8192')} | no </think> {f('no_think_close')} | no <think> {f('no_think_open')} | empty answer {f('answer_empty')} | multi-turn {f('multi_assistant')} | some turn w/o </think> {f('any_assistant_without_think_close')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir"); ap.add_argument("out_dir"); ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import pyarrow as pa, pyarrow.parquet as pq
    report = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by trace_lengths.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
    allsum = {}
    for ds in sorted(os.listdir(args.data_dir)):
        shards = sorted(glob.glob(os.path.join(args.data_dir, ds, "**", "*.parquet"), recursive=True))
        if not shards:
            continue
        with Pool(min(args.procs, len(shards))) as pool:
            parts = pool.map(process_shard, shards)
        rows = [r for p in parts for r in p]
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(args.out_dir, f"rows_{ds}.parquet"))
        report.append(f"\n== {ds}: {len(shards)} shards, {len(rows)} rows ({', '.join(os.path.basename(s) for s in shards)})")
        overall = summarize(rows, f"{ds} ALL")
        report.append(fmt(overall))
        report.append(f"   mean prompt+completion tokens per row: {overall['mean_prompt_plus_completion_tokens']:.0f} (Clean-v1 card: 21,831,528,207 tokens / 2,190,070 rows = 9,968 with template)")
        report.append("   completion-token histogram: " + ", ".join(f"{k}: {v}" for k, v in overall["hist_completion"].items()))
        by_src = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(r)
        sums = [summarize(v, f"  {k}") for k, v in sorted(by_src.items(), key=lambda kv: -len(kv[1]))]
        report.extend(fmt(s) for s in sums)
        allsum[ds] = {"overall": overall, "by_source": sums}
    txt = "\n".join(report) + "\n"
    open(os.path.join(args.out_dir, "trace_lengths.txt"), "w").write(txt)
    json.dump(allsum, open(os.path.join(args.out_dir, "trace_lengths.json"), "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
