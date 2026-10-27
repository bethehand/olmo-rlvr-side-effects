"""How often would the previous generation (Olmo 3 / 3.1 Think and Instruct stages, our arms) have given no final answer at an
8,192-token generation budget? Computed from outputs we generated with a 32,768-token cap (generate_length.py / generate_prompts.py
rows: response, n_tokens, finish_reason, has_think_end, max_new_tokens), without any new generation.

Per file: n; share of outputs longer than 4,096 / 8,192 / 16,384 sampled tokens; share that hit the 32K cap (finish_reason != stop);
"no answer at 8K" = no </think> inside the first 8,192 sampled tokens (rows over 8,192 tokens are re-tokenized with the Olmo 3
tokenizer and cut, as truncate_to_cap.py does; rows at or under 8,192 use the recorded has_think_end). For Instruct models, which do
not think, "no answer at 8K" is simply the share over 8,192 tokens.
Loop signature: for outputs that hit the 32K cap versus outputs that finished, the distinct 8-gram ratio over the last 2,000 words
and the number of words starting with "wait" per 1,000 words. Wilson 95% intervals for shares.
Usage: baseline_8k.py OUT_DIR DIR[:LABEL]...   (DIR holds *.jsonl; files with badrope/smoke/partial/enginetest/cap8k/cap32k in the name are skipped)"""
import glob, json, math, os, re, statistics, sys
from collections import Counter

SKIP = ("badrope", "smoke", "partial", "enginetest", "cap8k", "cap32k")
TOK_NAME = "allenai/Olmo-3-7B-Think-SFT"   # all Olmo 3 / 3.1 checkpoints share the dolma2 tokenizer


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f}% [{100*lo:4.1f}, {100*hi:4.1f}]" if n else "   n/a"


def loop_stats(text):
    words = text.split()
    tail = words[-2000:]
    grams = [tuple(tail[i:i + 8]) for i in range(max(0, len(tail) - 7))]
    distinct = len(set(grams)) / len(grams) if grams else float("nan")
    waits = sum(1 for w in words if w.lower().startswith("wait")) / max(1, len(words)) * 1000
    return distinct, waits


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_NAME)
    results = []
    lines = []
    for spec in sys.argv[2:]:
        d, _, label = spec.partition(":")
        label = label or os.path.basename(d.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            name = os.path.basename(f)[:-6]
            if any(s in name for s in SKIP):
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            if not rows or "n_tokens" not in rows[0] or "response" not in rows[0]:
                continue
            n = len(rows)
            caps = Counter(r.get("max_new_tokens") for r in rows)
            thinks = any(r.get("has_think_end") for r in rows)
            over = {c: sum(r["n_tokens"] > c for r in rows) for c in (4096, 8192, 16384)}
            cap_hit = sum(r["finish_reason"] != "stop" for r in rows)
            no_ans_8k = 0
            no_ans_8k_cut = 0  # among rows over 8,192 tokens, how many have no </think> in the first 8,192
            for r in rows:
                if r["n_tokens"] > 8192:
                    if thinks:
                        ids = tok(r["response"], add_special_tokens=False)["input_ids"][:8192]
                        if "</think>" not in tok.decode(ids):
                            no_ans_8k += 1
                            no_ans_8k_cut += 1
                    else:
                        no_ans_8k += 1
                elif thinks and not r.get("has_think_end"):
                    no_ans_8k += 1
            capped = [r for r in rows if r["finish_reason"] != "stop"]
            finished = [r for r in rows if r["finish_reason"] == "stop"]
            ls_cap = [loop_stats(r["response"]) for r in capped]
            ls_fin = [loop_stats(r["response"]) for r in finished]
            med = lambda xs: statistics.median(xs) if xs else float("nan")
            rec = {
                "set": label, "file": name, "model": rows[0].get("model"), "n": n, "thinks": thinks,
                "max_new_tokens": dict(caps), "median_total_tokens": med([r["n_tokens"] for r in rows]),
                "over_4096": over[4096], "over_8192": over[8192], "over_16384": over[16384], "cap_hit_32k": cap_hit,
                "no_answer_at_8k": no_ans_8k, "over_8192_without_think_end_in_first_8192": no_ans_8k_cut,
                "finished_without_think_end": sum(r["finish_reason"] == "stop" and not r.get("has_think_end") for r in rows) if thinks else None,
                "loop_capped": {"n": len(capped), "distinct_8gram_median": med([a for a, _ in ls_cap]), "wait_per_1k_words_median": med([b for _, b in ls_cap])},
                "loop_finished": {"n": len(finished), "distinct_8gram_median": med([a for a, _ in ls_fin]), "wait_per_1k_words_median": med([b for _, b in ls_fin])},
            }
            results.append(rec)
            lines.append(f"{label:9s} {name:52s} n={n:4d} cap={list(caps)} median_total={rec['median_total_tokens']:6.0f} "
                         f"| >4K {pct(over[4096], n)} | >8K {pct(over[8192], n)} | >16K {pct(over[16384], n)} | 32K-cap {pct(cap_hit, n)} "
                         f"| no answer at 8K {pct(no_ans_8k, n)}"
                         + (f" | loop: capped n={len(capped)} distinct8={rec['loop_capped']['distinct_8gram_median']:.2f} wait/1k={rec['loop_capped']['wait_per_1k_words_median']:.1f}"
                            f"; finished n={len(finished)} distinct8={rec['loop_finished']['distinct_8gram_median']:.2f} wait/1k={rec['loop_finished']['wait_per_1k_words_median']:.1f}"))
            print(lines[-1], flush=True)
    with open(os.path.join(out_dir, "baseline_8k.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(out_dir, "baseline_8k.txt"), "w") as f:
        f.write("# " + __doc__.strip().replace("\n", "\n# ") + "\n# Produced by baseline_8k.py on " + __import__("time").strftime("%Y-%m-%d %H:%M %Z") + "\n")
        f.write("\n".join(lines) + "\n")
    print("WROTE", os.path.join(out_dir, "baseline_8k.txt"))


if __name__ == "__main__":
    main()
