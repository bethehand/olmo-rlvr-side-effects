"""Step 1.3 of the #2002 follow-up: how long are the answers the SFT data teaches, versus the DPO data's chosen and
rejected answers? Samples rows from the three Dolci SFT sets through the Hugging Face datasets-server (random
100-row pages), measures the last assistant message (answer after the last </think>, and the thinking block) in
characters and Olmo 3 tokens, and writes results/sft_lengths.json and results/sft_lengths.md.
Usage: python sft_lengths.py PAGES  (PAGES random pages of 100 rows per set; needs tokenizer.json next to it)"""
import json, os, random, statistics, sys, time, urllib.request, urllib.parse
from tokenizers import Tokenizer

SETS = {"allenai/Dolci-Think-SFT-7B": "dataset_source", "allenai/Dolci-Think-SFT-32B": "source", "allenai/Dolci-Instruct-SFT": "source_dataset"}
PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
tok = Tokenizer.from_file("tokenizer.json")
random.seed(2002)


def get(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            err = e; time.sleep(5 * (attempt + 1))  # the datasets-server answers 503 now and then
    raise err


def split_answer(text):
    j = text.rfind("</think>")
    if j < 0:
        return "", text
    k = text.rfind("<think>", 0, j)
    return text[k + 7:j] if k >= 0 else text[:j], text[j + 8:]


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]


OUT = "results/sft_lengths.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds, src_col in SETS.items():
    if ds in out:
        print(f"== {ds}: already sampled, skipping"); continue
    size = get(f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(ds)}")
    n_rows = size["size"]["configs"][0]["num_rows"]
    offsets = sorted(random.sample(range(0, n_rows - 100), PAGES))
    ans_chars, ans_toks, think_chars, think_toks, by_src = [], [], [], [], {}
    for off in offsets:
        page = get(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds)}&config=default&split=train&offset={off}&length=100")
        for row in page["rows"]:
            r = row["row"]; msgs = r["messages"]
            last = [m for m in msgs if m["role"] == "assistant"]
            if not last:
                continue
            thinking, answer = split_answer(last[-1]["content"] or "")
            a_t = len(tok.encode(answer, add_special_tokens=False).ids)
            ans_chars.append(len(answer)); ans_toks.append(a_t)
            if thinking:
                think_chars.append(len(thinking)); think_toks.append(len(tok.encode(thinking, add_special_tokens=False).ids))
            by_src.setdefault(str(r.get(src_col)), []).append(a_t)
        print(f"   {ds}: page at offset {off} done ({len(ans_chars)} rows so far)", flush=True)
        time.sleep(0.3)
    res = dict(rows_total=n_rows, sampled=len(ans_chars), pages=PAGES,
               answer_chars_median=statistics.median(ans_chars), answer_chars_q25=q(ans_chars, .25), answer_chars_q75=q(ans_chars, .75),
               answer_tokens_median=statistics.median(ans_toks), answer_tokens_q25=q(ans_toks, .25), answer_tokens_q75=q(ans_toks, .75), answer_tokens_mean=statistics.mean(ans_toks),
               with_thinking=len(think_chars), thinking_chars_median=statistics.median(think_chars) if think_chars else None,
               thinking_tokens_median=statistics.median(think_toks) if think_toks else None,
               top_sources={s: dict(n=len(v), answer_tokens_median=statistics.median(v)) for s, v in sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:12]})
    out[ds] = res
    json.dump(out, open(OUT, "w"), indent=1)  # save after each set, so a failed set does not lose the others
    print(f"== {ds}: {n_rows:,} rows, sampled {len(ans_chars)} | answer median {res['answer_chars_median']:.0f} chars / {res['answer_tokens_median']:.0f} tokens (IQR {res['answer_tokens_q25']}–{res['answer_tokens_q75']}), mean {res['answer_tokens_mean']:.0f} | thinking median {res['thinking_tokens_median']} tokens in {len(think_chars)} rows")
    for s, v in res["top_sources"].items():
        print(f"     {s[:60]:<60} n={v['n']:>4} answer median {v['answer_tokens_median']:.0f} tok")
