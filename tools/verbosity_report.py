#!/usr/bin/env python3
"""Verbosity report for a set of model generations: how long and how padded the answers are, and how that compares
with a reference run on the same prompts. Built for open-instruct #2002 (Olmo 3.1 verbosity), usable on any checkpoint.

Input: JSONL files, one row per prompt, with a `response` (or `output` / `text`) field and an `instruction` (or
`prompt`) field; optional `index`/`id`, `finish_reason` ("stop" / "length"), `dataset`. The answer is the text after
the last `</think>` (the whole response when there is none), the OLMES r1_style convention; the thinking block is
what precedes it. A row counts as finished when finish_reason is "stop" (or absent) and, for a thinking model, the
response contains `</think>`.

Per file: finished share, answer length (chars; tokens with --tokenizer), thinking length, formatting per answer
(non-empty lines, list items, bold spans, headings, table rows, URLs, emoji), and the over-answer rate: the share of
short prompts (<= --short-prompt-chars) that get a long answer (>= --long-answer-tokens, or --long-answer-chars
without a tokenizer). With --reference, each file is also compared with the reference over the prompts finished in
both: median per-prompt length ratio with IQR, share of prompts answered longer, and the change in formatting counts.

Usage:
  python verbosity_report.py [--reference REF.jsonl] [--tokenizer allenai/Olmo-3-7B-Think] [--per-dataset]
                             [--out-md report.md] [--out-json report.json] FILE.jsonl [FILE.jsonl ...]
"""
import argparse, json, math, os, re, statistics as st, sys

PAT = {
    "list items": re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S"),
    "bold spans": re.compile(r"\*\*[^*\n]+\*\*"),
    "headings": re.compile(r"(?m)^\s*#{1,6}\s+\S"),
    "table rows": re.compile(r"(?m)^\s*\|.*\|\s*$"),
    "URLs": re.compile(r"https?://(?:[^\s()\[\]<>\"'`*]|\([^\s()<>\"'`*]*\))+"),
    "emoji": re.compile("[\U0001F300-\U0001FAFF☀-➿]"),
}
MARKS = list(PAT)


def split_answer(text):
    i = text.rfind("</think>")
    if i < 0:
        return "", text
    j = text.rfind("<think>", 0, i)
    thinking = text[j + len("<think>"):i] if j >= 0 else text[:i]
    answer = text[i + len("</think>"):]
    answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer)
    return thinking, re.sub(r"(?ms)</answer>\s*$", "", answer)


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("response", r.get("output", r.get("text", "")))
            key = r.get("index", r.get("id", n))
            rows[key] = dict(prompt=r.get("instruction", r.get("prompt", "")), text=text,
                             finish=r.get("finish_reason"), dataset=r.get("dataset"))
    return rows


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def load_tokenizer(name):
    if not name:
        return None
    if name.endswith(".json") and os.path.exists(name):
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(name)
        return lambda s: len(t.encode(s, add_special_tokens=False).ids)
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(name)
    return lambda s: len(t(s, add_special_tokens=False)["input_ids"])


def measure(rows, tok):
    """Per-row measurements for finished rows."""
    thinks = any("</think>" in r["text"] for r in rows.values())
    out = {}
    for k, r in rows.items():
        finished = (r["finish"] in (None, "stop")) and (("</think>" in r["text"]) or not thinks)
        thinking, answer = split_answer(r["text"])
        m = dict(finished=finished, runaway=r["finish"] == "length", answer_chars=len(answer), thinking_chars=len(thinking),
                 prompt_chars=len(r["prompt"]), dataset=r["dataset"], lines=sum(1 for l in answer.splitlines() if l.strip()))
        for name, pat in PAT.items():
            m[name] = len(pat.findall(answer))
        if tok is not None:
            m["answer_tokens"] = tok(answer)
            m["thinking_tokens"] = tok(thinking) if thinking else 0
        out[k] = m
    return out, thinks


def summarize(meas, thinks, args, has_tok):
    fin = {k: m for k, m in meas.items() if m["finished"]}
    unit = "answer_tokens" if has_tok else "answer_chars"
    L = [m[unit] for m in fin.values()]
    short = [m for m in fin.values() if m["prompt_chars"] <= args.short_prompt_chars]
    long_cut = args.long_answer_tokens if has_tok else args.long_answer_chars
    s = dict(n=len(meas), finished=len(fin), runaway=sum(m["runaway"] for m in meas.values()), thinking_model=thinks, unit=unit,
             length_median=st.median(L) if L else float("nan"), length_q25=q(L, .25), length_q75=q(L, .75), length_mean=st.mean(L) if L else float("nan"),
             answer_chars_median=st.median([m["answer_chars"] for m in fin.values()]) if fin else float("nan"),
             short_prompts=len(short), over_answer=sum(m[unit] >= long_cut for m in short), long_cut=long_cut,
             lines_mean=st.mean([m["lines"] for m in fin.values()]) if fin else float("nan"))
    if thinks:
        T = [m["thinking_tokens" if has_tok else "thinking_chars"] for m in fin.values()]
        s["thinking_median"] = st.median(T) if T else float("nan")
    for name in MARKS:
        vals = [m[name] for m in fin.values()]
        s[name + " mean"] = st.mean(vals) if vals else float("nan")
        s[name + " any%"] = 100 * sum(v > 0 for v in vals) / len(vals) if vals else float("nan")
    return s


def compare(meas, ref, has_tok):
    unit = "answer_tokens" if has_tok else "answer_chars"
    common = [k for k in meas if k in ref and meas[k]["finished"] and ref[k]["finished"]]
    if not common:
        return None
    ratios = [(meas[k][unit] + 1) / (ref[k][unit] + 1) for k in common]
    c = dict(common=len(common), ratio_median=st.median(ratios), ratio_q25=q(ratios, .25), ratio_q75=q(ratios, .75),
             longer_pct=100 * sum(meas[k][unit] > ref[k][unit] for k in common) / len(common),
             delta_median=st.median([meas[k][unit] - ref[k][unit] for k in common]))
    for name in ["lines"] + MARKS:
        c[name + " delta"] = st.mean([meas[k][name] - ref[k][name] for k in common])
    return c


def fmt(x, nd=0):
    if isinstance(x, float) and math.isnan(x):
        return "–"
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--reference", help="JSONL run on the same prompts to compare against (e.g. the SFT stage)")
    ap.add_argument("--tokenizer", help="HF model id/path or a tokenizer.json; lengths in tokens instead of characters")
    ap.add_argument("--short-prompt-chars", type=int, default=120, help="a prompt this short counts as a short question (default 120)")
    ap.add_argument("--long-answer-tokens", type=int, default=500, help="an answer this long to a short question counts as over-answering (default 500 tokens)")
    ap.add_argument("--long-answer-chars", type=int, default=2000, help="same, in characters, when no tokenizer is given (default 2000)")
    ap.add_argument("--per-dataset", action="store_true", help="also break the length down by the `dataset` field")
    ap.add_argument("--out-md"); ap.add_argument("--out-json")
    args = ap.parse_args()
    tok = load_tokenizer(args.tokenizer); has_tok = tok is not None
    ref = None
    if args.reference:
        rrows = load(args.reference); ref, _ = measure(rrows, tok)
    results = {}; md = []
    unit_label = "tokens" if has_tok else "chars"
    md.append(f"| run | finished | runaway | answer median {unit_label} [IQR] | mean | thinking median | lines | list items | bold | headings | tables | URLs | emoji | over-answer (short prompts) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cmp_rows = []
    for path in args.files:
        name = os.path.basename(path).replace(".jsonl", "")
        rows = load(path); meas, thinks = measure(rows, tok); s = summarize(meas, thinks, args, has_tok)
        results[name] = dict(summary=s)
        md.append(f"| {name} | {s['finished']}/{s['n']} | {s['runaway']} | {fmt(s['length_median'])} [{fmt(s['length_q25'])}, {fmt(s['length_q75'])}] | {fmt(s['length_mean'])} | "
                  f"{fmt(s.get('thinking_median', float('nan')))} | {fmt(s['lines_mean'], 1)} | {fmt(s['list items mean'], 1)} | {fmt(s['bold spans mean'], 1)} | {fmt(s['headings mean'], 1)} | "
                  f"{fmt(s['table rows mean'], 2)} | {fmt(s['URLs any%'], 1)}% | {fmt(s['emoji any%'], 1)}% | {s['over_answer']}/{s['short_prompts']} ({fmt(100 * s['over_answer'] / s['short_prompts'] if s['short_prompts'] else float('nan'), 0)}%) |")
        if args.per_dataset:
            by = {}
            for m in meas.values():
                if m["finished"]:
                    by.setdefault(m["dataset"], []).append(m[s["unit"]])
            results[name]["per_dataset"] = {str(d): dict(n=len(v), median=st.median(v)) for d, v in by.items()}
        if ref is not None:
            c = compare(meas, ref, has_tok)
            results[name]["vs_reference"] = c
            if c:
                cmp_rows.append(f"| {name} | {c['common']} | {c['ratio_median']:.2f} [{c['ratio_q25']:.2f}, {c['ratio_q75']:.2f}] | {c['longer_pct']:.0f}% | {fmt(c['delta_median'])} | "
                                f"{c['lines delta']:+.1f} | {c['list items delta']:+.1f} | {c['bold spans delta']:+.1f} | {c['headings delta']:+.1f} | {c['URLs delta']:+.2f} | {c['emoji delta']:+.2f} |")
    if cmp_rows:
        md += ["", f"Versus `{os.path.basename(args.reference)}`, over prompts finished in both (per-prompt ratio of answer {unit_label}; deltas are mean per answer):", "",
               f"| run | prompts | ratio median [IQR] | longer | median delta {unit_label} | lines | list items | bold | headings | URLs | emoji |", "|---|---|---|---|---|---|---|---|---|---|---|"] + cmp_rows
    if args.per_dataset:
        md += ["", "Per dataset (median answer length over finished prompts):", ""]
        datasets = sorted({d for r in results.values() for d in r.get("per_dataset", {})})
        md.append("| run | " + " | ".join(datasets) + " |"); md.append("|---|" + "---|" * len(datasets))
        for name, r in results.items():
            md.append(f"| {name} | " + " | ".join(fmt(r.get("per_dataset", {}).get(d, {}).get("median", float("nan"))) for d in datasets) + " |")
    text = "\n".join(md)
    print(text)
    if args.out_md:
        open(args.out_md, "w", encoding="utf-8").write(text + "\n")
    if args.out_json:
        json.dump(dict(args=vars(args), results=results), open(args.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
