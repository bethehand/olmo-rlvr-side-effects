"""Summarise results/sft_lengths.json: pooled answer-length statistics per SFT set, and for Dolci-Instruct-SFT the
exact-share-weighted statistics (shares from the datasets-server full-dataset statistics, results/instruct_sft_source_shares.json).
Writes results/sft_lengths_summary.md."""
import json, statistics

d = json.load(open("results/sft_lengths.json"))
shares = json.load(open("results/instruct_sft_source_shares.json"))


def wquantile(vals, weights, q):
    pairs = sorted(zip(vals, weights)); tot = sum(w for _, w in pairs); acc = 0
    for v, w in pairs:
        acc += w
        if acc >= q * tot:
            return v


lines = ["# Answer length the SFT data teaches (Olmo 3 tokenizer, answer after the last `</think>`)", "",
         "Random 100-row pages through the Hugging Face datasets-server; the sets are stored by source, so each page is one source and the sample is `pages` clusters. For Dolci-Instruct-SFT the whole-set source shares are known exactly and used as weights.", "",
         "| set | pages | rows | sources hit | pooled median tokens | pooled IQR | weighted median (Instruct) | thinking median tokens |", "|---|---|---|---|---|---|---|---|"]
for ds, r in d.items():
    rows = r["rows"]; toks = [t for _, t, _ in rows]
    by = {}
    for s, t, _ in rows:
        by.setdefault(s, []).append(t)
    wmed = ""
    if "Instruct" in ds:
        w = [shares.get(s, 0) / len(by[s]) for s, t, _ in rows]
        wmed = f"{wquantile(toks, w, .5):.0f} (q25 {wquantile(toks, w, .25):.0f}, q75 {wquantile(toks, w, .75):.0f})"
    lines.append(f"| {ds.split('/')[-1]} | {r['pages']} | {len(rows):,} | {len(by)} | {statistics.median(toks):.0f} | {r['answer_tokens_q25']}–{r['answer_tokens_q75']} | {wmed or '–'} | {r['thinking_tokens_median'] if r['with_thinking'] > 1 else '–'} |")
lines += ["", "## Per-source answer medians (tokens)", ""]
for ds, r in d.items():
    by = {}
    for s, t, _ in r["rows"]:
        by.setdefault(s, []).append(t)
    lines += [f"### {ds.split('/')[-1]}", "", "| source | sampled rows | median tokens | exact share (Instruct only) |", "|---|---|---|---|"]
    for s, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        sh = f"{shares[s]*100:.1f}%" if s in shares else "–"
        lines.append(f"| {s} | {len(v):,} | {statistics.median(v):.0f} | {sh} |")
    lines.append("")
lines += ["Reference, DPO answer tokens (results/tables.md): Think-DPO-7B chosen 527 / rejected 254; Think-DPO-32B 560 / 258; Instruct-DPO 333 / 256.", ""]
open("results/sft_lengths_summary.md", "w").write("\n".join(lines))
print("\n".join(lines[:12]))
