"""Apply a smaller generation cap to finished outputs after the fact. The baselines were generated with the 32K cap and the arms
with 8K (protocol change of 2026-10-23); to compare them, a baseline output that ran past the cap is cut to its first --cap
tokens (the first N sampled tokens do not depend on the cap), its finish_reason becomes "length", and answer / answer_chars /
answer_tokens / has_think_end are recomputed as generate_prompts.py and generate_length.py compute them. Rows at or under the
cap (by the recorded n_tokens) are copied unchanged. The cut re-tokenizes the text, which can differ from the sampled token
boundaries by a token or so; n_tokens is set to the cap.
Usage: truncate_to_cap.py IN.jsonl OUT.jsonl [--cap 8192] [--tokenizer MODEL]   (default tokenizer: the model named in the rows)"""
import argparse, json, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or rows[0]["model"], revision=None if args.tokenizer else rows[0].get("revision", "main"))
    cut = 0
    with open(args.out, "w") as f:
        for r in rows:
            if r["n_tokens"] > args.cap:
                ids = tok(r["response"], add_special_tokens=False)["input_ids"][:args.cap]
                text = tok.decode(ids)
                answer = re.sub(r"(?ms).*</think>", "", text)
                answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer); answer = re.sub(r"(?ms)</answer>\s*$", "", answer)
                r.update({"response": text, "finish_reason": "length", "n_tokens": args.cap, "has_think_end": "</think>" in text,
                          "answer": answer, "answer_chars": len(answer),
                          "answer_tokens": len(tok(answer, add_special_tokens=False)["input_ids"]),
                          "max_new_tokens": args.cap, "capped_after_the_fact_from": r["max_new_tokens"]})
                cut += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.inp}: {len(rows)} rows, {cut} cut to {args.cap} tokens -> {args.out}")


if __name__ == "__main__":
    main()
