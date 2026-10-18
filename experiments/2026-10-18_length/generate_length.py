"""Answer length on AlpacaEval (805 prompts), for open-instruct #2002 (Olmo 3.1 verbosity).
Follows OLMES `alpaca_eval_v3::olmo3:adapt`, the setting Ai2 used for Olmo 3: the model's own chat template,
temperature 0.6, top_p 0.95, one sample per prompt, no stop sequences. OLMES lets generation run to the end of the
context window; we cap it at 32,768 new tokens (the cap Ai2's safety-eval uses for Think models), because on
2x24 GB cards a few runaway loops otherwise hold the KV cache for hours (see README). Outputs that hit the cap are
counted as runaway and reported separately. Following OLMES's r1_style processor, the answer is the text after
the last </think> (the whole output when there is none). Each finished prompt is written at once, and a rerun
with the same --out skips prompts already written."""
import argparse, json, os, re, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompts", required=True, help="alpaca_eval.json from tatsu-lab/alpaca_eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument("--template_from", default=None, help="use this model's chat template (and so its default system prompt), for the system-prompt control")
    ap.add_argument("--dtype", default="auto", help="bfloat16 for checkpoints stored in fp32 (3.1 32B Instruct SFT/DPO), which don't fit 4x24 GB otherwise")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = json.load(open(args.prompts))[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.template_from:
        tok.chat_template = AutoTokenizer.from_pretrained(args.template_from).chat_template
    texts = [tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True) for r in rows]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["index"] for l in open(args.out) if l.strip()}
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"{len(done)} already written, {len(todo)} to go", flush=True)
    if not todo:
        return
    # 0.85 instead of vLLM's default 0.9: on 24 GB cards the default leaves too little room for vLLM's start-up
    # sampler warm-up (OOM). Memory headroom only; sampling is unchanged.
    llm = LLM(model=args.model, revision=args.revision, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85, dtype=args.dtype)
    window = llm.llm_engine.model_config.max_model_len
    engine = llm.llm_engine
    for i in todo:
        n = len(tok(texts[i], add_special_tokens=False)["input_ids"])
        engine.add_request(str(i), texts[i], SamplingParams(temperature=0.6, top_p=0.95, max_tokens=min(args.max_new_tokens, window - n - 16)))
    t0, k = time.time(), 0
    with open(args.out, "a") as f:
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                i = int(o.request_id); r = rows[i]; c = o.outputs[0]; text = c.text
                answer = re.sub(r"(?ms).*</think>", "", text)
                answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer); answer = re.sub(r"(?ms)</answer>\s*$", "", answer)
                f.write(json.dumps({"index": i, "dataset": r.get("dataset"), "instruction": r["instruction"],
                                    "model": args.model, "revision": args.revision, "response": text,
                                    "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids), "has_think_end": "</think>" in text,
                                    "answer_chars": len(answer), "answer_tokens": len(tok(answer, add_special_tokens=False)["input_ids"]),
                                    "max_new_tokens": args.max_new_tokens, "dtype": args.dtype, "template_from": args.template_from}) + "\n")
                f.flush(); k += 1
                if k % 25 == 0 or k == len(todo):
                    print(f"{k}/{len(todo)} done in {time.time() - t0:.0f}s", flush=True)
    print(f"{args.model}@{args.revision}: {len(todo)} prompts in {time.time() - t0:.0f}s, window {window}, cap {args.max_new_tokens}", flush=True)

if __name__ == "__main__":
    main()
