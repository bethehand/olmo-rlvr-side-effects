"""Identity, cutoff and link probe (open-instruct #2001, #1948, #1964, #1902).
Same engine and sampling as ../2026-10-18_length/generate_length.py (vLLM 0.11.0, temperature 0.6, top_p 0.95), so the
numbers line up with the AlpacaEval runs. Every prompt is asked under up to four system prompts (CONDITIONS), --n samples
each. "default" is the model's own chat template with no system message (what a plain API call gets). The other
conditions keep that rendering and swap only the identity sentence of its system block, leaving the template's own
framing in place (the Olmo templates append "You do not currently have access to any functions..." to some system
prompts and not others, and an explicit system message would change that). So "olmo" on a Think stage is byte-identical
to the 3.1 Think release template, and on an Instruct stage to the 3.1 Instruct one. A condition whose rendering equals
an earlier one's (e.g. "olmo" on the 3.1 release models, "generic" on the Olmo 3 Think stages) is skipped and listed in
the .meta.json. The answer is the text after the last </think> (OLMES r1_style). Resumable: a rerun with the same --out
skips rows already written."""
import argparse, json, os, re, time

RELEASE = ("You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, "
           "and your model weights are available at https://huggingface.co/allenai.")
CONDITIONS = {"default": None, "olmo": RELEASE, "generic": "You are a helpful AI assistant.",
              "robot": "You are Robot, a helpful AI assistant built by Acme Corp."}
MARK = "PROBE_USER_TEXT_7f3a"
SYS_RE = re.compile(r"(<\|im_start\|>system\n)(.*?)(<\|im_end\|>)", re.S)
FUNC_MARK = " You do not currently have access to any functions"

def render(tok, system, user):
    """The model's own default rendering of `user`; with `system`, the same rendering with the identity sentence of the
    system block replaced and the template's own suffix (if any) kept."""
    r0 = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    if system is None:
        return r0
    m = SYS_RE.search(r0)
    if not m:  # a template without a default system block: fall back to an explicit system message
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    s0 = m.group(2); i = s0.find(FUNC_MARK); suffix = s0[i:] if i >= 0 else ""
    return r0[:m.start(2)] + system + suffix + r0[m.end(2):]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompts", required=True, help="prompts.json: [{id, category, text}, ...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3, help="samples per prompt and condition")
    ap.add_argument("--conditions", default="default,olmo,generic,robot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_num_seqs", type=int, default=None)
    ap.add_argument("--dtype", default="auto", help="bfloat16 for the fp32 checkpoints (3.1 32B Instruct SFT/DPO)")
    ap.add_argument("--max_model_len", type=int, default=None, help="shorten the 65K window so a 7B fits one 24 GB card (KV cache for one 65K sequence needs 12.5 GiB); 20480 leaves room for the 16K cap")
    ap.add_argument("--print_only", action="store_true", help="print each condition's rendering of the first prompt and exit (no GPU)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    prompts = json.load(open(args.prompts))[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    conds = [c for c in args.conditions.split(",") if c]
    rendered = {c: render(tok, CONDITIONS[c], MARK) for c in conds}
    kept, skipped = [], {}
    for c in conds:
        same = [k for k in kept if rendered[k] == rendered[c]]
        if same:
            skipped[c] = same[0]; print(f"'{c}' renders exactly like '{same[0]}' for this model; skipping it", flush=True)
        else:
            kept.append(c)
    conds = kept
    meta = {"model": args.model, "revision": args.revision, "conditions": {c: CONDITIONS[c] for c in conds}, "skipped": skipped,
            "rendered_prefix": {c: rendered[c].split(MARK)[0] for c in conds},
            "rendered_suffix": rendered[conds[0]].split(MARK)[1], "n": args.n, "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype, "seed": args.seed, "max_model_len": args.max_model_len, "temperature": 0.6, "top_p": 0.95}
    if args.print_only:
        for c in conds:
            print(f"----- {c}\n{render(tok, CONDITIONS[c], prompts[0]['text'])}")
        print("skipped:", skipped)
        return
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1, ensure_ascii=False)
    done = set()
    if os.path.exists(args.out):
        done = {(r["id"], r["condition"], r["sample"]) for r in map(json.loads, filter(str.strip, open(args.out)))}
    todo = [(p, c, k) for p in prompts for c in conds for k in range(args.n) if (p["id"], c, k) not in done]
    print(f"{len(done)} already written, {len(todo)} to go", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    extra = {"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}
    if args.max_model_len:
        extra["max_model_len"] = args.max_model_len
    llm = LLM(model=args.model, revision=args.revision, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              dtype=args.dtype, seed=args.seed, **extra)
    engine = llm.llm_engine
    window = engine.model_config.max_model_len
    byid, text_of = {}, {}
    for p, c, k in todo:
        text = render(tok, CONDITIONS[c], p["text"])
        rid = f"{p['id']}|{c}|{k}"; byid[rid] = (p, c, k); text_of[rid] = text
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        engine.add_request(rid, text, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=min(args.max_new_tokens, window - n - 16)))
    t0, written = time.time(), 0
    with open(args.out, "a") as f:
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                p, c, k = byid[o.request_id]; out = o.outputs[0]; text = out.text
                answer = re.sub(r"(?ms).*</think>", "", text)
                answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer); answer = re.sub(r"(?ms)</answer>\s*$", "", answer)
                f.write(json.dumps({"id": p["id"], "category": p["category"], "prompt": p["text"], "condition": c,
                                    "system": (SYS_RE.search(text_of[o.request_id]).group(2) if SYS_RE.search(text_of[o.request_id]) else CONDITIONS[c]), "sample": k, "model": args.model, "revision": args.revision,
                                    "response": text, "answer": answer.strip(), "finish_reason": out.finish_reason,
                                    "n_tokens": len(out.token_ids), "has_think_end": "</think>" in text}, ensure_ascii=False) + "\n")
                f.flush(); written += 1
                if written % 50 == 0 or written == len(todo):
                    print(f"{written}/{len(todo)} done in {time.time() - t0:.0f}s", flush=True)
    print(f"{args.model}@{args.revision}: {len(todo)} generations in {time.time() - t0:.0f}s, window {window}, cap {args.max_new_tokens}", flush=True)

if __name__ == "__main__":
    main()
