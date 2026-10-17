"""Generate one answer per prompt with vLLM, using the prompt template the model was trained with
(open-instruct's CHAT_TEMPLATES). Raw outputs stay on the GPU machine; they are not committed."""
import argparse, json, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--model_path", default=None, help="local folder to load from (a model view); --model/--revision are kept as labels")
    ap.add_argument("--template", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--open_instruct", default="/mnt/nvme0/looped/oi_work/open-instruct")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    sys.path.insert(0, args.open_instruct)
    from open_instruct.dataset_transformation import CHAT_TEMPLATES
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    src = dict(model=args.model_path) if args.model_path else dict(model=args.model, revision=args.revision)
    tok = AutoTokenizer.from_pretrained(src["model"], revision=src.get("revision"))
    tok.chat_template = CHAT_TEMPLATES[args.template]
    rows = [json.loads(l) for l in open(args.prompts)]
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
             for r in rows]
    stop = ["</answer>"] if args.template == "olmo_thinker" else None   # the General run trained with this stop string
    llm = LLM(**src, dtype="bfloat16", max_model_len=args.max_tokens + 3072, gpu_memory_utilization=0.92, seed=0)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=0,
                        stop=stop, include_stop_str_in_output=True)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            f.write(json.dumps({**r, "model": args.model, "revision": args.revision, "template": args.template,
                                "response": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)}) + "\n")
    print(f"{args.model}@{args.revision}: {len(rows)} prompts in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
