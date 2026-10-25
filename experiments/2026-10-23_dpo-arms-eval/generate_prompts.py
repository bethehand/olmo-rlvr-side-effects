"""Generate answers for one prompt set (held-out set, IFEval, GSM8K) with the same settings as the 805-prompt length runs
(../2026-10-18_length/generate_length.py): the model's own chat template, temperature 0.6, top_p 0.95, one sample per
prompt, generation capped at 32,768 new tokens; the answer is the text after the last </think> (OLMES r1_style).
Each finished prompt is written at once; a rerun with the same --out skips prompts already written.

Prompt sets
  heldout  --prompts heldout_prompts.jsonl (fields id, category, prompt, plus gold / rubric where they exist)
  ifeval   google/IFEval, 541 prompts, the prompt as the user turn (fields key, instruction_id_list, kwargs kept for scoring)
  gsm8k    openai/gsm8k main test, 1,319 problems; the question alone is the user turn, as in Ai2's zero-shot RLVR set
           ai2-adapt-dev/rlvr_gsm8k_zs; the gold answer is the number after '####'
"""
import argparse, json, os, re, time


def load_rows(task, prompts_path, limit):
    if task == "heldout":
        rows = [json.loads(l) for l in open(prompts_path) if l.strip()]
        rows = [{"index": i, "id": r["id"], "category": r["category"], "prompt": r["prompt"],
                 "gold": r.get("gold"), "rubric": r.get("rubric")} for i, r in enumerate(rows)]
    elif task == "ifeval":
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", split="train")
        rows = [{"index": i, "key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                 "kwargs": r["kwargs"]} for i, r in enumerate(ds)]
    elif task == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [{"index": i, "prompt": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(ds)]
    else:
        raise SystemExit(f"unknown task {task}")
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--task", required=True, choices=["heldout", "ifeval", "gsm8k"])
    ap.add_argument("--prompts", default=None, help="heldout_prompts.jsonl (task heldout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=None, help="engine seed; None = vLLM default (0)")
    ap.add_argument("--max_num_seqs", type=int, default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--template_from", default=None, help="use this model's chat template instead of the model's own (control for the template / default system prompt)")
    ap.add_argument("--temperature", type=float, default=0.6, help="sampling temperature (protocol 0.6; 0 = greedy, for the stability diagnostic)")
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = load_rows(args.task, args.prompts, args.limit)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.template_from:
        tok.chat_template = AutoTokenizer.from_pretrained(args.template_from).chat_template
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in rows]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["index"] for l in open(args.out) if l.strip()}
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"{args.task}: {len(rows)} prompts, {len(done)} already written, {len(todo)} to go", flush=True)
    if not todo:
        return
    extra = {"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}
    if args.seed is not None:
        extra["seed"] = args.seed
    llm = LLM(model=args.model, revision=args.revision, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem, dtype=args.dtype, **extra)
    window = llm.llm_engine.model_config.max_model_len
    engine = llm.llm_engine
    for i in todo:
        n = len(tok(texts[i], add_special_tokens=False)["input_ids"])
        engine.add_request(str(i), texts[i], SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=min(args.max_new_tokens, window - n - 16)))
    t0, k = time.time(), 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                i = int(o.request_id); r = rows[i]; c = o.outputs[0]; text = c.text
                answer = re.sub(r"(?ms).*</think>", "", text)
                answer = re.sub(r"(?ms)^\s*<answer>\s*", "", answer); answer = re.sub(r"(?ms)</answer>\s*$", "", answer)
                rec = dict(r)
                rec.update({"task": args.task, "model": args.model, "revision": args.revision, "response": text,
                            "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids), "has_think_end": "</think>" in text,
                            "answer": answer, "answer_chars": len(answer),
                            "answer_tokens": len(tok(answer, add_special_tokens=False)["input_ids"]),
                            "max_new_tokens": args.max_new_tokens, "dtype": args.dtype, "seed": args.seed, "template_from": args.template_from,
                            "temperature": args.temperature, "top_p": args.top_p})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush(); k += 1
                if k % 25 == 0 or k == len(todo):
                    print(f"{k}/{len(todo)} done in {time.time() - t0:.0f}s", flush=True)
    print(f"{args.model}@{args.revision} {args.task}: {len(todo)} prompts in {time.time() - t0:.0f}s, window {window}, cap {args.max_new_tokens}", flush=True)


if __name__ == "__main__":
    main()
