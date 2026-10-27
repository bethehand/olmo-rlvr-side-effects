"""Re-run the 22 assistant turns of open-instruct #2080's chat_logs.md on an Olmo 3 checkpoint, with the exact message context the
issue author sent (for turn 2 that includes H045's own turn-1 answer), the same sampling (temperature 0.6, top_p 0.95,
max_tokens 8192, no system prompt beyond what the model's released chat template adds) and n samples per turn.
Writes one jsonl row per sample: case, turn, sample, h045_status, finish_reason, n_tokens, has_think_end, prompt_tokens, text.
Usage: rerun_h045_turns.py --chat_logs chat_logs.md --model allenai/Olmo-3-7B-Think-SFT --tp 2 --out OUT.jsonl
       [--n 8] [--max_tokens 8192] [--seed 1000] [--gpu_mem 0.85] [--max_num_seqs 64] [--max_model_len 16384] [--limit_turns 2] [--dtype bfloat16]"""
import argparse, json, os, re, time


def parse_turns(path):
    text = open(path, encoding="utf-8").read()
    turns = []
    for case, body in re.findall(r"\n## (\S+) — [^\n]*\n(.*?)(?=\n## \S+ — |\Z)", text, re.S):
        for m in re.finditer(r"### Request for assistant turn (\d+): messages sent\n(.*?)### Assistant turn \1: (answered|UNFINISHED[^\n]*)", body, re.S):
            turn, block, status = int(m.group(1)), m.group(2), m.group(3)
            msgs = [{"role": r, "content": c} for r, c in re.findall(r"\*\*(user|assistant|system):\*\*\n\n````text\n(.*?)\n````", block, re.S)]
            assert msgs and msgs[-1]["role"] == "user", (case, turn)
            turns.append({"case": case, "turn": turn, "messages": msgs, "h045_status": "UNFINISHED" if status.startswith("UNFINISHED") else "answered"})
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat_logs", required=True); ap.add_argument("--model", required=True); ap.add_argument("--revision", default="main")
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=None); ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--limit_turns", type=int, default=None); ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    turns = parse_turns(args.chat_logs)
    if args.limit_turns: turns = turns[:args.limit_turns]
    print(f"{len(turns)} turns parsed ({sum(t['h045_status']=='UNFINISHED' for t in turns)} unfinished for H045)", flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompts = [tok.apply_chat_template(t["messages"], tokenize=False, add_generation_prompt=True) for t in turns]
    print("--- rendered prompt for the first turn (head 600 chars) ---\n" + prompts[0][:600] + "\n--- tail 200 chars ---\n" + prompts[0][-200:] + "\n---", flush=True)
    extra = {"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}
    llm = LLM(model=args.model, revision=args.revision, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem, dtype=args.dtype,
              max_model_len=args.max_model_len, seed=args.seed, **extra)
    sps = [SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed + i) for i in range(len(turns))]
    t0 = time.time()
    outs = llm.generate(prompts, sps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for t, p, o in zip(turns, prompts, outs):
            for k, c in enumerate(o.outputs):
                f.write(json.dumps({"case": t["case"], "turn": t["turn"], "sample": k, "h045_status": t["h045_status"], "model": args.model, "revision": args.revision,
                                    "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids), "has_think_end": "</think>" in c.text,
                                    "prompt_tokens": len(o.prompt_token_ids), "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
                                    "seed": args.seed + turns.index(t), "text": c.text}, ensure_ascii=False) + "\n")
    n = sum(len(o.outputs) for o in outs); cap = sum(c.finish_reason != "stop" for o in outs for c in o.outputs); nothink = sum("</think>" not in c.text for o in outs for c in o.outputs)
    print(f"DONE {args.model}: {n} samples, {cap} hit max_tokens, {nothink} without </think>, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
