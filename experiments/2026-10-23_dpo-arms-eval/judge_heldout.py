"""Judge the held-out outputs with a local Qwen3-32B (vLLM, non-thinking mode, greedy), the judge Ai2 uses in its RL-Zero
General recipe. Two kinds of judgment, all requests batched in one engine session:

  pairwise  for each --pair X:Y and each prompt, which answer is better, asked in both orders (X shown first, then Y first).
            The instruction tells the judge to score correctness, completeness, usefulness and clarity and not to prefer length
            as such. A model wins a prompt only if it wins in both orders; anything else is a tie.
  rubric    for prompts of the detail category, for each file: which rubric points the answer covers (yes/no per point).

Answers are the text after the last </think>; answers longer than --max_answer_tokens are cut (noted in the record).
Usage (GS01, safety-eval venv, 4 GPUs):
  judge_heldout.py --pairs X.jsonl:Y.jsonl[,X.jsonl:Z.jsonl] [--rubric X.jsonl Y.jsonl ...] --out judgments.jsonl
"""
import argparse, json, os, re, time

PAIR_SYS = ("You are a careful evaluator comparing two assistant responses to the same user request. Judge which response "
            "serves the user better on correctness, completeness, usefulness and clarity. Length is not a merit in itself: do not "
            "prefer a response because it is longer or more elaborate, and do not prefer a response because it is shorter. Extra "
            "material the user did not ask for counts against a response only when it makes the answer harder to use; missing "
            "content the user needs counts against a response. Answer with a JSON object only.")
PAIR_USER = ("User request:\n{prompt}\n\n[Response 1]\n{a}\n\n[Response 2]\n{b}\n\n"
             "Which response is better? Reply with JSON: {{\"winner\": \"1\" or \"2\" or \"tie\", \"reason\": \"one or two sentences\"}}")
RUBRIC_SYS = ("You are a careful grader. You get a user request, a list of points a complete answer should cover, and a response. "
              "For each point, decide whether the response covers it in substance (wording may differ). Answer with a JSON object only.")
RUBRIC_USER = ("User request:\n{prompt}\n\nPoints:\n{points}\n\n[Response]\n{a}\n\n"
               "Reply with JSON: {{\"covered\": [true/false for each point, in order], \"note\": \"optional short note\"}}")


def load(path):
    return {json.loads(l)["index"]: json.loads(l) for l in open(path) if l.strip()}


def first_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="", help="comma-separated X.jsonl:Y.jsonl")
    ap.add_argument("--rubric", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen3-32B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=20000)
    ap.add_argument("--max_answer_tokens", type=int, default=6000)
    ap.add_argument("--quantization", default=None, help="fp8 if bf16 does not fit")
    ap.add_argument("--max_num_seqs", type=int, default=64, help="64 for a 32B judge on 4x24 GB: vLLM's default 256 ran out of memory in the sampler warm-up at start-up (2026-10-25)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge)

    def cut(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= args.max_answer_tokens:
            return text, False
        return tok.decode(ids[: args.max_answer_tokens]) + "\n[... cut for the judge ...]", True

    reqs = []  # (meta, chat text)
    def add(meta, system, user):
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        reqs.append((meta, text))

    for pair in [p for p in args.pairs.split(",") if p]:
        fx, fy = pair.split(":")
        X, Y = load(fx), load(fy)
        for i in sorted(set(X) & set(Y)):
            ax, cx = cut(X[i]["answer"]); ay, cy = cut(Y[i]["answer"])
            base = {"kind": "pairwise", "index": i, "id": X[i]["id"], "category": X[i]["category"], "x": fx, "y": fy, "cut": [cx, cy]}
            add({**base, "order": "xy"}, PAIR_SYS, PAIR_USER.format(prompt=X[i]["prompt"], a=ax, b=ay))
            add({**base, "order": "yx"}, PAIR_SYS, PAIR_USER.format(prompt=X[i]["prompt"], a=ay, b=ax))
    for f in args.rubric:
        R = load(f)
        for i, r in sorted(R.items()):
            if r.get("category") != "detail" or not r.get("rubric"):
                continue
            a, c = cut(r["answer"])
            pts = "\n".join(f"{k + 1}. {p}" for k, p in enumerate(r["rubric"]))
            add({"kind": "rubric", "index": i, "id": r["id"], "file": f, "n_points": len(r["rubric"]), "cut": c},
                RUBRIC_SYS, RUBRIC_USER.format(prompt=r["prompt"], points=pts, a=a))
    print(f"{len(reqs)} judge requests", flush=True)
    extra = {"quantization": args.quantization} if args.quantization else {}
    llm = LLM(model=args.judge, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, **extra)
    sp = SamplingParams(temperature=0.0, max_tokens=400)
    t0 = time.time()
    outs = llm.generate([t for _, t in reqs], sp)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bad = 0
    with open(args.out, "w") as f:
        for (meta, _), o in zip(reqs, outs):
            raw = o.outputs[0].text
            j = first_json(raw)
            rec = dict(meta); rec["raw"] = raw
            if meta["kind"] == "pairwise":
                w = str(j.get("winner", "")).strip().lower() if j else ""
                if w not in ("1", "2", "tie"):
                    bad += 1; w = "unparsed"
                # map to model names: in order xy, 1 = x; in order yx, 1 = y
                rec["winner"] = {"1": "x" if meta["order"] == "xy" else "y", "2": "y" if meta["order"] == "xy" else "x"}.get(w, w)
                rec["reason"] = (j or {}).get("reason", "")
            else:
                cov = (j or {}).get("covered")
                if not isinstance(cov, list) or len(cov) != meta["n_points"]:
                    bad += 1; cov = None
                rec["covered"] = cov; rec["note"] = (j or {}).get("note", "")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(reqs)} judgments to {args.out} in {time.time() - t0:.0f}s; unparsed: {bad}", flush=True)


if __name__ == "__main__":
    main()
