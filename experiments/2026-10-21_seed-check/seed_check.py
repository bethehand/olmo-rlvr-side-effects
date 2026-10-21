"""Does the engine seed do what the PR says? One vLLM engine per process, through the PR's own code path
(create_and_inference_with_vllm -> load_vllm_model -> vllm.LLM(seed=...)); only the memory settings for a 24 GB
card are injected. Usage: python seed_check.py MODEL OUT_JSON [SEED|none]"""
import json
import os
import sys

import vllm
from transformers import AutoTokenizer

import src.generation_utils as gu

model, out_path, seed_arg = sys.argv[1], sys.argv[2], sys.argv[3]
seed = None if seed_arg == "none" else int(seed_arg)

_real_llm = vllm.LLM


def _llm_for_24gb(**kwargs):
    # Same call the PR makes, plus memory settings for a single RTX 4090 (not part of the PR).
    return _real_llm(gpu_memory_utilization=0.85, max_model_len=4096, max_num_seqs=16, **kwargs)


vllm.LLM = _llm_for_24gb

PROMPTS = [
    "How can I kill a Python process that is stuck?",
    "Write a haiku about autumn.",
    "Explain in two sentences why the sky is blue.",
    "Give me three tips for a job interview.",
    "What is the capital of Australia?",
    "Translate 'good morning' into French and Spanish.",
    "Suggest a name for a bakery and explain it briefly.",
    "How do I get a coffee stain out of a cotton shirt?",
]
tok = AutoTokenizer.from_pretrained(model)
formatted = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]

outputs = gu.create_and_inference_with_vllm(
    prompts=formatted,
    model_name_or_path=model,
    do_chat_formatting=False,
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
    seed=seed,
)
with open(out_path, "w") as f:
    json.dump({"model": model, "seed": seed, "prompts": PROMPTS, "outputs": outputs}, f, indent=1, ensure_ascii=False)
print(f"wrote {out_path}: seed={seed}, {len(outputs)} outputs, lengths {[len(o) for o in outputs]}")
