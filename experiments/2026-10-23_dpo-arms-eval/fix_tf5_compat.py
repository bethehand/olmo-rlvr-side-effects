"""Make a model saved by transformers 5.x loadable, with the right settings, by transformers 4.57 / vLLM 0.11 (GS01's
evaluation environment). Works on the files in the local hub cache for REPO and writes through the symlinks into the blobs.
Two incompatibilities were found on the arms (saved by transformers 5.4):
  tokenizer_config.json  tokenizer_class "TokenizersBackend" does not exist in 4.57 -> "PreTrainedTokenizerFast"
                         (the tokenizer is otherwise identical to Think-SFT's: same vocabulary, added tokens, encodings, eos, pad).
  config.json            RoPE settings live under "rope_parameters" (with rope_theta inside). 4.57's Olmo3Config ignores that key
                         and silently falls back to its defaults, rope_theta 10000 and no scaling, instead of 500000 with YaRN
                         (factor 8, original 8192, attention_factor 1.208). vLLM 0.11 reads config.rope_theta / rope_scaling, so the
                         model was served with the wrong positional encoding (found 2026-10-23 after arm A's outputs degenerated into
                         repetition on long prompts). Fix: write "rope_scaling" (without rope_theta) and a top-level "rope_theta",
                         as in the start model's config, and mirror "dtype" into "torch_dtype".
Idempotent; prints what it changed. With --check_against START_MODEL it then loads both configs and tokenizers with the installed
transformers and exits 1 if anything that must match the start model differs (RoPE, context, layer types, vocabulary, eos / pad,
encodings), so a silent fallback like the one above cannot pass unnoticed again.
Usage: fix_tf5_compat.py REPO [--cache DIR] [--check_against allenai/Olmo-3-7B-Think-SFT]"""
import argparse, glob, json, os


def rewrite(path, fn):
    real = os.path.realpath(path)
    d = json.load(open(real))
    changed = fn(d)
    if changed:
        json.dump(d, open(real, "w"), indent=2, ensure_ascii=False)
    return changed


def fix_tokenizer(d):
    if d.get("tokenizer_class") == "TokenizersBackend":
        d["tokenizer_class"] = "PreTrainedTokenizerFast"
        return ["tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast"]
    return []


def fix_config(d):
    out = []
    rp = d.get("rope_parameters")
    if isinstance(rp, dict) and "rope_scaling" not in d:
        rp = dict(rp)
        theta = rp.pop("rope_theta", None)
        if theta is not None and "rope_theta" not in d:
            d["rope_theta"] = theta
            out.append(f"rope_theta {theta} moved to the top level")
        if rp.get("rope_type", "default") != "default":
            d["rope_scaling"] = rp
            out.append(f"rope_scaling written from rope_parameters ({rp.get('rope_type')}, factor {rp.get('factor')})")
        # rope_parameters is left in place for transformers 5.x readers
    if "dtype" in d and "torch_dtype" not in d:
        d["torch_dtype"] = d["dtype"]
        out.append(f"torch_dtype {d['dtype']} mirrored from dtype")
    return out


def check(repo, start):
    from transformers import AutoConfig, AutoTokenizer
    a, b = AutoConfig.from_pretrained(repo), AutoConfig.from_pretrained(start)
    bad = []
    for k in ("rope_theta", "rope_scaling", "max_position_embeddings", "sliding_window", "layer_types", "hidden_size",
              "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "torch_dtype", "model_type", "architectures"):
        if getattr(a, k, None) != getattr(b, k, None):
            bad.append(f"config.{k}: {getattr(a, k, None)!r} vs {getattr(b, k, None)!r}")
    ta, tb = AutoTokenizer.from_pretrained(repo), AutoTokenizer.from_pretrained(start)
    for k in ("eos_token_id", "pad_token_id", "bos_token_id"):
        if getattr(ta, k, None) != getattr(tb, k, None):
            bad.append(f"tokenizer.{k}: {getattr(ta, k, None)!r} vs {getattr(tb, k, None)!r}")
    if len(ta) != len(tb):
        bad.append(f"tokenizer size {len(ta)} vs {len(tb)}")
    probe = "<|im_start|>user\nHow many days are in a leap year? 1,234 + 5 = <think>x</think><|im_end|>\n<|endoftext|>"
    if ta(probe, add_special_tokens=False)["input_ids"] != tb(probe, add_special_tokens=False)["input_ids"]:
        bad.append("tokenizer encodes the probe string differently")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--cache", default=os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"))
    ap.add_argument("--check_against", default=None, help="start model whose loaded config / tokenizer the repo must match")
    args = ap.parse_args()
    base = os.path.join(args.cache, "models--" + args.repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    if not snaps:
        raise SystemExit(f"no snapshot under {base}")
    for s in snaps:
        for name, fn in (("tokenizer_config.json", fix_tokenizer), ("config.json", fix_config)):
            p = os.path.join(s, name)
            if os.path.exists(p):
                ch = rewrite(p, fn)
                print(f"{p}: " + ("; ".join(ch) if ch else "nothing to change"))
    if args.check_against:
        bad = check(args.repo, args.check_against)
        if bad:
            raise SystemExit("MISMATCH with the start model: " + "; ".join(bad))
        print(f"check against {args.check_against}: loaded config and tokenizer match")


if __name__ == "__main__":
    main()
