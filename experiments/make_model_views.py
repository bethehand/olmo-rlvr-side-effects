"""Olmo 3 checkpoints on Hugging Face carry a transformers-4.57 style YaRN config (rope_scaling with integer
beta_fast/beta_slow, rope_theta outside it). transformers 5.x rejects that. This builds a folder per checkpoint
with links to the cached files and a config.json in the 5.x form (same numbers), leaving the HF cache untouched.
Also maps the pre-release model type "olmo2-retrofit" (RL-Zero General) to "olmo3"; every other field is identical.
Usage: python make_model_views.py repo@rev [repo@rev ...]   (writes to $MODEL_VIEWS or ./model_views)"""
import json, os, sys
from huggingface_hub import snapshot_download
V = os.environ.get("MODEL_VIEWS", "model_views")
for spec in sys.argv[1:]:
    repo, rev = spec.split("@")
    snap = snapshot_download(repo, revision=rev, local_files_only=True)
    out = os.path.join(V, f"{repo.split('/')[1]}@{rev}")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(snap):
        dst = os.path.join(out, f)
        if f != "config.json" and not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(snap, f)), dst)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type") == "yarn":
        rs = dict(cfg.pop("rope_scaling"))
        rs["rope_theta"] = float(cfg["rope_theta"])
        for k in ("beta_fast", "beta_slow", "factor"):
            rs[k] = float(rs[k])
        cfg["rope_parameters"] = rs
    if cfg.get("model_type") == "olmo2-retrofit":   # pre-release name of the Olmo 3 architecture (RL-Zero General)
        cfg["model_type"] = "olmo3"; cfg["architectures"] = ["Olmo3ForCausalLM"]
        cfg["dtype"] = cfg.pop("torch_dtype", "bfloat16"); cfg["use_cache"] = True
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print("view ready:", out)
