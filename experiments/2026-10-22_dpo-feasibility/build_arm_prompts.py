"""Write arm_<X>_prompts.jsonl (id, source, prompt) for an arm from its id list and the public dataset
allenai/Dolci-Think-DPO-7B, for the prompt-overlap (contamination) check against the evaluation sets.
The prompt is the last user turn of the chosen conversation, as in the arm_A/arm_B prompt files taken from the node.
Usage: build_arm_prompts.py ARM_IDS.json OUT.jsonl    (needs network for the first download; pandas + pyarrow)"""
import glob, json, os, sys

import pandas as pd
from huggingface_hub import snapshot_download

ids_path, out_path = sys.argv[1], sys.argv[2]
ids = json.load(open(ids_path))
want = set(ids)
local = snapshot_download("allenai/Dolci-Think-DPO-7B", repo_type="dataset")
files = sorted(glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True))
print(len(files), "parquet files under", local, flush=True)
rows = {}
for f in files:
    df = pd.read_parquet(f)
    sub = df[df["id"].isin(want)]
    for _, r in sub.iterrows():
        users = [m["content"] for m in r["chosen"] if m.get("role") == "user"]
        rows[r["id"]] = {"id": r["id"], "source": r.get("source"), "prompt": users[-1] if users else ""}
    print(os.path.basename(f), len(df), "rows,", len(rows), "collected", flush=True)
missing = [i for i in ids if i not in rows]
with open(out_path, "w") as fo:
    for i in ids:
        if i in rows:
            fo.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
print(f"written {len(rows)} prompts to {out_path}; missing ids: {len(missing)} {missing[:5]}", flush=True)
