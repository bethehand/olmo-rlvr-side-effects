"""Rebuild the training arms on the node from the public dataset and the id lists in ../arms/.

Fallback for when the private dataset repo with the ready-made JSONL files is unavailable or
incomplete. The id lists are what defines an arm (build_arms.py wrote them), so the rebuilt files hold
the same pairs; only the row order differs, which the trainer's seeded shuffle makes irrelevant.
Also writes the 64- and 256-pair timing subsets of A (its eight longest pairs plus a seeded sample).
Usage: python rebuild_arms_from_ids.py OUT_DIR [A B C D ...]   (default: A B)"""
import json, os, random, sys
from datasets import load_dataset

out = sys.argv[1]; arms = sys.argv[2:] or ["A", "B"]
os.makedirs(out, exist_ok=True)
ids_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arms")
ds = load_dataset("allenai/Dolci-Think-DPO-7B", split="train")
by_id = {x: i for i, x in enumerate(ds["id"])}
for name in arms:
    ids = json.load(open(os.path.join(ids_dir, f"arm_{name}_ids.json")))
    sub = ds.select([by_id[i] for i in ids])
    path = os.path.join(out, f"arm_{name}.jsonl")
    sub.to_json(path, lines=True, force_ascii=False)
    print(f"arm {name}: {len(sub)} pairs -> {path}")
if "A" in arms:
    rows = [json.loads(l) for l in open(os.path.join(out, "arm_A.jsonl"))]
    def length(r):
        return sum(len(m.get("content", "")) for m in r["chosen"]) + sum(len(m.get("content", "")) for m in r["rejected"])
    order = sorted(range(len(rows)), key=lambda i: -length(rows[i]))
    longest = order[:8]; rest = [i for i in range(len(rows)) if i not in set(longest)]
    random.seed(1); pick = longest + random.sample(rest, 248)
    for n in (256, 64):
        with open(os.path.join(out, f"arm_A_timing{n}.jsonl"), "w") as f:
            for i in pick[:n]:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    print("timing subsets: arm_A_timing64.jsonl, arm_A_timing256.jsonl")
