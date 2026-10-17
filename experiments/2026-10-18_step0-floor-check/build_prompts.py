"""Step 0 prompt set: 100 harmful + 100 benign-but-scary prompts, fixed seed.
harmful: 50 HarmBench standard behaviors + 50 WildJailbreak harmful (adversarial form)
benign:  50 XSTest safe prompts + 50 WildJailbreak benign (adversarial form)
Data comes from allenai/safety-eval (the files Ai2's own safety evals use)."""
import csv, json, random, sys
SE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme0/looped/oi_work/safety-eval/evaluation/tasks/generation"
out = sys.argv[2] if len(sys.argv) > 2 else "prompts.jsonl"
rng = random.Random(0)
rows = []
hb = [r for r in csv.DictReader(open(f"{SE}/harmbench/harmbench_behaviors_text_test.csv")) if r["FunctionalCategory"] == "standard"]
rows += [{"id": f"harmbench/{r['BehaviorID']}", "source": "harmbench", "kind": "harmful", "prompt": r["Behavior"]} for r in rng.sample(hb, 50)]
wjh = [json.loads(l) for l in open(f"{SE}/wildjailbreak/harmful.jsonl")]
rows += [{"id": f"wj_harmful/{i}", "source": "wj_harmful", "kind": "harmful", "prompt": wjh[i]["adversarial"]} for i in sorted(rng.sample(range(len(wjh)), 50))]
xs = [r for r in json.load(open(f"{SE}/xstest/exaggerated_safety.json")) if not r["type"].startswith("contrast")]
rows += [{"id": f"xstest/{r['id']}", "source": "xstest_safe", "kind": "benign", "prompt": r["prompt"]} for r in rng.sample(xs, 50)]
wjb = [json.loads(l) for l in open(f"{SE}/wildjailbreak/benign.jsonl")]
rows += [{"id": f"wj_benign/{i}", "source": "wj_benign", "kind": "benign", "prompt": wjb[i]["adversarial"]} for i in sorted(rng.sample(range(len(wjb)), 50))]
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} prompts to {out}")
