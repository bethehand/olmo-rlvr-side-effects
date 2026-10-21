"""Compare the seed_check.py outputs pairwise: how many of the prompts got identical text."""
import json
import sys
from itertools import combinations

runs = {}
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    runs[path.rsplit("/", 1)[-1].removesuffix(".json")] = d
names = list(runs)
n = len(next(iter(runs.values()))["outputs"])
print(f"{n} prompts; identical outputs per pair:")
for a, b in combinations(names, 2):
    same = sum(x == y for x, y in zip(runs[a]["outputs"], runs[b]["outputs"]))
    print(f"  {a:>8} vs {b:<8} seeds {runs[a]['seed']!s:>4} / {runs[b]['seed']!s:<4}: {same}/{n} identical")
