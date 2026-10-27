"""Sanity check of the shard sample: per-source share in the sampled rows versus the prompt counts on the dataset card.
Clean-v1 is compared with the Dolci-Think-SFT-32B card (its source, 2,253,684 rows; Clean-v1 keeps 2,190,070 of them), the 7B sample
with the Dolci-Think-SFT-7B card (2,268,178 rows). Card labels are matched to the parquet source strings by keyword.
Usage: composition_check.py rows.parquet {32b|7b} OUT.txt"""
import sys, time
import pyarrow.parquet as pq
from collections import Counter

CARDS = {
    "32b": (2253684, {  # Dolci-Think-SFT-32B README
        "OpenThoughts3": ("OpenThoughts 3 (Math+STEM+Code)", 941164), "SYNTHETIC": ("SYNTHETIC-2 SFT-Verified", 104568),
        "Nemotron": ("Nemotron code", 113777), "Persona": ("Dolci Think Persona IF", 220530), "Precise IF": ("Dolci Precise IF", 135722),
        "Python": ("Dolci Think Python", 466676), "WildChat": ("WildChat", 76209), "OpenAssistant": ("OpenAssistant", 6647),
        "CoCoNot": ("CoCoNot", 9549), "WildGuard": ("WildGuardMix", 36673), "WildJailbreak": ("WildJailbreak", 40002),
        "Aya": ("Aya", 97156), "TableGPT": ("TableGPT", 4973), "Identity": ("Olmo Identity", 58)}),
    "7b": (2268178, {  # Dolci-Think-SFT-7B README
        "OpenThoughts3": ("OpenThoughts 3 (Math+STEM+Code)", 941166), "SYNTHETIC": ("SYNTHETIC-2 SFT-Verified", 104569),
        "Nemotron": ("Nemotron code", 113777), "Persona": ("Dolci Think Persona IF", 223123), "Precise IF": ("Dolci Precise IF", 135792),
        "Python": ("Dolci Think Python", 466677), "WildChat": ("WildChat", 83054), "OpenAssistant": ("OpenAssistant", 6800),
        "CoCoNot": ("CoCoNot", 10227), "WildGuard": ("WildGuardMix", 38315), "WildJailbreak": ("WildJailbreak", 41100),
        "Aya": ("Aya", 98597), "TableGPT": ("TableGPT", 4981), "Identity": ("Olmo Identity", 58)}),
}


def match(src):
    s = src.lower()
    if "persona" in s: return "Persona"
    if "precise" in s or "precise-if" in s or "if_qwq" in s: return "Precise IF"   # the 7B set names Dolci Precise IF "if_qwq_reasoning_verified..."
    if "python" in s: return "Python"
    if "oasst" in s or "openassistant" in s: return "OpenAssistant"
    for k in ("OpenThoughts3", "SYNTHETIC", "Nemotron", "WildChat", "CoCoNot", "WildGuard", "WildJailbreak", "Aya", "TableGPT", "Identity"):
        if k.lower() in s: return k
    return None


def main():
    rows_path, card, out = sys.argv[1], sys.argv[2], sys.argv[3]
    total, labels = CARDS[card]
    cnt = Counter(pq.read_table(rows_path, columns=["source"]).column("source").to_pylist()); n = sum(cnt.values())
    agg, unmatched = Counter(), Counter()
    for src, c in cnt.items():
        k = match(src)
        (agg.__setitem__(k, agg[k] + c) if k else unmatched.__setitem__(src, c))
    lines = [f"# {__doc__.strip()}", f"# Produced by composition_check.py on {time.strftime('%Y-%m-%d %H:%M %Z')}",
             f"sampled rows: {n} from {rows_path}; card: Dolci-Think-SFT-{card.upper()} ({total} rows)",
             f"{'card label':36s} {'sample share':>12s} {'card share':>11s} {'sample n':>9s}"]
    for k, (lab, card_n) in labels.items():
        lines.append(f"{lab:36s} {100*agg.get(k,0)/n:11.1f}% {100*card_n/total:10.1f}% {agg.get(k,0):9d}")
    if unmatched: lines.append("unmatched sources: " + str(dict(unmatched)))
    lines.append("raw source counts: " + ", ".join(f"{k}: {v}" for k, v in cnt.most_common()))
    open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
