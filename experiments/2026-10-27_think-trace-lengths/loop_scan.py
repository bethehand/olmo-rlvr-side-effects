"""Do the SFT thinking traces themselves contain repetition loops? For every sampled row (same shards as trace_lengths.py) take the
thinking part of the last assistant message and compute the distinct 8-gram ratio over its last 2,000 words (the measure used on the
H045 logs and on our Olmo 3 outputs: finished traces 0.93-1.00, loops 0.00-0.03), plus the most repeated 8-gram and its count.
Reports per source and overall the share of traces below 0.5 (looping tail) and below 0.8, and writes the flagged rows
(ratio < 0.8) with id, source, ratio, repeat count and the repeated 8-gram to results/loop_flagged_<dataset>.parquet.
Clean-v1's own cleanup dropped 239 "degenerate tail" rows with a C4-style filter, so anything flagged here in Clean-v1 is what that filter kept.
Usage: loop_scan.py DATA_DIR OUT_DIR [--procs 32]"""
import argparse, glob, json, os, time
from collections import Counter
from multiprocessing import Pool

THRESH = (0.5, 0.8)


def tail_ratio(words, n=2000, k=8):
    tail = words[-n:]
    grams = [tuple(tail[i:i + k]) for i in range(max(0, len(tail) - k + 1))]
    if len(grams) < 50:
        return float("nan"), "", 0
    c = Counter(grams)
    top, cnt = c.most_common(1)[0]
    return len(c) / len(grams), " ".join(top), cnt


def process_shard(path):
    import pyarrow.parquet as pq
    src_col = "source" if "source" in pq.read_schema(path).names else "dataset_source"
    t = pq.read_table(path, columns=["id", "messages", src_col])
    out = []
    for rid, msgs, src in zip(t.column("id").to_pylist(), t.column("messages").to_pylist(), t.column(src_col).to_pylist()):
        asst = [m.get("content") or "" for m in msgs if m.get("role") == "assistant"]
        a = asst[-1] if asst else ""
        k = a.rfind("</think>")
        think = a[:k] if k >= 0 else a
        words = think.split()
        r, top, cnt = tail_ratio(words)
        out.append({"id": rid, "source": src, "words": len(words), "ratio": r, "top_count": cnt, "top_gram": top if r == r and r < THRESH[1] else ""})
    print(time.strftime('%H:%M:%S'), os.path.basename(path), len(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("out_dir"); ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    import pyarrow as pa, pyarrow.parquet as pq
    lines = ["# " + __doc__.strip().replace("\n", "\n# "), "# Produced by loop_scan.py on " + time.strftime("%Y-%m-%d %H:%M %Z")]
    summary = {}
    for ds in sorted(os.listdir(args.data_dir)):
        shards = sorted(glob.glob(os.path.join(args.data_dir, ds, "**", "*.parquet"), recursive=True))
        if not shards:
            continue
        with Pool(min(args.procs, len(shards))) as pool:
            rows = [r for part in pool.map(process_shard, shards) for r in part]
        valid = [r for r in rows if r["ratio"] == r["ratio"]]
        flagged = [r for r in valid if r["ratio"] < THRESH[1]]
        pq.write_table(pa.Table.from_pylist(flagged or [{"id": "", "source": "", "words": 0, "ratio": 0.0, "top_count": 0, "top_gram": ""}][:0]), os.path.join(args.out_dir, f"loop_flagged_{ds}.parquet")) if flagged else None
        lines.append(f"\n== {ds}: {len(rows)} rows, {len(valid)} with a tail of at least 57 words (shorter traces skipped)")
        lines.append(f"{'source':60s} {'n':>7s} {'<0.5':>14s} {'<0.8':>14s} {'median ratio':>12s}")
        by = {}
        for r in valid:
            by.setdefault(r["source"], []).append(r)
        def row(label, g):
            n = len(g); a = sum(r["ratio"] < THRESH[0] for r in g); b = sum(r["ratio"] < THRESH[1] for r in g)
            med = sorted(r["ratio"] for r in g)[n // 2] if n else float("nan")
            return f"{label[:60]:60s} {n:7d} {a:5d} ({100*a/n:5.2f}%) {b:5d} ({100*b/n:5.2f}%) {med:12.3f}"
        for src, g in sorted(by.items(), key=lambda kv: -len(kv[1])):
            lines.append(row(src, g))
        lines.append(row("ALL", valid))
        worst = sorted(flagged, key=lambda r: r["ratio"])[:8]
        lines.append("most repetitive traces (id | source | words | ratio | repeats | 8-gram):")
        for r in worst:
            lines.append(f"  {r['id'][:40]:40s} | {r['source'][:28]:28s} | {r['words']:6d} | {r['ratio']:.3f} | {r['top_count']:4d} | {r['top_gram'][:60]}")
        summary[ds] = {"rows": len(rows), "valid": len(valid), "below_0_5": sum(r["ratio"] < 0.5 for r in valid), "below_0_8": len(flagged),
                       "by_source": {s: {"n": len(g), "below_0_5": sum(r["ratio"] < 0.5 for r in g), "below_0_8": sum(r["ratio"] < 0.8 for r in g)} for s, g in by.items()}}
    txt = "\n".join(lines) + "\n"
    open(os.path.join(args.out_dir, "loop_scan.txt"), "w").write(txt)
    json.dump(summary, open(os.path.join(args.out_dir, "loop_scan.json"), "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
