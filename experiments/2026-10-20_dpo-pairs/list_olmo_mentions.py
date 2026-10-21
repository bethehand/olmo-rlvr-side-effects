"""Every assistant message in the three Dolci DPO sets that mentions Olmo/Ai2, with context, so the claim
'none is self-identification' can be checked by eye. Uses pyarrow compute for the scan, Python only for the hits."""
import glob, os, re
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
cache = os.environ["HF_HUB_CACHE"]
PAT = r"\bOLMo\b|\bOlmo\b|\bAi2\b|\bAI2\b|Allen Institute"
OLMO = re.compile(PAT)
total = 0
for repo in ("Dolci-Think-DPO-32B", "Dolci-Think-DPO-7B", "Dolci-Instruct-DPO"):
    n_msgs = 0
    for f in sorted(glob.glob(f"{cache}/datasets--allenai--{repo}/snapshots/*/**/*.parquet", recursive=True)):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=5000, columns=["chosen", "rejected"]):
            for side in ("chosen", "rejected"):
                col = batch.column(side)
                flat = pc.list_flatten(col); parents = pc.list_parent_indices(col)
                content = pc.fill_null(flat.field("content"), ""); role = flat.field("role")
                hit = pc.and_(pc.match_substring_regex(content, PAT), pc.equal(role, "assistant"))
                n_msgs += pc.sum(pc.cast(pc.equal(role, "assistant"), pa.int64())).as_py() or 0
                for i in [k for k, h in enumerate(hit.to_pylist()) if h]:
                    text = content[i].as_py(); j = text.rfind("</think>"); ans = text[j + 8:] if j >= 0 else text
                    where = "answer" if OLMO.search(ans) else "thinking"
                    m = OLMO.search(ans) if where == "answer" else OLMO.search(text)
                    seg = ans if where == "answer" else text; k = m.start()
                    total += 1
                    print(f"[{repo} | {side} | {where} | row {parents[i].as_py()}] …{seg[max(0, k-110):k+110].replace(chr(10), ' ')}…")
    print(f"-- {repo}: assistant messages scanned: {n_msgs}")
print(f"TOTAL assistant messages with a mention: {total}")
