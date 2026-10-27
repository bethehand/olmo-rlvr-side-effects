"""Source layout of a Hub parquet dataset from the parquet footers only: per shard, the min/max statistics of the source column
(one source per shard when min == max). Reads a few KB per shard through range requests. Falls back to reading the column when
statistics are missing. Usage: shard_sources_fast.py REPO COLUMN N_SHARDS OUT.txt"""
import sys, time, collections
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

repo, col, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
fs = HfFileSystem()
lines = []
for i in range(n):
    path = f"datasets/{repo}/data/train-{i:05d}-of-{n:05d}.parquet"
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        # row-group column indices are *leaf* indices (nested `messages` contributes two leaves), so find the leaf by path
        rg0 = md.row_group(0)
        idx = next(j for j in range(rg0.num_columns) if rg0.column(j).path_in_schema == col)
        mins, maxs, rows = set(), set(), 0
        for g in range(md.num_row_groups):
            c = md.row_group(g).column(idx); rows += md.row_group(g).num_rows
            st = c.statistics
            if st is not None and st.has_min_max:
                mins.add(st.min if isinstance(st.min, str) else st.min.decode()); maxs.add(st.max if isinstance(st.max, str) else st.max.decode())
        if not mins:
            vals = collections.Counter(pq.read_table(fh, columns=[col]).column(col).to_pylist())
            desc = "; ".join(f"{k}:{v}" for k, v in vals.most_common())
        else:
            desc = "SINGLE " + next(iter(mins)) if len(mins) == 1 and mins == maxs else f"MIXED min={sorted(mins)} max={sorted(maxs)}"
    line = f"shard {i:3d} rows={rows:6d} {desc}"
    lines.append(line); print(time.strftime('%H:%M:%S'), line, flush=True)
open(out, "w").write("\n".join(lines) + "\n")
print("FAST_SCAN_DONE", flush=True)
