"""Count one string column across remote parquet shards without downloading them: curl the footer and
that column's chunks into a sparse local copy, read the column locally, then fetch full row groups only
where a source matches a needle. Usage: sparse_col_scan.py REPO WORKDIR OUT.json needle1 [needle2 ...]"""
import json, os, struct, subprocess, sys, collections, concurrent.futures as cf
import pyarrow.parquet as pq
from huggingface_hub import HfApi

repo, work, out, needles = sys.argv[1], sys.argv[2], sys.argv[3], [n.lower() for n in sys.argv[4:]]
KEYS = ("dataset_source", "source", "source_dataset")
base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
info = HfApi().dataset_info(repo, files_metadata=True)
shards = sorted((s.rfilename, s.size) for s in info.siblings if s.rfilename.endswith(".parquet"))

def get_range(name, start, end):
    for attempt in range(8):
        r = subprocess.run(["curl", "-sfL", "--max-time", "180", "-r", f"{start}-{end}", base + name], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == end - start + 1:
            return r.stdout
    raise RuntimeError(f"{name}: range {start}-{end} failed")

def put(local, start, data):
    with open(local, "r+b") as f:
        f.seek(start); f.write(data)

def chunk_span(col):
    start = col.data_page_offset
    if col.has_dictionary_page and col.dictionary_page_offset:
        start = min(start, col.dictionary_page_offset)
    return start, start + col.total_compressed_size - 1

def scan(shard):
    name, size = shard
    local = os.path.join(work, os.path.basename(name))
    with open(local, "wb") as f:
        f.truncate(size)
    flen = struct.unpack("<I", get_range(name, size - 8, size - 5))[0]
    put(local, size - 8 - flen, get_range(name, size - 8 - flen, size - 1))
    md = pq.ParquetFile(local).metadata
    key = next(k for k in KEYS if k in pq.ParquetFile(local).schema_arrow.names)
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        col = next(rg.column(j) for j in range(rg.num_columns) if rg.column(j).path_in_schema == key)
        s, e = chunk_span(col)
        put(local, s, get_range(name, s, e))
    pf = pq.ParquetFile(local)
    per_group = [collections.Counter(pf.read_row_group(g, columns=[key]).column(0).to_pylist()) for g in range(md.num_row_groups)]
    hits = []
    for g, c in enumerate(per_group):
        if any(n in (s or "").lower() for s in c for n in needles):
            rg = md.row_group(g)
            spans = [chunk_span(rg.column(j)) for j in range(rg.num_columns)]
            s, e = min(x[0] for x in spans), max(x[1] for x in spans)
            put(local, s, get_range(name, s, e))
            rows = pq.ParquetFile(local).read_row_group(g).to_pylist()
            hits += [dict(r, _shard=name, _row_group=g) for r in rows
                     if any(n in (r.get(key) or "").lower() for n in needles)]
    os.remove(local)
    print(f"  done {name}: {md.num_row_groups} row groups, {md.num_rows} rows, {len(hits)} hits", flush=True)
    return sum(per_group, collections.Counter()), hits

counts, rows = collections.Counter(), []
with cf.ThreadPoolExecutor(6) as ex:
    for c, h in ex.map(scan, shards):
        counts.update(c); rows += h
print(len(shards), "shards;", sum(counts.values()), "rows;", len(counts), "sources")
for s, n in counts.most_common():
    print(f"{n:9d}  {s}")
print("matching rows:", len(rows))
json.dump(rows, open(out, "w"), ensure_ascii=False, indent=0, default=str)
