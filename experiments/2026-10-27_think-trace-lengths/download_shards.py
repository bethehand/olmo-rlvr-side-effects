"""Download evenly spaced parquet shards of two public Dolci-Think SFT datasets (no token needed).
Clean-v1 default: 128 shards -> every 8th (16 shards, ~3.1 GB). Dolci-Think-SFT-7B: 156 shards -> every 20th (8 shards, ~0.8 GB).
Evenly spaced shards give an unweighted sample of the dataset; the rows inside a shard are in source order, so shards are not iid, but
16 of 128 spread across the file order cover every region of the dataset."""
import os, time
from huggingface_hub import hf_hub_download

W = os.path.dirname(os.path.abspath(__file__))
JOBS = [
    ("allenai/Dolci-Think-SFT-Clean-v1", "data/train-{i:05d}-of-00128.parquet", range(0, 128, 8), "clean_v1"),
    ("allenai/Dolci-Think-SFT-7B", "data/train-{i:05d}-of-00156.parquet", range(0, 156, 20), "think_sft_7b"),
]
for repo, pat, idxs, sub in JOBS:
    out = os.path.join(W, "data", sub)
    os.makedirs(out, exist_ok=True)
    for i in idxs:
        fn = pat.format(i=i)
        t0 = time.time()
        p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=fn, local_dir=out)
        mb = os.path.getsize(p) / 1e6
        print(time.strftime('%H:%M:%S'), repo, fn, '->', '%.0f MB' % mb, 'in', '%.0fs' % (time.time() - t0), flush=True)
print("DOWNLOAD_DONE", flush=True)
