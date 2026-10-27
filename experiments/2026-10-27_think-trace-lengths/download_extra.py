"""Download specific parquet shards of a public Hub dataset into data/<subdir>/ (no token needed).
Usage: download_extra.py REPO N_SHARDS SUBDIR IDX [IDX ...]   e.g. download_extra.py allenai/Dolci-Think-SFT-7B 156 think_sft_7b 4 67 70"""
import os, sys, time
from huggingface_hub import hf_hub_download
W = os.path.dirname(os.path.abspath(__file__))
repo, n, sub = sys.argv[1], int(sys.argv[2]), sys.argv[3]
out = os.path.join(W, "data", sub); os.makedirs(out, exist_ok=True)
for i in map(int, sys.argv[4:]):
    fn = f"data/train-{i:05d}-of-{n:05d}.parquet"; t0 = time.time()
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=fn, local_dir=out)
    print(time.strftime('%H:%M:%S'), repo, fn, '->', '%.0f MB' % (os.path.getsize(p) / 1e6), 'in', '%.0fs' % (time.time() - t0), flush=True)
print("EXTRA_DONE", flush=True)
