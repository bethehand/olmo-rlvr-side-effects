"""Resolve the hosts of URLs whose HTTP check was inconclusive, via DNS-over-HTTPS (the Mac's proxy answers every name with a fake IP).
NXDOMAIN = the domain does not exist -> the link is broken. Then re-tabulate per run."""
import json, sys, re, collections, urllib.request, urllib.parse, concurrent.futures as cf
urls_by_run = json.load(open(sys.argv[1])); status = json.load(open(sys.argv[2])); out = sys.argv[3]
def bucket(s):
    if s in ("placeholder", "malformed"): return s
    if s == "dns": return "dead-domain"
    if s in ("404", "410", "hf-missing"): return "not-found"
    if s[:1] in "23": return "ok"
    return "unknown"
unk = [u for u, s in status.items() if bucket(s) == "unknown"]
hosts = sorted({(urllib.parse.urlparse(u).hostname or "").lower() for u in unk} - {""})
def doh(h):
    for url in (f"https://dns.google/resolve?name={h}&type=A", f"https://cloudflare-dns.com/dns-query?name={h}&type=A"):
        try:
            req = urllib.request.Request(url, headers={"accept": "application/dns-json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
            return h, {0: "ok", 3: "nxdomain"}.get(d.get("Status"), f"rcode{d.get('Status')}")
        except Exception:
            continue
    return h, "doh-failed"
with cf.ThreadPoolExecutor(16) as ex:
    res = dict(ex.map(doh, hosts))
print(len(unk), "inconclusive URLs on", len(hosts), "hosts; DNS:", collections.Counter(res.values()))
new = dict(status)
for u in unk:
    if res.get((urllib.parse.urlparse(u).hostname or "").lower()) == "nxdomain": new[u] = "dns"
json.dump({"hosts": res, "status": new}, open(out, "w"), indent=0)
print("dead domains:", sorted(h for h, v in res.items() if v == "nxdomain")[:25])
tot = collections.Counter()
print(f"{'run':40} {'URLs':>5} {'ok':>4} {'notfd':>5} {'dead':>4} {'unkn':>4} {'broken% of checkable':>21}")
for f, rows in urls_by_run.items():
    us = [u for _, _, x in rows for u in x]
    b = collections.Counter(bucket(new[u]) for u in us); tot.update(b)
    chk = b["ok"] + b["not-found"] + b["dead-domain"]
    print(f"{f:40} {len(us):5d} {b['ok']:4d} {b['not-found']:5d} {b['dead-domain']:4d} {b['unknown']:4d} {100*(b['not-found']+b['dead-domain'])/max(chk,1):20.1f}%")
chk = tot["ok"] + tot["not-found"] + tot["dead-domain"]
print("pooled:", dict(tot), f"broken {tot['not-found']+tot['dead-domain']}/{chk} = {100*(tot['not-found']+tot['dead-domain'])/chk:.1f}%")
