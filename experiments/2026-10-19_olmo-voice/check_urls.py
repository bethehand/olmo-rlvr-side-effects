"""Check every URL found in model answers. Run on the Mac (unrestricted network via the system proxy).
broken = HTTP 404/410, a Hugging Face repo that does not exist, or a domain that does not resolve.
unknown = 401/403/429/5xx/timeouts (bot blocking or flaky sites), counted separately."""
import json, re, sys, socket, collections, urllib.request, urllib.error, urllib.parse, concurrent.futures as cf
src, out = sys.argv[1], sys.argv[2]
data = json.load(open(src))
urls = sorted({u for rows in data.values() for _, _, us in rows for u in us})
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"}
PLACEHOLDER = re.compile(r"//([\w-]+\.)?example\.(com|org|net)\b|//localhost|//127\.0\.0\.1|//0\.0\.0\.0|your[-_]?(domain|website|site|username|repo)|//[\w.-]*\.(local|test|invalid)\b", re.I)
def get(u, timeout=15):
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, socket.gaierror): return "dns"
        return "neterr:" + type(reason).__name__
    except Exception as e:
        return "neterr:" + type(e).__name__
def check(u):
    if PLACEHOLDER.search(u): return u, "placeholder"
    p = urllib.parse.urlparse(u)
    host = (p.hostname or "").lower()
    if not host or "." not in host: return u, "malformed"
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror:
        return u, "dns"
    except Exception:
        pass
    if host in ("huggingface.co", "www.huggingface.co"):
        parts = [x for x in p.path.split("/") if x]
        kind = "models"
        if parts and parts[0] in ("datasets", "spaces"): kind, parts = parts[0], parts[1:]
        if len(parts) >= 2 and parts[0] not in ("docs", "blog", "papers", "learn", "api", "models", "collections", "posts", "join", "settings", "pricing", "tasks"):
            s = get(f"https://huggingface.co/api/{kind}/{parts[0]}/{parts[1]}")
            return u, "200" if s == "200" else ("hf-missing" if s in ("401", "404") else s)
    return u, get(u)
with cf.ThreadPoolExecutor(24) as ex:
    status = dict(ex.map(check, urls))
json.dump(status, open(out, "w"), indent=0)
def bucket(s):
    if s in ("placeholder", "malformed"): return s
    if s in ("dns",): return "dead-domain"
    if s in ("404", "410", "hf-missing"): return "not-found"
    if s[:1] in "23": return "ok"
    return "unknown"
print(f"{len(urls)} unique URLs. Status counts:", collections.Counter(bucket(s) for s in status.values()))
print(f"{'run':40} {'ans':>4} {'w/URL':>5} {'URLs':>5} {'ok':>4} {'notfd':>5} {'dead':>4} {'unkn':>4} {'plch':>4} {'broken% (of checkable)':>22}")
for f, rows in data.items():
    us = [u for _, _, x in rows for u in x]
    b = collections.Counter(bucket(status[u]) for u in us)
    chk = b["ok"] + b["not-found"] + b["dead-domain"]
    br = b["not-found"] + b["dead-domain"]
    print(f"{f:40} {len(rows):4d} {sum(1 for _,_,x in rows if x):5d} {len(us):5d} {b['ok']:4d} {b['not-found']:5d} {b['dead-domain']:4d} {b['unknown']:4d} {b['placeholder']:4d} {100*br/max(chk,1):21.1f}%")
