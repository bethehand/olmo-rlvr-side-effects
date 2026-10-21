"""Direct re-check of two claims about the three Dolci DPO sets, straight from the cached parquet files:
(1) no pair carries a system turn; (2) how many answers mention Olmo/Ai2 (answer = text after the last </think>;
also counted on the full last assistant message, thinking included)."""
import collections, glob, os, re
import pyarrow.parquet as pq
cache = os.environ["HF_HUB_CACHE"]
OLMO = re.compile(r"\bOLMo\b|\bOlmo\b|\bAi2\b|\bAI2\b|Allen Institute")
def answer(text):
    j = text.rfind("</think>")
    return text[j + len("</think>"):] if j >= 0 else text
grand = collections.Counter()
for repo in ("Dolci-Think-DPO-32B", "Dolci-Think-DPO-7B", "Dolci-Instruct-DPO"):
    files = sorted(glob.glob(f"{cache}/datasets--allenai--{repo}/snapshots/*/**/*.parquet", recursive=True))
    c = collections.Counter(); roles = collections.Counter(); per_side = collections.Counter(); examples = []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000, columns=["chosen", "rejected"]):
            for row in batch.to_pylist():
                c["pairs"] += 1
                sys_here = False
                for side in ("chosen", "rejected"):
                    msgs = row[side]
                    per_side[len(msgs)] += 1
                    for m in msgs: roles[m["role"]] += 1
                    if any(m["role"] == "system" for m in msgs): sys_here = True
                    last = [m for m in msgs if m["role"] == "assistant"][-1]["content"] or ""
                    a = answer(last)
                    if OLMO.search(a): c["answers_with_mention"] += 1
                    if OLMO.search(last):
                        c["full_messages_with_mention"] += 1
                        if len(examples) < 4:
                            k = OLMO.search(last).start(); examples.append(("answer" if OLMO.search(a) else "thinking-only") + ": …" + last[max(0, k-70):k+70].replace("\n", " ") + "…")
                if sys_here: c["pairs_with_system_turn"] += 1
    print(f"== {repo}: files={len(files)} pairs={c['pairs']} | pairs with a system turn: {c['pairs_with_system_turn']} | roles: {dict(roles)} | messages per side: {dict(per_side)}")
    print(f"   Olmo/Ai2: answers with a mention {c['answers_with_mention']} of {2*c['pairs']}; full last messages with a mention {c['full_messages_with_mention']}")
    for e in examples: print("   ", e[:200])
    grand.update(c)
print(f"== ALL: pairs={grand['pairs']} system-turn pairs={grand['pairs_with_system_turn']} answers with mention={grand['answers_with_mention']} of {2*grand['pairs']} | full messages with mention={grand['full_messages_with_mention']}")
