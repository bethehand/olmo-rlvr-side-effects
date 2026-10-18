"""Answer formatting per stage (for open-instruct #2002): mean lines, headers, list items, bold spans and table rows
per answer, over the prompts that finished in all the given runs. The answer is the text after the last </think>
(the whole output for Instruct models, which write no </think>). Non-empty lines only; a list item is a line
starting with -, *, • or "1." / "1)"; a bold span is a pair of **; a table row is a line starting with |.
Usage: python format_counts.py outputs/A.jsonl outputs/B.jsonl ..."""
import json, os, re, statistics as st, sys

def answer(text):
    i = text.rfind("</think>")
    return text[i + len("</think>"):] if i >= 0 else text

def shape(a):
    lines = [l for l in a.splitlines() if l.strip()]
    return {"lines": len(lines), "headers": sum(l.lstrip().startswith("#") for l in lines),
            "list items": sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", l)) for l in lines),
            "bold spans": a.count("**") // 2, "table rows": sum(l.strip().startswith("|") for l in lines)}

def main():
    runs = []
    for p in sys.argv[1:]:
        rows = {r["index"]: r for r in map(json.loads, open(p))}
        runs.append((os.path.basename(p).replace(".jsonl", ""), rows, any(r["has_think_end"] for r in rows.values())))
    done = lambda r, thinks: r["finish_reason"] == "stop" and (r["has_think_end"] or not thinks)
    common = [i for i in sorted(runs[0][1]) if all(done(R[i], t) for _, R, t in runs)]
    print(f"mean per answer over the {len(common)} prompts finished in all {len(runs)} runs")
    for name, R, _ in runs:
        sh = [shape(answer(R[i]["response"])) for i in common]
        print(f"  {name:36} " + "  ".join(f"{k} {st.mean(x[k] for x in sh):5.1f}" for k in sh[0]))

if __name__ == "__main__":
    main()
