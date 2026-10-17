"""Table 52 of the Olmo 3 report (7B Think: SFT / DPO / final, safety scores) next to our runs of Ai2's own protocol.
Reads outputs/<model>/<run>/metrics.json.<task> written by safety-eval. Scores are percentages, higher = safer
(for WJ benign and XSTest, higher = fewer over-refusals). Usage: python summarize.py [outputs_dir]"""
import json, os, statistics as st, sys

TASKS = {  # task -> (column, primary metric used by safety-eval's overall average / OLMES)
    "do_anything_now:wildguard_reasoning_answer": ("DAN", "inverted_macro_asr"),
    "harmbench:wildguard_reasoning_answer": ("HarmBench", "inverted_micro_asr_lower"),
    "trustllm_jailbreaktrigger:wildguard_reasoning_answer": ("TrustLLM", "inverted_macro_asr"),
    "wildjailbreak:harmful_wildguard_reasoning_answer": ("WJ harmful", "inverted_macro_asr"),
    "wildjailbreak:benign_wildguard_reasoning_answer": ("WJ benign", "macro_asr"),
    "wildguardtest:wildguard_reasoning_answer": ("WildGuardTest", "inverted_micro_harm_lower"),
    "xstest:wildguard_reasoning_answer": ("XSTest", "overall_accuracy"),
    "toxigen:tiny_reasoning_answer": ("Toxigen", "safe_overall"),
    "strongreject:logprobs_reasoning_answer": ("StrongReject", "inverted_asr"),
}
# Olmo 3 tech report, Table 52 (mean of 3 runs).
AI2 = {
    "Olmo-3-7B-Think-SFT": {"DAN": 19.3, "HarmBench": 67.8, "TrustLLM": 64.8, "WJ harmful": 23.4, "WJ benign": 99.1, "WildGuardTest": 90.2, "XSTest": 91.6, "Toxigen": 100.0, "StrongReject": 74.8},
    "Olmo-3-7B-Think-DPO": {"DAN": 19.6, "HarmBench": 72.7, "TrustLLM": 65.2, "WJ harmful": 27.5, "WJ benign": 98.5, "WildGuardTest": 93.9, "XSTest": 91.6, "Toxigen": 99.9, "StrongReject": 75.5},
    "Olmo-3-7B-Think":     {"DAN": 23.4, "HarmBench": 75.4, "TrustLLM": 72.0, "WJ harmful": 39.0, "WJ benign": 98.8, "WildGuardTest": 93.8, "XSTest": 90.9, "Toxigen": 100.0, "StrongReject": 79.0},
}

def load(out_dir):
    res = {}  # model -> run -> column -> (score, n, empty answers)
    for model in sorted(os.listdir(out_dir)):
        for run in sorted(os.listdir(os.path.join(out_dir, model))):
            d = os.path.join(out_dir, model, run)
            for task, (col, metric) in TASKS.items():
                f = os.path.join(d, f"metrics.json.{task}")
                if not os.path.exists(f):
                    continue
                rep = json.load(open(f))[task]
                err = rep.get("error_metrics", {})
                res.setdefault(model, {}).setdefault(run, {})[col] = (100 * rep[metric], err.get("empty_response"), err.get("truncations"))
    return res

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    res = load(out_dir)
    cols = [c for c, _ in TASKS.values()]
    for model, runs in res.items():
        print(f"\n{model}")
        print(f"  {'':12}" + "".join(f"{c:>14}" for c in cols))
        for run, r in runs.items():
            print(f"  {run:12}" + "".join(f"{r[c][0]:14.1f}" if c in r else f"{'-':>14}" for c in cols))
        full = [r for name, r in runs.items() if not name.startswith("smoke")]
        if full:
            means = {c: st.mean(r[c][0] for r in full if c in r) for c in cols if any(c in r for r in full)}
            print(f"  {'mean':12}" + "".join(f"{means[c]:14.1f}" if c in means else f"{'-':>14}" for c in cols))
            if len(full) > 1:
                sds = {c: st.stdev([r[c][0] for r in full if c in r]) for c in cols if sum(c in r for r in full) > 1}
                print(f"  {'sd (runs)':12}" + "".join(f"{sds[c]:14.1f}" if c in sds else f"{'-':>14}" for c in cols))
        if model in AI2:
            print(f"  {'Ai2 Table 52':12}" + "".join(f"{AI2[model].get(c, float('nan')):14.1f}" for c in cols))
        empties = {c: int(sum((r[c][1] or 0) for r in runs.values() if c in r)) for c in cols}
        print(f"  {'empty ans.':12}" + "".join(f"{empties[c]:14d}" for c in cols) + "   (no </think>: judged unsafe on harmful tasks, as a refusal on benign ones)")

if __name__ == "__main__":
    main()
