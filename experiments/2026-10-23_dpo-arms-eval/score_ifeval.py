"""Score IFEval outputs from generate_prompts.py with Google's official checkers (instruction_following_eval, Apache 2.0;
the four files instructions.py, instructions_registry.py, instructions_util.py, evaluation_lib.py from
github.com/google-research/google-research, placed in the directory given by --ife_src). Strict and loose accuracy follow
evaluation_lib.py: loose tries the response with the first/last line removed and with asterisks removed. The response that is
checked is the answer after the last </think> (what the user reads), as OLMES does for thinking models.
Prints prompt-level and instruction-level accuracy with Wilson 95% intervals per file and, with two or more files, the paired
prompt-level difference to the first file with a bootstrap interval over shared prompts.
Usage: score_ifeval.py --ife_src DIR OUT.jsonl [OUT2.jsonl ...]"""
import argparse, json, math, random, sys, types


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def check(registry, row, response):
    """Returns (strict_all, loose_all, strict_per_instruction, loose_per_instruction), as evaluation_lib does."""
    strict, loose = [], []
    r = response.strip()
    lines = r.split("\n")
    variants = [r, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    variants += [v.replace("*", "") for v in variants]
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        ins = registry.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**kw)
        ia = ins.get_instruction_args()
        if ia and "prompt" in ia:
            ins.build_description(prompt=row["prompt"])
        strict.append(bool(r) and ins.check_following(r))
        loose.append(any(bool(v) and ins.check_following(v) for v in variants))
    return all(strict), all(loose), strict, loose


def load(path):
    rows = {}
    for l in open(path):
        if l.strip():
            x = json.loads(l); rows[x["index"]] = x
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ife_src", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    # Google's files import one another as the package instruction_following_eval: expose the directory under that name
    pkg = types.ModuleType("instruction_following_eval"); pkg.__path__ = [args.ife_src]; sys.modules["instruction_following_eval"] = pkg
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from instruction_following_eval import instructions_registry as registry
    # Two sources of run-to-run variation in Google's code, both seeded here so a score can be reproduced: langdetect (language
    # checker) and random.choice in the checkers where the data gives no usable value (two prompts ask for the letter '#',
    # which LetterFrequencyChecker replaces by a random letter). Without seeding, prompt accuracy moves by up to 2 prompts.
    import langdetect; langdetect.DetectorFactory.seed = 0
    res = {}
    for f in args.files:
        random.seed(0)   # re-seeded per file, so a file's score does not depend on which files precede it on the command line
        rows = load(f)
        out = {}
        ins_s = ins_l = ins_n = 0
        for i, x in rows.items():
            s, l, ss, ll = check(registry, x, x["answer"])
            out[i] = (s, l); ins_s += sum(ss); ins_l += sum(ll); ins_n += len(ss)
        res[f] = out
        n = len(out); ks = sum(s for s, _ in out.values()); kl = sum(l for _, l in out.values())
        cap = sum(x["finish_reason"] == "length" for x in rows.values())
        print(f"{f}: n={n}  prompt strict={ks/n:.3f} [{wilson(ks, n)[0]:.3f}, {wilson(ks, n)[1]:.3f}]  "
              f"prompt loose={kl/n:.3f} [{wilson(kl, n)[0]:.3f}, {wilson(kl, n)[1]:.3f}]  "
              f"instruction strict={ins_s/ins_n:.3f} loose={ins_l/ins_n:.3f}  hit cap={cap}")
    if len(args.files) > 1:
        random.seed(0)
        base = res[args.files[0]]
        for f in args.files[1:]:
            idx = sorted(set(base) & set(res[f]))
            for name, j in (("strict", 0), ("loose", 1)):
                d = [res[f][i][j] - base[i][j] for i in idx]
                mean = sum(d) / len(d)
                bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(2000))
                print(f"  {f} minus {args.files[0]}, prompt {name}: n={len(d)} diff={mean:+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]")


if __name__ == "__main__":
    main()
