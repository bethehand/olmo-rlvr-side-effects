"""Training-loss summary of the three DPO arms from the rented node's logs (dpo_tune_cache.py prints 'Step: N, LR: x, Loss: y'
every 10 optimizer steps, 667 steps per arm, 1 epoch over 20,000 pairs, effective batch 30, length-normalised DPO loss, beta 5).
Reports the mean logged loss over the first and last third of training, the last five logged values, and exp(-loss), which for
the DPO loss -log sigmoid(beta * margin) is the implied mean probability that the policy prefers the chosen answer: a higher loss
means a smaller implicit reward margin was learned from the pairs.
Usage: train_loss_table.py LOG_DIR   (LOG_DIR holds arm_A_seed1.log, arm_B_seed1.log, arm_Bp_seed1.log)"""
import math, os, re, statistics, sys

D = sys.argv[1]
pat = re.compile(r"Step: (\d+), LR: ([0-9.e-]+), Loss: ([0-9.]+)")
print("arm   | logged steps | mean loss steps 1-220 | steps 221-440 | steps 441-667 | last five logged | exp(-loss) last third")
for lab, f in (("A", "arm_A_seed1.log"), ("B", "arm_B_seed1.log"), ("B'", "arm_Bp_seed1.log")):
    pts = [(int(m.group(1)), float(m.group(3))) for m in (pat.search(l) for l in open(os.path.join(D, f), errors="ignore")) if m]
    if not pts:
        print(f"  {lab}: no step lines"); continue
    thirds = [[v for s, v in pts if lo < s <= hi] for lo, hi in ((0, 220), (220, 440), (440, 700))]
    last = ", ".join(f"{v:.3f}" for s, v in pts[-5:])
    m3 = statistics.mean(thirds[2])
    print(f"  {lab:3s} | {len(pts):4d} (to step {pts[-1][0]}) | {statistics.mean(thirds[0]):.3f} | {statistics.mean(thirds[1]):.3f} | {m3:.3f} | {last} | {math.exp(-m3):.3f}")
