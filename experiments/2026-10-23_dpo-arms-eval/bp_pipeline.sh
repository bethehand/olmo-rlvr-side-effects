#!/bin/bash
# Arm B' (icedduck/olmo3-7b-think-dpo-arm_Bp_seed1) evaluation with a smoke gate, then the judge over all arms. GS01, 2026-10-25.
# Protocol as for A and B (32K cap, full sets, seed 1, same software): the held-out / IFEval / GSM8K chain on GPUs 2,3 and the
# 805-prompt length run on GPUs 0,1. Differences from start_arm_eval.sh:
#   1. the compatibility fix and the loaded-config check against the start model were run by hand first
#      (fix_tf5_compat.py ... --check_against allenai/Olmo-3-7B-Think-SFT: "loaded config and tokenizer match");
#   2. the length run starts only after the held-out run (150 prompts, ~10 min) passes a smoke check (cap hits <= 5 %,
#      </think> present on >= 95 %, sane median answer length), so a broken checkpoint cannot burn hours; the chain on 2,3 is the
#      longer one (~3 h for B), so the gate costs no wall time;
#   3. when both chains are complete with full row counts (one resume attempt each), judge_all.sh runs on all four GPUs.
# Markers written by the other scripts are matched as whole lines; this script's own lines carry the prefix BPPIPE so that they
# can never satisfy a wait condition (lesson 4 in bluedot/project/experiment_lessons.md).
set -u
W=/mnt/nvme0/looped/oi_work; E=$W/experiments/2026-10-23_dpo-arms-eval; LE=$W/experiments/2026-10-18_length
R=icedduck/olmo3-7b-think-dpo-arm_Bp_seed1; N=icedduck_olmo3-7b-think-dpo-arm_Bp_seed1@main
source $W/env.sh >/dev/null; export HF_HUB_OFFLINE=1 MAXTOK=32768
cd $E
log() { echo "$(date '+%F %T') BPPIPE $*" >> $E/logs/done.txt; }
gpus_busy() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=",$1," 'index(g, "," $1 ",") && $2 > 1000 {n++} END {exit n > 0 ? 0 : 1}'; }
rows() { cat "$1" 2>/dev/null | wc -l; }
HELD_RE="^[0-9-]+ [0-9:]+ heldout $N run1 exit=[0-9]+ rows=[0-9]+ seconds=[0-9]+\$"
QUEUE_RE="^[0-9-]+ [0-9:]+ queue done $R run1\$"
LEN_RE="^$N run1 exit=[0-9]+ seconds=[0-9]+\$"

# 0. preconditions: fixed config, no other run of ours, idle GPUs, markers not already present
python3 - "$HF_HUB_CACHE"/models--icedduck--olmo3-7b-think-dpo-arm_Bp_seed1/snapshots/*/config.json <<'PY' || { log "config.json of B' is not the fixed one; stopping"; exit 1; }
import json, sys
c = json.load(open(sys.argv[1]))
assert c.get("rope_theta") == 500000 and (c.get("rope_scaling") or {}).get("rope_type") == "yarn", c.get("rope_theta")
PY
pgrep -af "generate_length|generate_prompts|judge_heldout" | grep -v pgrep >/dev/null && { log "another generation run of ours is active; stopping"; exit 1; }
gpus_busy 0,1,2,3 && { log "GPUs are not all idle; stopping"; exit 1; }
grep -qE "$HELD_RE" logs/done.txt && { log "held-out marker for B' already present; stopping"; exit 1; }

# 1. generation chain on GPUs 2,3 (held-out first, then IFEval, then GSM8K)
nohup bash queue_model.sh $R main run1 2,3 > logs/chain_gen_arm_Bp_seed1.out 2>&1 < /dev/null &
log "started the held-out/IFEval/GSM8K chain for $R on GPUs 2,3 (cap $MAXTOK, full sets, seed 1)"

# 2. smoke gate on the held-out run
H=outputs/heldout/${N}_run1.jsonl
for i in $(seq 1 360); do grep -qE "$HELD_RE" logs/done.txt && break; sleep 30; done
grep -qE "$HELD_RE" logs/done.txt || { log "held-out marker not seen within 3 h; stopping (generation chain left running)"; exit 1; }
smoke=$(python3 - "$H" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = len(rows)
cap = sum(r.get("finish_reason") == "length" for r in rows)
nothink = sum(not r.get("has_think_end") for r in rows)
med = statistics.median(r.get("answer_tokens", 0) for r in rows) if rows else 0
ok = n == 150 and cap <= 7 and nothink <= 7 and 20 <= med <= 5000
print(f"{'PASS' if ok else 'FAIL'} rows={n} cap_hits={cap} no_think_end={nothink} median_answer_tokens={med}")
PY
)
log "smoke check on the held-out run: $smoke (A and B had 0 cap hits of 150)"
case "$smoke" in PASS*) ;; *) log "smoke check failed; the length run is not started (generation chain left running for inspection)"; exit 1;; esac

# 3. length run on GPUs 0,1
nohup bash -c "cd $LE && bash run_length.sh $R main run1 0,1 '' $MAXTOK" > logs/chain_len_arm_Bp_seed1.out 2>&1 < /dev/null &
log "started the 805-prompt length run for $R on GPUs 0,1"

# 4. wait for both chains (up to 12 h), then check row counts with one resume attempt each
for i in $(seq 1 1440); do grep -qE "$QUEUE_RE" logs/done.txt && grep -qE "$LEN_RE" $LE/logs/done.txt && break; sleep 30; done
grep -qE "$QUEUE_RE" logs/done.txt && grep -qE "$LEN_RE" $LE/logs/done.txt || { log "chains not finished within 12 h; stopping"; exit 1; }
L=$LE/outputs/${N}_run1.jsonl
if [ "$(rows $L)" -lt 805 ]; then
  log "length run has $(rows $L) of 805 rows; resuming once"
  (cd $LE && bash run_length.sh $R main run1 0,1 '' $MAXTOK) >> logs/chain_len_arm_Bp_seed1.out 2>&1
fi
for t in heldout:150 ifeval:541 gsm8k:1319; do
  task=${t%%:*}; want=${t##*:}; f=outputs/$task/${N}_run1.jsonl
  if [ "$(rows $f)" -lt "$want" ]; then
    log "$task has $(rows $f) of $want rows; resuming once"
    bash run_gen.sh $R main $task run1 2,3 >> logs/chain_gen_arm_Bp_seed1.out 2>&1
  fi
done
log "B' evaluation complete: length $(rows $L)/805, heldout $(rows outputs/heldout/${N}_run1.jsonl)/150, ifeval $(rows outputs/ifeval/${N}_run1.jsonl)/541, gsm8k $(rows outputs/gsm8k/${N}_run1.jsonl)/1319"

# 5. judge on all four GPUs once they are idle (checked twice, a minute apart)
sleep 30; while gpus_busy 0,1,2,3; do sleep 60; done; sleep 60; while gpus_busy 0,1,2,3; do sleep 60; done
bash judge_all.sh
log "pipeline finished"
