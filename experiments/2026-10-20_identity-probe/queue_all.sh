#!/bin/bash
# The whole probe, in order of how much each run tells us. Runs one job at a time on all four cards (32B needs them),
# then the three 7B probes side by side. Each job is retried once if it crashes or comes up short (both scripts resume).
#   B = the 805 AlpacaEval prompts with the 3.1 release template (unprompted self-mentions; ../2026-10-18_length/run_length.sh)
#   A = the 43-prompt identity/cutoff/link probe under four system prompts, 3 samples each (generate_identity.py)
W=/mnt/nvme0/looped/oi_work; L=$W/experiments/2026-10-18_length; E=$W/experiments/2026-10-20_identity-probe
cd $E; mkdir -p logs; Q=logs/queue.log
log() { echo "$(date '+%F %T') $*" | tee -a $Q; }
rows() { cat "$1" 2>/dev/null | wc -l; }
PY=$W/safety-eval/.venv-se/bin/python
# go EXPECTED_ROWS OUTFILE CMD... : run CMD, rerun once if OUTFILE has fewer than EXPECTED_ROWS lines
go() { want=$1 out=$2; shift 2; for try in 1 2; do
  [ "$(rows $out)" -ge "$want" ] && break
  log "START try $try: $*"; "$@"; log "END rc=$? rows=$(rows $out)/$want: $*"; done; }
# goA OUTFILE CMD... : same for the probe, whose expected rows (43 x n x conditions kept) come from its .meta.json
wantA() { $PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(43*m['n']*len(m['conditions']))" $1.meta.json 2>/dev/null || echo 999999; }
goA() { out=$1; shift; for try in 1 2; do
  [ "$(rows $out)" -ge "$(wantA $out)" ] && break
  log "START try $try: $*"; "$@"; log "END rc=$? rows=$(rows $out)/$(wantA $out): $*"; done; }
# run_length.sh queues its own retry on a shared queue file; point that at an unused file in this folder instead
export RETRY_QUEUE=$E/logs/retry_unused
G=0,1,2,3
S32="GPU_MEM=0.80 MAX_SEQS=64"
log "queue start"
# 1. the key arm: do the SFT models already do it under the release prompt?
go 805 $L/outputs/Olmo-3-32B-Think-SFT@main_rel1.jsonl env $S32 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Think bash $L/run_length.sh allenai/Olmo-3-32B-Think-SFT main rel1 $G
go 805 $L/outputs/Olmo-3.1-32B-Instruct-SFT@main_rel1.jsonl env $S32 DTYPE=bfloat16 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Instruct bash $L/run_length.sh allenai/Olmo-3.1-32B-Instruct-SFT main rel1 $G
# 2. the identity probe on the seven 32B models
goA outputs/Olmo-3.1-32B-Think@main_p1.jsonl env $S32 bash run_probe.sh allenai/Olmo-3.1-32B-Think main p1 $G
goA outputs/Olmo-3-32B-Think-SFT@main_p1.jsonl env $S32 bash run_probe.sh allenai/Olmo-3-32B-Think-SFT main p1 $G
goA outputs/Olmo-3-32B-Think-DPO@main_p1.jsonl env $S32 bash run_probe.sh allenai/Olmo-3-32B-Think-DPO main p1 $G
goA outputs/Olmo-3-32B-Think@main_p1.jsonl env $S32 bash run_probe.sh allenai/Olmo-3-32B-Think main p1 $G
goA outputs/Olmo-3.1-32B-Instruct@main_p1.jsonl env $S32 bash run_probe.sh allenai/Olmo-3.1-32B-Instruct main p1 $G
goA outputs/Olmo-3.1-32B-Instruct-SFT@main_p1.jsonl env $S32 DTYPE=bfloat16 bash run_probe.sh allenai/Olmo-3.1-32B-Instruct-SFT main p1 $G
goA outputs/Olmo-3.1-32B-Instruct-DPO@main_p1.jsonl env $S32 DTYPE=bfloat16 bash run_probe.sh allenai/Olmo-3.1-32B-Instruct-DPO main p1 $G
# 3. the rest of B: where along the stages does it start?
go 805 $L/outputs/Olmo-3-32B-Think-DPO@main_rel1.jsonl env $S32 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Think bash $L/run_length.sh allenai/Olmo-3-32B-Think-DPO main rel1 $G
go 805 $L/outputs/Olmo-3.1-32B-Instruct-DPO@main_rel1.jsonl env $S32 DTYPE=bfloat16 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Instruct bash $L/run_length.sh allenai/Olmo-3.1-32B-Instruct-DPO main rel1 $G
go 805 $L/outputs/Olmo-3-32B-Think@main_rel1.jsonl env $S32 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Think bash $L/run_length.sh allenai/Olmo-3-32B-Think main rel1 $G
# 4. the three 7B Think stages, two cards each (one 24 GB card can't hold even a 20K-token KV cache next to the weights),
#    two at a time
S7="GPU_MEM=0.85 MAX_SEQS=64 MAXLEN=20480"
goA outputs/Olmo-3-7B-Think-SFT@main_p1.jsonl env $S7 bash run_probe.sh allenai/Olmo-3-7B-Think-SFT main p1 0,1 &
goA outputs/Olmo-3-7B-Think-DPO@main_p1.jsonl env $S7 bash run_probe.sh allenai/Olmo-3-7B-Think-DPO main p1 2,3 &
wait
goA outputs/Olmo-3-7B-Think@main_p1.jsonl env $S7 bash run_probe.sh allenai/Olmo-3-7B-Think main p1 0,1
log "queue done"; touch logs/QUEUE_DONE
