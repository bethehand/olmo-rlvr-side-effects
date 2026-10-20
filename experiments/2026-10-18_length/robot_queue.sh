#!/bin/bash
W=/mnt/nvme0/looped/oi_work; E=$W/experiments/2026-10-18_length; cd $E
log(){ echo "$(date '+%F %T') $*" >> logs/robot_queue.log; }
log "START Olmo-3-32B-Think-SFT rob1 (system block = robot_system_think.txt, 3.1 Think template)"
RETRY=2 GPU_MEM=0.8 MAX_SEQS=64 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Think SYSTEM_FILE=$E/robot_system_think.txt bash run_length.sh allenai/Olmo-3-32B-Think-SFT main rob1 0,1,2,3
log "END Olmo-3-32B-Think-SFT rob1 rows=$(cat outputs/Olmo-3-32B-Think-SFT@main_rob1.jsonl 2>/dev/null | wc -l)"
log "START Olmo-3.1-32B-Instruct-DPO rob1 (system block = robot_system_instruct.txt, 3.1 Instruct template)"
RETRY=2 GPU_MEM=0.8 MAX_SEQS=64 DTYPE=bfloat16 TEMPLATE_FROM=allenai/Olmo-3.1-32B-Instruct SYSTEM_FILE=$E/robot_system_instruct.txt bash run_length.sh allenai/Olmo-3.1-32B-Instruct-DPO main rob1 0,1,2,3
log "END Olmo-3.1-32B-Instruct-DPO rob1 rows=$(cat outputs/Olmo-3.1-32B-Instruct-DPO@main_rob1.jsonl 2>/dev/null | wc -l)"
touch logs/ROBOT_DONE; log "queue done"
