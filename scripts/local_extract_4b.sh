#!/usr/bin/env bash
# Local Q3-4B extraction (exp1 v1, exp1 v2, exp2) on MPS. Rerunning resumes: completed passages are skipped.
# Patching can also run locally on MPS; for the 100-passage BBC run use scripts/local_exp1.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-runs/Q3-4B-local}
TOTAL_GB=${TOTAL_GB:-52}   # measured scale-up of the 0.6B run: ~49 GB exp1 (v1+v2) + ~2.5 GB exp2
KEEP_GB=${KEEP_GB:-15}     # free space left for the OS after the run
LOG="$RUN/extract.log"
mkdir -p "$RUN"

have_gb=$(df -g . | awk 'NR==2 {print $4}')
done_gb=$(du -sg "$RUN" | awk '{print $1}')
need_gb=$(( TOTAL_GB - done_gb + KEEP_GB ))
if (( have_gb < need_gb )); then
  echo "free ${have_gb} GB < needed ${need_gb} GB (remaining output + ${KEEP_GB} GB headroom); aborting" | tee -a "$LOG"
  exit 1
fi

# Keep the Mac awake while this script runs (system sleep is prevented on AC power only; closing the lid still sleeps).
caffeinate -is -w $$ &

step() {
  echo "[$(date '+%F %T')] start: $*" | tee -a "$LOG"
  uv run python -m kvmeaning.extract --model Q3-4B "$@" 2>&1 | grep --line-buffered -v 'Loading weights' | tee -a "$LOG"
  echo "[$(date '+%F %T')] done: $*" | tee -a "$LOG"
}

step --exp 1 --design v1 --run-dir "$RUN/v1"
step --exp 1 --design v2 --run-dir "$RUN/v2"
step --exp 2 --run-dir "$RUN/v1"

for d in v1 v2; do
  uv run python -m kvmeaning.summary "$RUN/$d" "$RUN/$d/summary.json" 2>&1 | tee -a "$LOG" \
    || echo "summary $d failed (extraction itself is complete)" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] all done; $(du -sh "$RUN" | cut -f1) in $RUN, $(df -h . | awk 'NR==2 {print $4}') free" | tee -a "$LOG"
