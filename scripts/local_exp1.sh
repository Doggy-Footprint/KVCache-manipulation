#!/usr/bin/env bash
# Exp1 on the 100 BBC passages (designs bbc-v1, bbc-v2) with Q3-4B on MPS: extraction, stats, Drive upload of raw
# tensors, and patching, in one resumable process. Rerunning with the same NAME resumes; extra args go to
# kvmeaning.local_exp1 (e.g. --only bus482, --upload-only).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-main}
RUN=runs/exp1-100p-4b/$NAME
mkdir -p "$RUN"
LOG="$RUN/run.log"

# Keep the Mac awake while this script runs (system sleep is prevented on AC power only; closing the lid still sleeps).
caffeinate -is -w $$ &

echo "[$(date '+%F %T')] start: $RUN $*" | tee -a "$LOG"
status=0
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 uv run python -m kvmeaning.local_exp1 --run-dir "$RUN" "$@" 2>&1 \
  | { grep --line-buffered -v 'Loading weights' || true; } | tee -a "$LOG" || status=$?
echo "[$(date '+%F %T')] exit $status: $RUN" | tee -a "$LOG"
exit "$status"
