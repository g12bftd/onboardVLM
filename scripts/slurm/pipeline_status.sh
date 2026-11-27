#!/usr/bin/env bash
# Status of the data-pipeline jobs (ovlm-*) of the last N days (default 7): the latest attempt of each job
# with its state, exit code and run time, the finished steps, and the end of the log of every job that ran
# and did not complete (the cause). Jobs CANCELLED because a job they depend on failed never ran: look at
# that job instead.
#
#   source scripts/paths.sh && bash scripts/slurm/pipeline_status.sh [days] [log_lines]
set -euo pipefail
: "${SLURM_LOG_DIR:?run 'source scripts/paths.sh' first}"
DAYS="${1:-7}" LINES="${2:-25}"
ME="${USER:-$(id -un)}"
STEPS_DIR="${DATA_ROOT:-$ONBOARD_VLM_ROOT/data}/.steps"

# latest attempt per job name: JobID|JobName|State|ExitCode|Elapsed|NodeList
jobs=$(sacct -X -n -P -u "$ME" -S "now-${DAYS}days" -o JobID,JobName,State,ExitCode,Elapsed,NodeList 2>/dev/null \
  | awk -F'|' '$2 ~ /^ovlm-/' | sort -t'|' -k1,1n | awk -F'|' '{last[$2]=$0} END {for (n in last) print last[n]}' | sort -t'|' -k1,1n)
if [ -z "$jobs" ]; then
  echo "no ovlm-* jobs in the last $DAYS day(s) (sacct): were they submitted? The submit script prints one line per job."
  exit 0
fi

echo "== ovlm-* jobs, latest attempt each (last $DAYS day(s))"
printf '%-10s %-28s %-24s %-6s %-10s %s\n' JOBID NAME STATE EXIT ELAPSED NODE
echo "$jobs" | awk -F'|' '{printf "%-10s %-28s %-24s %-6s %-10s %s\n", $1, $2, $3, $4, $5, $6}'
echo
echo "== queued or running now (squeue; jobs waiting on another job are only listed here, not by sacct)"
q=$(squeue -u "$ME" -h -o '%i|%j|%T|%M|%R' 2>/dev/null | awk -F'|' '$2 ~ /^ovlm-/' | sort -t'|' -k1,1n || true)
if [ -n "$q" ]; then
  printf '%-10s %-28s %-10s %-10s %s\n' JOBID NAME STATE TIME "REASON / NODE"
  echo "$q" | awk -F'|' '{printf "%-10s %-28s %-10s %-10s %s\n", $1, $2, $3, $4, $5}'
else
  echo "none"
fi
SKIPPED=" $(cat "${DATA_ROOT:-$ONBOARD_VLM_ROOT/data}/.pipeline_skip" 2>/dev/null || true) "   # --skip of the last submission
echo
echo "== finished steps ($STEPS_DIR)"
ls "$STEPS_DIR" 2>/dev/null | sed 's/\.ok$//' | tr '\n' ' '; echo

echo "$jobs" | while IFS='|' read -r id name state code elapsed node; do
  case "$state" in COMPLETED*|RUNNING*|PENDING*|REQUEUED*|CONFIGURING*|COMPLETING*) continue ;; esac
  log="$SLURM_LOG_DIR/$name-$id.out"
  [ -f "$log" ] || continue                                   # never started (e.g. cancelled: dependency failed)
  src="${name#ovlm-}"; src="${src#dl-}"; src="${src#corpus-}"
  if [[ "$SKIPPED" == *" $src "* ]]; then
    echo; echo "== $name (job $id, $state): skipped by the last submission (--skip $src); not needed now"; continue
  fi
  echo
  echo "== $name (job $id, $state, exit $code, ran $elapsed): last $LINES lines of $log"
  tail -n "$LINES" "$log"
done
