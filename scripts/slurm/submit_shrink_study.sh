#!/usr/bin/env bash
# Submit a shrinking study (docs/shrink_study.md): one training job per (configuration, seed), each followed
# (afterok) by the evaluation of its trajectory exports (scripts/trajectory_eval.py, E1 / E2 / E3 on 4 GPUs),
# plus one evaluation of the teacher (reference lines). Prints the plot command at the end.
#
#   source scripts/paths.sh
#   bash scripts/slurm/submit_shrink_study.sh --account <project> --study depth [--seeds 0,1,2] [--dry-run] [-- <hydra overrides>]
#
# Studies (every run: +experiment=shrink/base, same data, objective, schedule and compute; one factor changes):
#   identity  16 x 768, init teacher: the teacher itself (sanity anchor: fidelity 1 at step 0)
#   depth     {12, 8, 6, 4, 2} layers x 768, init teacher (uniform layer selection)
#   width     16 layers x {512, 384, 256, 128}, init teacher (weight selection)
#   init      8 x 384: teacher uniform | teacher first | random
#   grid      {4, 8, 12} layers x {256, 384, 512}, init teacher
# Re-running is safe: finished runs ($ONBOARD_VLM_OUT/students/<run_id>) and finished evaluations are skipped,
# queued or running jobs of the same name are reused. Overrides after `--` go to every training run
# (e.g. -- allow_missing_sources=true trainer.max_epochs=10).
set -euo pipefail
: "${ONBOARD_VLM_ROOT:?run 'source scripts/paths.sh' first}"
ACCOUNT="" STUDY="" SEEDS="0,1,2" TIME="24:00:00" EVAL_TIME="12:00:00" DRY="" TEACHER_CFG="gemma4_e2b"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --study) STUDY="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --teacher) TEACHER_CFG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$ACCOUNT" ] || [ -n "$DRY" ] || { echo "--account <project> is required" >&2; exit 2; }
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
export ONBOARD_VLM_CODE="$CODE"
JOB="$CODE/scripts/slurm/run.sbatch"
OUT="$ONBOARD_VLM_OUT"
ME="${USER:-$(id -un)}"
if [ -z "${CONDA_BASE:-}" ] && [ -n "${CONDA_EXE:-}" ]; then CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"; fi
export CONDA_BASE="${CONDA_BASE:-$ONBOARD_VLM_ROOT/miniforge3}" ONBOARD_VLM_ENV="${ONBOARD_VLM_ENV:-onboardvlm}"

# configurations: "layers width init layer_selection width_selection"
case "$STUDY" in
  identity) CONFIGS=("16 768 teacher uniform uniform") ;;
  depth) CONFIGS=("12 768 teacher uniform uniform" "8 768 teacher uniform uniform" "6 768 teacher uniform uniform" "4 768 teacher uniform uniform" "2 768 teacher uniform uniform") ;;
  width) CONFIGS=("16 512 teacher uniform uniform" "16 384 teacher uniform uniform" "16 256 teacher uniform uniform" "16 128 teacher uniform uniform") ;;
  init) CONFIGS=("8 384 teacher uniform uniform" "8 384 teacher first first" "8 384 random uniform uniform") ;;
  grid) CONFIGS=(); for L in 4 8 12; do for D in 256 384 512; do CONFIGS+=("$L $D teacher uniform uniform"); done; done ;;
  *) echo "--study identity | depth | width | init | grid" >&2; exit 2 ;;
esac

declare -A QUEUED
if command -v squeue >/dev/null 2>&1; then
  while read -r jid jname; do QUEUED[$jname]="$jid"; done < <(squeue -u "$ME" -h -o '%i %j' 2>/dev/null || true)
fi
sub() {  # sub <job-name> <time> <dependency or ""> <args for run.sbatch...>; prints the job id
  local name="$1" time="$2" dep="$3"; shift 3
  if [ -n "${QUEUED[$name]:-}" ]; then echo "  queued   $name (job ${QUEUED[$name]})" >&2; echo "${QUEUED[$name]}"; return; fi
  if [ -n "$DRY" ]; then echo "  submit   $name ($time)${dep:+ after $dep}: $*" >&2; echo "<$name>"; return; fi
  local id
  id=$(sbatch --parsable --account="$ACCOUNT" --time="$time" --job-name="$name" --output="$SLURM_LOG_DIR/%x-%j.out" \
    --kill-on-invalid-dep=yes ${dep:+--dependency=afterok:$dep} "$JOB" "$@")
  id="${id%%;*}"
  echo "  submit   $name -> job $id${dep:+ after $dep}" >&2
  echo "$id"
}

echo "shrink study '$STUDY': ${#CONFIGS[@]} configurations x seeds {$SEEDS} (${DRY:+dry run, }outputs in $OUT)" >&2
TEACHER_DIR="$OUT/trajectories/_teacher_${TEACHER_CFG#gemma4_}"
if [ -f "$TEACHER_DIR/eval/e2/results.json" ] && [ -f "$TEACHER_DIR/eval/e3/cvbench/metrics.json" ]; then
  echo "  done     teacher reference ($TEACHER_DIR)" >&2
else
  sub "ovlm-eval-teacher-${TEACHER_CFG#gemma4_}" "$EVAL_TIME" "" scripts/trajectory_eval.py --run_dir "$TEACHER_DIR" --teacher \
    --teacher_config "$TEACHER_CFG" --gpus 4 >/dev/null
fi
patterns=()
for c in "${CONFIGS[@]}"; do
  read -r L D INIT LS WS <<< "$c"
  NAME="shrink_L${L}_D${D}_${INIT}_${LS}-${WS}"
  patterns+=("$NAME")
  for s in ${SEEDS//,/ }; do
    RUN="${NAME}_s${s}"
    if [ -f "$OUT/students/$RUN/student.safetensors" ]; then
      echo "  done     $RUN (trained)" >&2; train=""
    else
      train=$(sub "$RUN" "$TIME" "" run_distill.py +experiment=shrink/base teacher="$TEACHER_CFG" student.num_hidden_layers="$L" \
        student.hidden_size="$D" student.init="$INIT" student.layer_selection="$LS" student.width_selection="$WS" seed="$s" "${EXTRA[@]}")
    fi
    n_steps=$(find "$OUT/trajectories/$RUN" -maxdepth 1 -name 'step_*' 2>/dev/null | wc -l || true)
    n_done=$(find "$OUT/trajectories/$RUN" -path '*/eval/e3/cvbench/metrics.json' 2>/dev/null | wc -l || true)
    if [ -z "$train" ] && [ "$n_steps" -gt 0 ] && [ "$n_done" -eq "$n_steps" ]; then
      echo "  done     eval-$RUN ($n_steps checkpoints evaluated)" >&2
    else
      sub "eval-$RUN" "$EVAL_TIME" "$train" scripts/trajectory_eval.py --run_dir "$OUT/trajectories/$RUN" --teacher_config "$TEACHER_CFG" \
        --gpus 4 >/dev/null
    fi
  done
done
case "$STUDY" in depth) LAB='{L} layers'; ORD="--order L" ;; width) LAB='width {D}'; ORD="--order D" ;;
  init) LAB='{init} ({ls})'; ORD="" ;; grid) LAB='L{L} D{D}'; ORD="" ;; *) LAB='{L} x {D}'; ORD="" ;; esac
echo "plots, once the evaluations have finished (login node):" >&2
echo "  python scripts/plot_study.py --root $OUT --out $OUT/plots/$STUDY --runs ${patterns[*]} --label '$LAB' $ORD --teacher_dir $TEACHER_DIR" >&2
