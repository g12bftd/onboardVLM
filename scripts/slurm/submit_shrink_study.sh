#!/usr/bin/env bash
# Submit a study: one training job per (configuration, seed), each followed (afterok) by the evaluation of its
# trajectory exports (scripts/trajectory_eval.py: E1 / E2 / E3 on every export, the full general suite on the last),
# plus one evaluation of the teacher (reference lines, and the blind control of the general suite).
# Prints the plot commands at the end.
#
#   source scripts/paths.sh
#   bash scripts/slurm/submit_shrink_study.sh --account <project> --study depth [--seeds 0,1,2] [--dry-run] [-- <hydra overrides>]
#
# Shrinking studies (docs/shrink_study.md; +experiment=shrink/base, one factor changes):
#   identity  16 x 768, init teacher: the teacher itself (sanity anchor: fidelity 1 at step 0)
#   depth     {12, 8, 6, 4, 2} layers x 768, init teacher (uniform layer selection)
#   width     16 layers x {512, 384, 256, 128}, init teacher (weight selection)
#   init      8 x 384: teacher uniform | teacher first | random
#   grid      {4, 8, 12} layers x {256, 384, 512}, init teacher
# Literature studies (docs/methods.md; +experiment=lit/base, student --student "8 384" from the teacher):
#   objective  (1) FM, FM without mixup, (2) MD, FM+MD on output tokens; (3) FM + FM / MD on intermediate layers
#   md_scale   FM + MD with the MD weights x {1, 10, 100, 1000}: choose the scale on held-out data (1 seed is enough)
#   multiplex  (4) weight sharing at the width of --student: same weights {8 x 1, 16 x 2 none | linear | dwconv},
#              same depth {8 x 1, 8 x 2 linear, 4 x 1} (objective --objective, default fm)
# Re-running is safe: finished runs ($ONBOARD_VLM_OUT/students/<run_id>) and finished evaluations are skipped,
# queued or running jobs of the same name are reused. Overrides after `--` go to every training run
# (e.g. -- allow_missing_sources=true trainer.max_epochs=10). --list prints "<run name>|<overrides>" per configuration.
set -euo pipefail
: "${ONBOARD_VLM_ROOT:?run 'source scripts/paths.sh' first}"
ACCOUNT="" STUDY="" SEEDS="0,1,2" TIME="24:00:00" EVAL_TIME="24:00:00" DRY="" LIST="" TEACHER_CFG="gemma4_e2b"
LIT_STUDENT="8 384" LIT_OBJECTIVE="fm" MD_SCALE="1.0" LMMS_STEPS="last" EVALS="e1,e2,e3,lmms"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --study) STUDY="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --teacher) TEACHER_CFG="$2"; shift 2 ;;
    --student) LIT_STUDENT="$2"; shift 2 ;;          # "<layers> <width>" of the literature studies
    --objective) LIT_OBJECTIVE="$2"; shift 2 ;;      # objective of the multiplex study (config/objective/*.yaml)
    --md-scale) MD_SCALE="$2"; shift 2 ;;            # MD weight scale of the objective / multiplex studies (a decimal, e.g. 10.0)
    --evals) EVALS="$2"; shift 2 ;;                  # trajectory evaluations (drop lmms to skip the full general suite)
    --lmms-steps) LMMS_STEPS="$2"; shift 2 ;;        # exports that get the full general suite: last | first,last | all
    --list) LIST=1; DRY=1; shift ;;
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

# configurations: "<run name>|<hydra overrides>"; the name is the experiment_name the overrides produce (tests check it)
shrink() {  # shrink <layers> <width> <init> <layer_selection> <width_selection>
  echo "shrink_L$1_D$2_$3_$4-$5|+experiment=shrink/base student.num_hidden_layers=$1 student.hidden_size=$2 student.init=$3 student.layer_selection=$4 student.width_selection=$5"
}
lit() {  # lit <objective> <mixup alpha> <layers> <width> <share_every> <share_transform> [md scale]
  local obj="$1" tag="$1" md=""
  case "$obj" in *md*) tag="${obj}_w${7:-$MD_SCALE}"; md=" md.scale=${7:-$MD_SCALE}" ;; esac
  echo "lit_${tag}_mx$2_L$3_D$4_x$5-$6|+experiment=lit/base objective=$obj data_module.collator.mixup_alpha=$2 student.num_hidden_layers=$3 student.hidden_size=$4 student.share_every=$5 student.share_transform=$6$md"
}
read -r SL SD <<< "$LIT_STUDENT"
PLOTS=()  # "<plot dir>|<label>|<order>|<run names...>"
case "$STUDY" in
  identity) CONFIGS=("$(shrink 16 768 teacher uniform uniform)") ;;
  depth) CONFIGS=(); for L in 12 8 6 4 2; do CONFIGS+=("$(shrink "$L" 768 teacher uniform uniform)"); done ;;
  width) CONFIGS=(); for D in 512 384 256 128; do CONFIGS+=("$(shrink 16 "$D" teacher uniform uniform)"); done ;;
  init) CONFIGS=("$(shrink 8 384 teacher uniform uniform)" "$(shrink 8 384 teacher first first)" "$(shrink 8 384 random uniform uniform)") ;;
  grid) CONFIGS=(); for L in 4 8 12; do for D in 256 384 512; do CONFIGS+=("$(shrink "$L" "$D" teacher uniform uniform)"); done; done ;;
  objective)
    CONFIGS=("$(lit fm 1.0 "$SL" "$SD" 1 none)" "$(lit fm 0.0 "$SL" "$SD" 1 none)" "$(lit md 1.0 "$SL" "$SD" 1 none)"
             "$(lit fm_md 1.0 "$SL" "$SD" 1 none)" "$(lit fm_fmL 1.0 "$SL" "$SD" 1 none)" "$(lit fm_mdL 1.0 "$SL" "$SD" 1 none)")
    PLOTS=("objective_outputs|{obj} (mixup {mx})||${CONFIGS[0]%%|*} ${CONFIGS[1]%%|*} ${CONFIGS[2]%%|*} ${CONFIGS[3]%%|*}"
           "objective_layers|{obj}||${CONFIGS[0]%%|*} ${CONFIGS[4]%%|*} ${CONFIGS[5]%%|*}") ;;
  md_scale)
    CONFIGS=(); for W in 1.0 10.0 100.0 1000.0; do CONFIGS+=("$(lit fm_md 1.0 "$SL" "$SD" 1 none "$W")"); done
    PLOTS=("md_scale|MD x {w}|w|$(for c in "${CONFIGS[@]}"; do printf '%s ' "${c%%|*}"; done)") ;;
  multiplex)
    O="$LIT_OBJECTIVE"
    CONFIGS=("$(lit "$O" 1.0 8 "$SD" 1 none)" "$(lit "$O" 1.0 16 "$SD" 2 none)" "$(lit "$O" 1.0 16 "$SD" 2 linear)"
             "$(lit "$O" 1.0 16 "$SD" 2 dwconv)" "$(lit "$O" 1.0 8 "$SD" 2 linear)" "$(lit "$O" 1.0 4 "$SD" 1 none)")
    PLOTS=("multiplex_same_weights|{L} layers, shared x{k} ({tr})||${CONFIGS[0]%%|*} ${CONFIGS[1]%%|*} ${CONFIGS[2]%%|*} ${CONFIGS[3]%%|*}"
           "multiplex_same_depth|{L} layers, shared x{k} ({tr})||${CONFIGS[0]%%|*} ${CONFIGS[4]%%|*} ${CONFIGS[5]%%|*}") ;;
  *) echo "--study identity | depth | width | init | grid | objective | md_scale | multiplex" >&2; exit 2 ;;
esac
if [ -n "$LIST" ]; then printf '%s\n' "${CONFIGS[@]}"; exit 0; fi

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

echo "study '$STUDY': ${#CONFIGS[@]} configurations x seeds {$SEEDS} (${DRY:+dry run, }outputs in $OUT)" >&2
TEACHER_DIR="$OUT/trajectories/_teacher_${TEACHER_CFG#gemma4_}"
LMMS_DONE=1
case ",$EVALS," in *,lmms,*) [ "$(find "$TEACHER_DIR/eval/lmms_blind" -name '*_results.json' 2>/dev/null | wc -l || true)" -ge 6 ] || LMMS_DONE="" ;; esac
if [ -f "$TEACHER_DIR/eval/e2/results.json" ] && [ -f "$TEACHER_DIR/eval/e3/cvbench/metrics.json" ] && [ -n "$LMMS_DONE" ]; then
  echo "  done     teacher reference ($TEACHER_DIR)" >&2
else
  sub "ovlm-eval-teacher-${TEACHER_CFG#gemma4_}" "$EVAL_TIME" "" scripts/trajectory_eval.py --run_dir "$TEACHER_DIR" --teacher \
    --teacher_config "$TEACHER_CFG" --evals "$EVALS" --gpus 4 >/dev/null
fi
patterns=()
for c in "${CONFIGS[@]}"; do
  NAME="${c%%|*}"; read -r -a OVR <<< "${c#*|}"
  patterns+=("$NAME")
  for s in ${SEEDS//,/ }; do
    RUN="${NAME}_s${s}"
    if [ -f "$OUT/students/$RUN/student.safetensors" ]; then
      echo "  done     $RUN (trained)" >&2; train=""
    else
      train=$(sub "$RUN" "$TIME" "" run_distill.py "${OVR[@]}" teacher="$TEACHER_CFG" seed="$s" "${EXTRA[@]}")
    fi
    n_steps=$(find "$OUT/trajectories/$RUN" -maxdepth 1 -name 'step_*' 2>/dev/null | wc -l || true)
    n_done=$(find "$OUT/trajectories/$RUN" -path '*/eval/e3/cvbench/metrics.json' 2>/dev/null | wc -l || true)
    n_lmms=$(find "$OUT/trajectories/$RUN" -path '*/eval/lmms/*' -name '*_results.json' 2>/dev/null | wc -l || true)
    case ",$EVALS," in *,lmms,*) need_lmms=1 ;; *) need_lmms=0 ;; esac
    if [ -z "$train" ] && [ "$n_steps" -gt 0 ] && [ "$n_done" -eq "$n_steps" ] && { [ "$need_lmms" -eq 0 ] || [ "$n_lmms" -ge 6 ]; }; then
      echo "  done     eval-$RUN ($n_steps checkpoints evaluated)" >&2
    else
      sub "eval-$RUN" "$EVAL_TIME" "$train" scripts/trajectory_eval.py --run_dir "$OUT/trajectories/$RUN" --teacher_config "$TEACHER_CFG" \
        --evals "$EVALS" --lmms_steps "$LMMS_STEPS" --gpus 4 >/dev/null
    fi
  done
done
if [ ${#PLOTS[@]} -eq 0 ]; then
  case "$STUDY" in depth) LAB='{L} layers'; ORD="L" ;; width) LAB='width {D}'; ORD="D" ;;
    init) LAB='{init} ({ls})'; ORD="" ;; grid) LAB='L{L} D{D}'; ORD="" ;; *) LAB='{L} x {D}'; ORD="" ;; esac
  PLOTS=("$STUDY|$LAB|$ORD|${patterns[*]}")
fi
echo "plots, once the evaluations have finished (login node):" >&2
for pl in "${PLOTS[@]}"; do
  IFS='|' read -r PDIR LAB ORD RUNS <<< "$pl"
  echo "  python scripts/plot_study.py --root $OUT --out $OUT/plots/$PDIR --runs $RUNS --label '$LAB'${ORD:+ --order $ORD} --teacher_dir $TEACHER_DIR" >&2
done
echo "general-suite table (seed mean +- std, retention and visual retention vs the teacher), once evaluated:" >&2
AGG=""
for n in "${patterns[@]}"; do
  for s in ${SEEDS//,/ }; do AGG+=" --lmms $n=\$(ls -d $OUT/trajectories/${n}_s${s}/step_* | tail -1)/eval/lmms"; done
done
echo "  python scripts/aggregate_results.py --out $OUT/tables/$STUDY --lmms teacher=$TEACHER_DIR/eval/lmms --lmms blind=$TEACHER_DIR/eval/lmms_blind$AGG --reference teacher --blind blind" >&2
