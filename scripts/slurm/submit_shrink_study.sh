#!/usr/bin/env bash
# Submit a controlled study: one training job per (configuration, seed), each followed (afterok) by the evaluation of
# its trajectory exports (scripts/trajectory_eval.py: E1 / E2 / E3 on every export, the full general suite on the
# last), plus the teacher references (and the blind floor of the general suite) in every VLM involved.
# Prints the plot and table commands at the end. The order of the studies: docs/methods.md ("Building up").
#
#   source scripts/paths.sh
#   bash scripts/slurm/submit_shrink_study.sh --account <project> --study <study> [--seeds 0,1,2] [--dry-run] [-- <hydra overrides>]
#
# Every run starts from the simplest recipe (+experiment=lit/base): student 8 x 768 from the teacher (--student),
# whole images at 280 tokens (--input full280), one MSE term in the LM-input space (--objective mse), no mixup,
# lr 0.001 (--lr). A study changes one factor of it; the flags carry what earlier studies selected.
#   references  no training: the teacher in every VLM of --eval-vlms (ceiling) and the blind floor
#   identity    the teacher itself as the student (16 x 768): does training keep it exact? (pipeline check, 1 seed)
#   lr          learning rate {1e-4, 3e-4, 1e-3} of the simplest recipe (chosen on the val split, 1 seed)
#   target      where to match: vision tokens (mseV, one student for both VLMs) | E2B's | E4B's LM-input space;
#               every student evaluated in both VLMs
#   input       full280 -> full (every budget) -> aspect (aspect-ratio coverage) -> crop (scale augmentation)
#   objective   mse -> fm (+ cosine); md, fm_md (token relations, Hao et al.); fm_fmL, fm_mdL (+ intermediate layers)
#   md_scale    fm_md with the relation weights x {1, 10, 100, 1000} (1 seed; choose on held-out data)
#   mixup       --objective without / with consistent mixup (Beyer et al.)
#   multiplex   weight sharing (MiniViT): same weights {8 x 1, 16 x 2 none | linear | dwconv}, same depth {8 x 1, 8 x 2, 4 x 1}
#   depth / width / init / grid   the size axis (+experiment=shrink/base, same flags)
#
# --teacher gemma4_e2b | gemma4_e4b: the LM-input space distilled into (in every run name); --eval-vlms e2b,e4b also
# evaluates every student in the other VLM (E3 + general suite, in eval_vlm_<key>/); --only '<glob>' keeps the
# configurations whose run name matches. Re-running is safe: finished runs ($ONBOARD_VLM_OUT/students/<run_id>) and
# finished evaluations are skipped, queued or running jobs of the same name are reused. Overrides after `--` go to
# every training run (e.g. -- allow_missing_sources=true). --list prints "<run name>|<overrides>" per configuration.
set -euo pipefail
: "${ONBOARD_VLM_ROOT:?run 'source scripts/paths.sh' first}"
ACCOUNT="" STUDY="" SEEDS="0,1,2" TIME="24:00:00" EVAL_TIME="24:00:00" DRY="" LIST="" TEACHER_CFG="gemma4_e2b"
LIT_STUDENT="8 768" LIT_OBJECTIVE="mse" LIT_INPUT="full280" LIT_LR="0.001" MD_SCALE="1.0" LMMS_STEPS="last" EVALS="e1,e2,e3,lmms"
EVAL_VLMS="" ONLY="*"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --study) STUDY="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --teacher) TEACHER_CFG="$2"; shift 2 ;;          # gemma4_e2b | gemma4_e4b: the LM-input space distilled into
    --eval-vlms) EVAL_VLMS="$2"; shift 2 ;;          # e.g. e2b,e4b: also evaluate every student in these VLMs
    --only) ONLY="$2"; shift 2 ;;                    # glob over run names
    --student) LIT_STUDENT="$2"; shift 2 ;;          # "<layers> <width>" (default 8 768: half the teacher's layers)
    --objective) LIT_OBJECTIVE="$2"; shift 2 ;;      # objective of every study but objective / target (config/objective/)
    --input) LIT_INPUT="$2"; shift 2 ;;              # input recipe of every study but input (config/input/)
    --lr) LIT_LR="$2"; shift 2 ;;                    # learning rate, as a decimal (0.0003), of every study but lr
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

cfg_of() { case "$1" in gemma4_*) echo "$1" ;; *) echo "gemma4_$1" ;; esac; }   # e4b -> gemma4_e4b
key_of() { case "$1" in gemma4_e2b) echo e2b_it ;; gemma4_e4b) echo e4b_it ;; *) echo "${1#gemma4_}" ;; esac; }   # = teacher.key
vlm_of() { case "$1" in gemma4_e4b|gemma4_e4b_base) echo google/gemma-4-E4B-it ;; *) echo google/gemma-4-E2B-it ;; esac; }
TEACHER_CFG="$(cfg_of "$TEACHER_CFG")"

# configurations: "<run name>|<hydra overrides>"; the name is the experiment_name the overrides produce (tests check it)
mdo() { case "$1" in *md*) echo "_w${2:-$MD_SCALE}|md.scale=${2:-$MD_SCALE}" ;; *) echo "|" ;; esac; }   # MD scale: name tag | override
shrink() {  # shrink <layers> <width> <init> <layer_selection> <width_selection>   (env: T, IN, OBJ, LR)
  local t="${T:-$TEACHER_CFG}" in="${IN:-$LIT_INPUT}" obj="${OBJ:-$LIT_OBJECTIVE}" lr="${LR:-$LIT_LR}" m; m="$(mdo "${OBJ:-$LIT_OBJECTIVE}")"
  echo "shrink_$(key_of "$t")_${in}_${obj}${m%%|*}_lr${lr}_L$1_D$2_$3_$4-$5|+experiment=shrink/base teacher=$t input=$in objective=$obj task.optimizer.lr=$lr student.num_hidden_layers=$1 student.hidden_size=$2 student.init=$3 student.layer_selection=$4 student.width_selection=$5 ${m#*|}"
}
lit() {  # lit <objective> <mixup alpha> <layers> <width> <share_every> <share_transform> [md scale]   (env: T, IN, LR)
  local t="${T:-$TEACHER_CFG}" in="${IN:-$LIT_INPUT}" lr="${LR:-$LIT_LR}" m; m="$(mdo "$1" "${7:-}")"
  echo "lit_$(key_of "$t")_${in}_$1${m%%|*}_mx$2_lr${lr}_L$3_D$4_x$5-$6|+experiment=lit/base teacher=$t input=$in objective=$1 data_module.collator.mixup_alpha=$2 task.optimizer.lr=$lr student.num_hidden_layers=$3 student.hidden_size=$4 student.share_every=$5 student.share_transform=$6 ${m#*|}"
}
names() { local n=""; for i in "$@"; do n+="${CONFIGS[$i]%%|*} "; done; echo "${n% }"; }
read -r SL SD <<< "$LIT_STUDENT"
O="$LIT_OBJECTIVE"
PLOTS=()  # "<plot dir>|<label>|<order>|<run names...>"
case "$STUDY" in
  references) CONFIGS=() ;;
  identity) CONFIGS=("$(lit "$O" 0.0 16 768 1 none)") ;;
  lr) CONFIGS=(); for v in 0.0001 0.0003 0.001; do CONFIGS+=("$(LR=$v lit "$O" 0.0 "$SL" "$SD" 1 none)"); done
    PLOTS=("lr|lr {lr}|lr|$(names 0 1 2)") ;;
  target)
    CONFIGS=("$(T=gemma4_e2b lit mseV 0.0 "$SL" "$SD" 1 none)" "$(T=gemma4_e2b lit mse 0.0 "$SL" "$SD" 1 none)"
             "$(T=gemma4_e4b lit mse 0.0 "$SL" "$SD" 1 none)")
    EVAL_VLMS="${EVAL_VLMS:-e2b,e4b}"
    PLOTS=("target|{teacher} {obj}||$(names 0 1 2)") ;;
  input) CONFIGS=(); for v in full280 full aspect crop; do CONFIGS+=("$(IN=$v lit "$O" 0.0 "$SL" "$SD" 1 none)"); done
    PLOTS=("input|{inp}||$(names 0 1 2 3)") ;;
  objective)
    CONFIGS=("$(lit mse 0.0 "$SL" "$SD" 1 none)" "$(lit fm 0.0 "$SL" "$SD" 1 none)" "$(lit md 0.0 "$SL" "$SD" 1 none)"
             "$(lit fm_md 0.0 "$SL" "$SD" 1 none)" "$(lit fm_fmL 0.0 "$SL" "$SD" 1 none)" "$(lit fm_mdL 0.0 "$SL" "$SD" 1 none)")
    PLOTS=("objective_outputs|{obj}||$(names 0 1 2 3)" "objective_layers|{obj}||$(names 1 4 5)") ;;
  md_scale)
    CONFIGS=(); for W in 1.0 10.0 100.0 1000.0; do CONFIGS+=("$(lit fm_md 0.0 "$SL" "$SD" 1 none "$W")"); done
    PLOTS=("md_scale|MD x {w}|w|$(names 0 1 2 3)") ;;
  mixup) CONFIGS=("$(lit "$O" 0.0 "$SL" "$SD" 1 none)" "$(lit "$O" 1.0 "$SL" "$SD" 1 none)")
    PLOTS=("mixup|mixup alpha {mx}||$(names 0 1)") ;;
  multiplex)
    CONFIGS=("$(lit "$O" 0.0 8 "$SD" 1 none)" "$(lit "$O" 0.0 16 "$SD" 2 none)" "$(lit "$O" 0.0 16 "$SD" 2 linear)"
             "$(lit "$O" 0.0 16 "$SD" 2 dwconv)" "$(lit "$O" 0.0 8 "$SD" 2 linear)" "$(lit "$O" 0.0 4 "$SD" 1 none)")
    PLOTS=("multiplex_same_weights|{L} layers, shared x{k} ({tr})||$(names 0 1 2 3)"
           "multiplex_same_depth|{L} layers, shared x{k} ({tr})||$(names 0 4 5)") ;;
  depth) CONFIGS=(); for L in 12 8 6 4 2; do CONFIGS+=("$(shrink "$L" 768 teacher uniform uniform)"); done ;;
  width) CONFIGS=(); for D in 512 384 256 128; do CONFIGS+=("$(shrink 16 "$D" teacher uniform uniform)"); done ;;
  init) CONFIGS=("$(shrink 8 384 teacher uniform uniform)" "$(shrink 8 384 teacher first first)" "$(shrink 8 384 random uniform uniform)") ;;
  grid) CONFIGS=(); for L in 4 8 12; do for D in 256 384 512; do CONFIGS+=("$(shrink "$L" "$D" teacher uniform uniform)"); done; done ;;
  *) echo "--study references | identity | lr | target | input | objective | md_scale | mixup | multiplex | depth | width | init | grid" >&2; exit 2 ;;
esac
KEPT=(); for c in "${CONFIGS[@]}"; do [[ "${c%%|*}" == $ONLY ]] && KEPT+=("$c"); done; CONFIGS=("${KEPT[@]}")
if [ -n "$LIST" ]; then [ ${#CONFIGS[@]} -eq 0 ] || printf '%s\n' "${CONFIGS[@]}"; exit 0; fi

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
count() { find "$@" 2>/dev/null | wc -l || true; }
has_lmms() { case ",$EVALS," in *,lmms,*) return 0 ;; *) return 1 ;; esac; }

# VLMs everything is evaluated in: the run teachers' own, plus --eval-vlms
declare -A VLMS
VLMS[$TEACHER_CFG]=1
for c in "${CONFIGS[@]}"; do for o in ${c#*|}; do case "$o" in teacher=*) VLMS[${o#teacher=}]=1 ;; esac; done; done
for v in ${EVAL_VLMS//,/ }; do VLMS[$(cfg_of "$v")]=1; done
tdir() { echo "$OUT/trajectories/_teacher_${1#gemma4_}"; }

echo "study '$STUDY': ${#CONFIGS[@]} configurations x seeds {$SEEDS}, evaluated in ${!VLMS[*]} (${DRY:+dry run, }outputs in $OUT)" >&2
for v in "${!VLMS[@]}"; do  # teacher references (and the blind floor) in every VLM
  TD="$(tdir "$v")"
  if [ -f "$TD/eval/e2/results.json" ] && [ -f "$TD/eval/e3/cvbench/metrics.json" ] && { ! has_lmms || [ "$(count "$TD/eval/lmms_blind" -name '*_results.json')" -ge 6 ]; }; then
    echo "  done     teacher reference ($TD)" >&2
  else
    sub "ovlm-eval-teacher-${v#gemma4_}" "$EVAL_TIME" "" scripts/trajectory_eval.py --run_dir "$TD" --teacher \
      --teacher_config "$v" --evals "$EVALS" --gpus 4 >/dev/null
  fi
done
patterns=()
for c in "${CONFIGS[@]}"; do
  NAME="${c%%|*}"; read -r -a OVR <<< "${c#*|}"
  RT="$TEACHER_CFG"; for o in "${OVR[@]}"; do case "$o" in teacher=*) RT="${o#teacher=}" ;; esac; done
  patterns+=("$NAME")
  for s in ${SEEDS//,/ }; do
    RUN="${NAME}_s${s}"; TR="$OUT/trajectories/$RUN"
    if [ -f "$OUT/students/$RUN/student.safetensors" ]; then
      echo "  done     $RUN (trained)" >&2; train=""
    else
      train=$(sub "$RUN" "$TIME" "" run_distill.py "${OVR[@]}" seed="$s" "${EXTRA[@]}")
    fi
    n_steps=$(count "$TR" -maxdepth 1 -name 'step_*')
    for v in "${!VLMS[@]}"; do  # in its own VLM: E1 / E2 / E3 / suite; in another: E3 / suite (eval_vlm_<key>/)
      if [ "$v" = "$RT" ]; then ED="eval"; EV="$EVALS"; JN="eval-$RUN"; VA=()
      else ED="eval_vlm_$(key_of "$v")"; EV="$(echo ",$EVALS," | sed 's/,e1,/,/; s/,e2,/,/; s/^,//; s/,$//')"; JN="eval-${RUN}-in-${v#gemma4_}"
        VA=(--vlm_id "$(vlm_of "$v")" --eval_tag "vlm_$(key_of "$v")"); fi
      [ -n "$EV" ] || continue
      n_done=$(count "$TR" -path "*/$ED/e3/cvbench/metrics.json")
      n_lmms=$(count "$TR" -path "*/$ED/lmms/*" -name '*_results.json')
      if [ -z "$train" ] && [ "$n_steps" -gt 0 ] && [ "$n_done" -eq "$n_steps" ] && { ! has_lmms || [ "$n_lmms" -ge 6 ]; }; then
        echo "  done     $JN ($n_steps checkpoints evaluated)" >&2
      else
        sub "$JN" "$EVAL_TIME" "$train" scripts/trajectory_eval.py --run_dir "$TR" --teacher_config "$RT" \
          --evals "$EV" --lmms_steps "$LMMS_STEPS" "${VA[@]}" --gpus 4 >/dev/null
      fi
    done
  done
done
[ ${#CONFIGS[@]} -gt 0 ] || exit 0
if [ ${#PLOTS[@]} -eq 0 ]; then
  case "$STUDY" in depth) LAB='{L} layers'; ORD="L" ;; width) LAB='width {D}'; ORD="D" ;;
    init) LAB='{init} ({ls})'; ORD="" ;; grid) LAB='L{L} D{D}'; ORD="" ;; *) LAB='{L} x {D}'; ORD="" ;; esac
  PLOTS=("$STUDY|$LAB|$ORD|${patterns[*]}")
fi
for v in "${!VLMS[@]}"; do
  K="$(key_of "$v")"; TD="$(tdir "$v")"
  echo "plots in $v, once the evaluations have finished (login node):" >&2
  for pl in "${PLOTS[@]}"; do
    IFS='|' read -r PDIR LAB ORD RUNS <<< "$pl"
    echo "  python scripts/plot_study.py --root $OUT --out $OUT/plots/${PDIR}_in_${K} --runs $RUNS --label '$LAB'${ORD:+ --order $ORD} --vlm $K --teacher_dir $TD" >&2
  done
  AGG=""
  for n in "${patterns[@]}"; do
    case "$n" in *_"$K"_*) ED="eval" ;; *) ED="eval_vlm_$K" ;; esac
    for s in ${SEEDS//,/ }; do AGG+=" --lmms $n=\$(ls -d $OUT/trajectories/${n}_s${s}/step_* | tail -1)/$ED/lmms"; done
  done
  echo "general-suite table in $v (seed mean +- std, retention and visual retention vs its teacher):" >&2
  echo "  python scripts/aggregate_results.py --out $OUT/tables/${STUDY}_in_${K} --lmms teacher=$TD/eval/lmms --lmms blind=$TD/eval/lmms_blind$AGG --reference teacher --blind blind" >&2
done
