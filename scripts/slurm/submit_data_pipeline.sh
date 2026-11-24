#!/usr/bin/env bash
# Submit the whole data pipeline as SLURM jobs: every large download is its own job (they run in
# parallel), and processing jobs start when their inputs are complete (--dependency=afterok).
#
#   source scripts/paths.sh
#   bash scripts/slurm/submit_data_pipeline.sh --account <project> [--partition normal] [--dl-partition <p>]
#                                              [--time 24:00:00] [--skip sa1b,ego4d] [--dry-run] [--no-preflight]
#
# --dl-partition: partition for the download jobs (default: --partition; use a data-transfer partition if
#                 your cluster has one and it reaches the internet: `sinfo -s`).
# --skip:         corpus sources not to download / build yet, e.g. while their licence is pending (their jobs
#                 and corpus_finalize are not submitted).
#
# Safe to re-run at any time, e.g. without --skip once a licence has arrived, or after a failed job:
#   * a step that finished leaves $DATA_ROOT/.steps/<step>.ok (prepare_all.sh) and is not submitted again
#     (delete the marker to redo it);
#   * a job of the same name that is still queued or running is reused (new jobs depend on it), never
#     duplicated; one whose dependency failed is cancelled and resubmitted;
#   * everything else is submitted. Jobs are cancelled when a job they depend on fails (--kill-on-invalid-dep).
# Logs: $SLURM_LOG_DIR/<job-name>-<id>.out
set -euo pipefail
: "${ONBOARD_VLM_ROOT:?run 'source scripts/paths.sh' first}"
ACCOUNT="" PART="normal" DLPART="" TIME="24:00:00" SKIP="" DRY="" PREFLIGHT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PART="$2"; shift 2 ;;
    --dl-partition) DLPART="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --skip) SKIP=",$2,"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --no-preflight) PREFLIGHT=""; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$ACCOUNT" ] || [ -n "$DRY" ] || { echo "--account <project> is required" >&2; exit 2; }
DLPART="${DLPART:-$PART}"
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
JOB="$CODE/scripts/slurm/prepare_data.sbatch"
OBJECTS="${OBJECTS_SOURCE:-imagenet1k}"
ME="${USER:-$(id -un)}"
STEPS_DIR="${DATA_ROOT:-$ONBOARD_VLM_ROOT/data}/.steps"   # written by prepare_all.sh after each finished step
skipped() { [[ "$SKIP" == *",$1,"* ]]; }

# ---- preflight: credentials the jobs will need (fail here, not 10 hours into a job) -------------------
problems=()
hf auth whoami >/dev/null 2>&1 || problems+=("Hugging Face: run 'hf auth login' (after 'source scripts/paths.sh'), accept the terms of timm/imagenet-1k-wds")
kaggle competitions files egoschema-public >/dev/null 2>&1 \
  || problems+=("Kaggle: cannot list egoschema-public (EgoSchema videos): API token in ~/.kaggle/access_token (chmod 600), and accept the rules at kaggle.com/competitions/egoschema-public")
skipped sa1b || [ -f "${SA1B_LINKS:-$RAW_ROOT/sa1b/sa1b_links.txt}" ] || problems+=("SA-1B: save the link file as $RAW_ROOT/sa1b/sa1b_links.txt (or --skip sa1b)")
skipped ego4d || grep -q "^\[ego4d\]" "$HOME/.aws/credentials" 2>/dev/null || problems+=("Ego4D: run 'aws configure --profile ego4d' with your Ego4D keys (or --skip ego4d)")
command -v ego4d >/dev/null 2>&1 || skipped ego4d || problems+=("Ego4D CLI: pip install ego4d awscli")
if [ -n "$PREFLIGHT" ] && [ ${#problems[@]} -gt 0 ]; then
  printf 'preflight failed:\n'; printf '  - %s\n' "${problems[@]}"; exit 1
fi

# ---- what is already finished or queued ------------------------------------------------------------------
declare -A QUEUED   # job name -> "<id> <reason>" of this user's pending / running jobs
if command -v squeue >/dev/null 2>&1; then
  while read -r jid jname reason; do QUEUED[$jname]="$jid $reason"; done < <(squeue -u "$ME" -h -o '%i %j %r' 2>/dev/null || true)
fi
step_done() { [ "$1" = check ] || [ -f "$STEPS_DIR/${1//[:\/]/_}.ok" ]; }

sub() {  # sub <name> <partition> <time> <deps (colon-separated, may be empty)> <step...>; prints the job id ("" if done)
  local name="$1" part="$2" time="$3" deps="$4"; shift 4
  local s pending="" q id
  for s in "$@"; do step_done "$s" || pending=1; done
  if [ -z "$pending" ]; then printf '  %-8s ovlm-%s\n' done "$name" >&2; return; fi
  q="${QUEUED[ovlm-$name]:-}"
  if [ -n "$q" ]; then
    if [[ "$q" == *DependencyNeverSatisfied* ]]; then
      printf '  %-8s ovlm-%s (job %s: a job it depends on failed)\n' cancel "$name" "${q%% *}" >&2
      [ -n "$DRY" ] || scancel "${q%% *}"
    else
      printf '  %-8s ovlm-%s (job %s, %s)\n' queued "$name" "${q%% *}" "${q#* }" >&2; echo "${q%% *}"; return
    fi
  fi
  if [ -n "$DRY" ]; then
    printf '  %-8s ovlm-%s (%s, %s) after [%s]: %s\n' submit "$name" "$part" "$time" "${deps//:/ }" "$*" >&2; echo "<$name>"; return
  fi
  id=$(sbatch --parsable --account="$ACCOUNT" --partition="$part" --time="$time" --job-name="ovlm-$name" \
    --output="$SLURM_LOG_DIR/%x-%j.out" --kill-on-invalid-dep=yes ${deps:+--dependency=afterok:$deps} "$JOB" "$@")
  id="${id%%;*}"
  printf '  %-8s ovlm-%s -> job %s after [%s]\n' submit "$name" "$id" "${deps//:/ }" >&2
  echo "$id"
}
join() { local out="" x; for x in "$@"; do [ -n "$x" ] && out="${out:+$out:}$x"; done; echo "$out"; }
declare -A J
echo "data pipeline (${DRY:+dry run, }markers in $STEPS_DIR):" >&2

# ---- downloads (parallel) ------------------------------------------------------------------------------
J[models]=$(sub models "$DLPART" "$TIME" "" models)
for b in vizwiz textvqa wearvqa egoblind egotextvqa egoschema cvbench; do J[dl_$b]=$(sub "dl-$b" "$DLPART" "$TIME" "" "download:$b"); done
J[eval_sets]=$(sub eval-sets "$DLPART" "$TIME" "" eval_sets)
J[probe]=$(sub probe-data "$DLPART" "$TIME" "" probe_data)           # ImageNet-1k train (also the corpus objects source) + NYUv2
for c in text_rich openimages cc12m sa1b epic_kitchens; do
  skipped "$c" || J[dl_$c]=$(sub "dl-$c" "$DLPART" "$TIME" "" "download:$c")
done
[ "$OBJECTS" = imagenet21k ] && J[dl_imagenet21k]=$(sub dl-imagenet21k "$DLPART" "$TIME" "" download:imagenet21k)

# ---- benchmark LMDBs, blocklist, Ego4D ---------------------------------------------------------------------
J[extract]=$(sub bench-extract "$PART" 04:00:00 "$(join "${J[dl_egoblind]}" "${J[dl_egotextvqa]}")" benchmarks_extract)
bench_deps=$(join "${J[dl_vizwiz]}" "${J[dl_textvqa]}" "${J[dl_wearvqa]}" "${J[dl_egoschema]}" "${J[dl_cvbench]}" "${J[extract]}")
J[lmdb]=$(sub bench-lmdb "$PART" "$TIME" "$bench_deps" benchmarks_lmdb)
J[blocklist]=$(sub blocklist "$PART" "$TIME" "$(join "${J[lmdb]}" "${J[eval_sets]}")" blocklist)
skipped ego4d || J[dl_ego4d]=$(sub dl-ego4d "$DLPART" "$TIME" "${J[blocklist]}" download:ego4d)

# ---- corpus sources (each waits for its download and the blocklist), then finalize -----------------------
builds=()
for c in text_rich openimages cc12m sa1b epic_kitchens ego4d "$OBJECTS"; do
  skipped "$c" && continue
  case "$c" in
    imagenet1k) dl="${J[probe]}" ;;
    *) dl="${J[dl_$c]}" ;;
  esac
  J[build_$c]=$(sub "corpus-$c" "$PART" "$TIME" "$(join "$dl" "${J[blocklist]}")" "corpus:$c")
  builds+=("${J[build_$c]}")
done
if [ -z "$SKIP" ]; then
  J[finalize]=$(sub corpus-finalize "$PART" 06:00:00 "$(join "${builds[@]}")" corpus_finalize check)
else
  s="${SKIP#,}"; s="${s%,}"
  echo "NOTE: skipped ${s//,/ }: corpus_finalize is submitted when you re-run this script without --skip." >&2
fi

echo "logs: $SLURM_LOG_DIR | watch: squeue -u $ME -o '%.10i %.28j %.8T %.10M %R' | status: python scripts/check_data.py" >&2
