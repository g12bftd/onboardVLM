#!/usr/bin/env bash
# Every dataset (and the teacher weights) from download to the processed formats the code reads, in
# dependency order. Raw downloads go to $RAW_ROOT and are never modified; everything the code reads is
# written under $PROCESSED_ROOT. Paths come from scripts/paths.sh. On a cluster, run each step as a batch
# job: scripts/slurm/submit_data_pipeline.sh submits all of them with the right dependencies.
#
#   source scripts/paths.sh
#   bash make_datasets/prepare_all.sh                       # every step, in order
#   bash make_datasets/prepare_all.sh download:cc12m        # one download
#   bash make_datasets/prepare_all.sh corpus:cc12m          # one corpus source
#
# Steps (in order):
#   models               Gemma 4 checkpoints at the pinned revisions -> $HF_HOME           (~26 GB for E2B-it + E4B-it)
#   download:<name>      one raw download -> $RAW_ROOT/<name>, resumable. <name> is
#                          a benchmark: vizwiz textvqa wearvqa egoblind egotextvqa egoschema cvbench
#                          a corpus source: text_rich openimages cc12m imagenet1k imagenet21k sa1b epic_kitchens ego4d
#                          probe data: nyuv2
#   benchmarks_download  download:<every benchmark>
#   benchmarks_extract   unpack the EgoBlind video archives (Google Drive) and check that no video is missing
#   benchmarks_lmdb      $RAW_ROOT/<benchmark> -> $LMDB_ROOT/<benchmark>/<split>.lmdb
#   eval_sets            E1 / E3 image sets -> raw $RAW_ROOT/eval_sets/<set>, processed $EVAL_ROOT/<set>
#   probe_data           download:imagenet1k + download:nyuv2, NYUv2 -> $EVAL_ROOT/nyuv2
#   corpus_download      download:<every corpus image source> (objects source = $OBJECTS_SOURCE, default imagenet1k)
#   blocklist            fingerprints of every evaluation image -> $CORPUS_ROOT/eval_blocklist.json
#   ego4d_download       download:ego4d (needs the blocklist: benchmark source videos are never downloaded)
#   corpus_build         every corpus source -> $CORPUS_ROOT/<source>   (one source: corpus:<name>)
#   corpus_finalize      global de-duplication, dataset.json, manifest, CORPUS_CARD.md
#   check                summary of everything that is in place (scripts/check_data.py)
#
# Credentials needed first (docs/setup_cscs.md): `hf auth login` (+ accepted terms of timm/imagenet-1k-wds),
# a Kaggle API token in ~/.kaggle/access_token (EgoSchema), the SA-1B link file ($RAW_ROOT/sa1b/sa1b_links.txt),
# Ego4D AWS keys (`aws configure --profile ego4d`).
#
# Every step that finishes leaves $DATA_ROOT/.steps/<step>.ok (':' -> '_'); submit_data_pipeline.sh does not
# submit a step that has one again (delete the marker to redo it). Running a step by hand ignores the markers.
set -euo pipefail
: "${RAW_ROOT:?run 'source scripts/paths.sh' first}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
TEACHERS="${TEACHERS:-google/gemma-4-E2B-it google/gemma-4-E4B-it}"   # add google/gemma-4-E2B google/gemma-4-E4B for the base models
OBJECTS_SOURCE="${OBJECTS_SOURCE:-imagenet1k}"                         # = objects_source in config/defaults.yaml
BENCHMARKS=(vizwiz textvqa wearvqa egoblind egotextvqa egoschema cvbench)
CORPUS_IMAGE_SOURCES=(text_rich openimages cc12m "$OBJECTS_SOURCE" sa1b epic_kitchens)

STEPS_DIR="${DATA_ROOT:-$(dirname "$RAW_ROOT")}/.steps"   # completion markers (see above)
INCOMPLETE=""                                              # set by a step that skipped part of its work
mkdir -p "$STEPS_DIR"

log() { echo "[prepare $(date '+%F %T')] $*"; }

extract_any() {  # extract_any <archive> <dest>: zip / tar(.gz|.bz2|.xz) detected from content
  python - "$1" "$2" <<'PY'
import os, sys, tarfile, zipfile
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
if zipfile.is_zipfile(src):
    zipfile.ZipFile(src).extractall(dst)
elif tarfile.is_tarfile(src):
    with tarfile.open(src) as t:
        t.extractall(dst, filter="data")
else:
    sys.exit(f"{src}: not a zip or tar archive; extract it manually into {dst}")
print(f"extracted {src} -> {dst}")
PY
}

step_models() {
  python - $TEACHERS <<'PY'
import sys
from huggingface_hub import snapshot_download
from onboard_vlm.teachers.gemma4 import SUPPORTED_TEACHERS
for model_id in sys.argv[1:]:
    rev = SUPPORTED_TEACHERS[model_id]
    print(f"{model_id}@{rev[:7]} ->", snapshot_download(model_id, revision=rev), flush=True)
PY
}

download() {  # download <name>
  local n="$1"
  case "$n" in
    vizwiz|textvqa|wearvqa|egoblind|egotextvqa|egoschema|cvbench) bash "$HERE/download/download_$n.sh" ;;
    sa1b)
      export SA1B_LINKS="${SA1B_LINKS:-$RAW_ROOT/sa1b/sa1b_links.txt}"   # link file from the SA-1B download page
      if [ -f "$SA1B_LINKS" ]; then bash "$HERE/download/corpus/download_sa1b.sh"
      else log "WARNING: $SA1B_LINKS missing: SA-1B skipped (docs/setup_cscs.md)"; INCOMPLETE=1; fi ;;
    text_rich|openimages|cc12m|imagenet1k|imagenet21k|epic_kitchens|ego4d) bash "$HERE/download/corpus/download_$n.sh" ;;
    nyuv2)
      mkdir -p "$RAW_ROOT/nyuv2"
      curl -L --fail --retry 5 -C - -o "$RAW_ROOT/nyuv2/nyu_depth_v2_labeled.mat" http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
      curl -L --fail --retry 5 -C - -o "$RAW_ROOT/nyuv2/splits.mat" http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat ;;
    *) echo "unknown download '$n'" >&2; return 2 ;;
  esac
}

step_benchmarks_download() { local b; for b in "${BENCHMARKS[@]}"; do log "download:$b"; download "$b"; done; }

step_benchmarks_extract() {
  # EgoBlind: the video archives of the Train / Test Google Drive folders -> $RAW_ROOT/egoblind/videos/<train|test>/.../<archive>/
  local a rel
  while IFS= read -r a; do
    [ -f "$a.extracted" ] && continue
    rel="${a#"$RAW_ROOT/egoblind/"}"
    extract_any "$a" "$RAW_ROOT/egoblind/videos/${rel%%.*}" && touch "$a.extracted"
  done < <(find "$RAW_ROOT/egoblind/train" "$RAW_ROOT/egoblind/test" -type f \( -iname "*.zip" -o -iname "*.tar" \
             -o -iname "*.tar.gz" -o -iname "*.tgz" \) 2>/dev/null | sort)
  # complete? every test video, and (almost) every train video
  python -I "$HERE/download/verify_videos.py" --name egoblind --video_dir "$RAW_ROOT/egoblind" --key video_name \
    --ids "$RAW_ROOT/egoblind/test.csv" "$RAW_ROOT/egoblind/train.csv" --required "$RAW_ROOT/egoblind/test.csv" --max_missing 0.01
}

step_benchmarks_lmdb() {
  if [ -f "$LMDB_ROOT/.done" ]; then log "benchmark LMDBs already built (rm $LMDB_ROOT/.done to rebuild)"; return; fi
  rm -rf "${LMDB_ROOT:?}"/*  # LMDB writers refuse duplicate keys: always build into an empty directory
  bash "$HERE/build_all_lmdb.sh"
  touch "$LMDB_ROOT/.done"
}

step_eval_sets() { python -u "$HERE/eval_sets/build_eval_sets.py" --raw_root "$RAW_ROOT/eval_sets" --eval_root "$EVAL_ROOT" --workers "${EVAL_WORKERS:-8}"; }

step_probe_data() {
  download imagenet1k
  download nyuv2
  [ -f "$EVAL_ROOT/nyuv2/.done" ] || { rm -rf "$EVAL_ROOT/nyuv2"; python "$HERE/eval_sets/make_nyuv2.py" --mat "$RAW_ROOT/nyuv2/nyu_depth_v2_labeled.mat" \
    --splits "$RAW_ROOT/nyuv2/splits.mat" --out "$EVAL_ROOT/nyuv2" && touch "$EVAL_ROOT/nyuv2/.done"; }
}

step_corpus_download() { local c; for c in "${CORPUS_IMAGE_SOURCES[@]}"; do log "download:$c"; download "$c"; done; }

step_blocklist() { bash "$HERE/build_corpus.sh" blocklist; }
step_ego4d_download() { download ego4d; }
step_corpus_build() { bash "$HERE/build_corpus.sh" text_rich openimages sa1b "$OBJECTS_SOURCE" cc12m ego4d epic_kitchens; }
step_corpus_finalize() { bash "$HERE/build_corpus.sh" finalize; }
step_check() { python "$REPO/scripts/check_data.py"; }

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(models benchmarks_download benchmarks_extract benchmarks_lmdb eval_sets probe_data corpus_download blocklist ego4d_download corpus_build corpus_finalize check)
for s in "${STEPS[@]}"; do
  log "===> $s"
  INCOMPLETE=""
  case "$s" in
    corpus:*) bash "$HERE/build_corpus.sh" "${s#corpus:}" ;;
    download:*) download "${s#download:}" ;;
    *) "step_$s" ;;
  esac
  if [ -z "$INCOMPLETE" ] && [ "$s" != check ]; then
    echo "$(date '+%F %T') job ${SLURM_JOB_ID:-none} on $(hostname)" > "$STEPS_DIR/${s//[:\/]/_}.ok"
  fi
  log "<=== $s done${INCOMPLETE:+ (incomplete: not marked as finished)}"
done
