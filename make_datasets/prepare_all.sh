#!/usr/bin/env bash
# Every dataset (and the teacher weights) from download to the processed formats the code reads, in
# dependency order. Raw downloads go to $RAW_ROOT and are never modified; everything the code reads is
# written under $PROCESSED_ROOT. Paths come from scripts/paths.sh.
#
#   source scripts/paths.sh
#   bash make_datasets/prepare_all.sh                 # every step, in order
#   bash make_datasets/prepare_all.sh eval_sets check # selected steps
#
# Steps (in order):
#   models             Gemma 4 checkpoints at the pinned revisions -> $HF_HOME            (~26 GB for E2B-it + E4B-it)
#   benchmarks_download  VizWiz, TextVQA, WearVQA, EgoBlind, EgoTextVQA, EgoSchema, CV-Bench -> $RAW_ROOT/<name>
#   benchmarks_extract   unpack the video archives fetched from Google Drive (EgoBlind, EgoTextVQA)
#   benchmarks_lmdb      $RAW_ROOT/<name> -> $LMDB_ROOT/<name>/<split>.lmdb
#   eval_sets            E1 / E3 image sets -> raw $RAW_ROOT/eval_sets/<set>, processed $EVAL_ROOT/<set>
#   probe_data           E2: ImageNet-1k train tars -> $RAW_ROOT/imagenet1k; NYUv2 -> $RAW_ROOT/nyuv2 -> $EVAL_ROOT/nyuv2
#   corpus_download      CC12M, ImageNet-21k, Open Images, SA-1B, text-rich, EPIC-KITCHENS -> $RAW_ROOT/<source>
#   blocklist            fingerprints of every evaluation image -> $CORPUS_ROOT/eval_blocklist.json
#   ego4d_download       Ego4D subset (needs the blocklist: benchmark source videos are never downloaded)
#   corpus_build         every corpus source -> $CORPUS_ROOT/<source>   (one source: corpus:<name>, e.g. corpus:cc12m)
#   corpus_finalize      global de-duplication, dataset.json, manifest, CORPUS_CARD.md
#   check                summary of everything that is in place (scripts/check_data.py)
#
# Credentials needed first (docs/setup_cscs.md): `hf auth login` (+ accepted terms for timm/imagenet-22k-wds
# and timm/imagenet-1k-wds), ~/.kaggle/kaggle.json (EgoSchema), the SA-1B link file ($SA1B_LINKS),
# Ego4D AWS keys (`aws configure --profile ego4d`).
set -euo pipefail
: "${RAW_ROOT:?run 'source scripts/paths.sh' first}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
TEACHERS="${TEACHERS:-google/gemma-4-E2B-it google/gemma-4-E4B-it}"   # add google/gemma-4-E2B google/gemma-4-E4B for the base models

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

step_benchmarks_download() { bash "$HERE/download/download_all.sh"; }

step_benchmarks_extract() {
  local f
  for f in "$RAW_ROOT/egoblind/videos_archive" "$RAW_ROOT/egotextvqa/videos/indoor_archive"; do
    if [ -f "$f" ] && [ ! -f "$f.extracted" ]; then
      extract_any "$f" "$(dirname "$f")/$( [ "$(basename "$f")" = videos_archive ] && echo videos || echo indoor )" && touch "$f.extracted"
    fi
  done
  find "$RAW_ROOT/egotextvqa/videos" -maxdepth 2 \( -name "*.zip" -o -name "*.tar*" \) 2>/dev/null | while read -r a; do
    [ -f "$a.extracted" ] || { extract_any "$a" "$(dirname "$a")" && touch "$a.extracted"; }
  done
}

step_benchmarks_lmdb() {
  if [ -z "${EGOBLIND_TRAIN_CSV:-}" ]; then  # the EgoBlind train CSV ships inside the Google Drive folder
    mapfile -t csvs < <(find "$RAW_ROOT/egoblind/train" -name "*.csv" 2>/dev/null)
    if [ "${#csvs[@]}" -eq 1 ]; then export EGOBLIND_TRAIN_CSV="${csvs[0]}"; log "EgoBlind train CSV: $EGOBLIND_TRAIN_CSV"
    else log "WARNING: set EGOBLIND_TRAIN_CSV (found ${#csvs[@]} CSVs under $RAW_ROOT/egoblind/train); EgoBlind gets no train split"; fi
  fi
  if [ -f "$LMDB_ROOT/.done" ]; then log "benchmark LMDBs already built (rm $LMDB_ROOT/.done to rebuild)"; return; fi
  rm -rf "${LMDB_ROOT:?}"/*  # LMDB writers refuse duplicate keys: always build into an empty directory
  bash "$HERE/build_all_lmdb.sh"
  touch "$LMDB_ROOT/.done"
}

step_eval_sets() { python -u "$HERE/eval_sets/build_eval_sets.py" --raw_root "$RAW_ROOT/eval_sets" --eval_root "$EVAL_ROOT" --workers "${EVAL_WORKERS:-8}"; }

step_probe_data() {
  hf download timm/imagenet-1k-wds --repo-type dataset --local-dir "$RAW_ROOT/imagenet1k" --include "imagenet1k-train-*.tar"
  mkdir -p "$RAW_ROOT/nyuv2"
  curl -L --fail --retry 5 -C - -o "$RAW_ROOT/nyuv2/nyu_depth_v2_labeled.mat" http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
  curl -L --fail --retry 5 -C - -o "$RAW_ROOT/nyuv2/splits.mat" http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
  [ -f "$EVAL_ROOT/nyuv2/.done" ] || { rm -rf "$EVAL_ROOT/nyuv2"; python "$HERE/eval_sets/make_nyuv2.py" --mat "$RAW_ROOT/nyuv2/nyu_depth_v2_labeled.mat" \
    --splits "$RAW_ROOT/nyuv2/splits.mat" --out "$EVAL_ROOT/nyuv2" && touch "$EVAL_ROOT/nyuv2/.done"; }
}

step_corpus_download() {
  local D="$HERE/download/corpus"
  bash "$D/download_text_rich.sh"
  bash "$D/download_openimages.sh"
  bash "$D/download_cc12m.sh"
  bash "$D/download_imagenet21k.sh"
  export SA1B_LINKS="${SA1B_LINKS:-$RAW_ROOT/sa1b/sa1b_links.txt}"   # link file from the SA-1B download page
  if [ -f "$SA1B_LINKS" ]; then bash "$D/download_sa1b.sh"; else log "WARNING: $SA1B_LINKS missing: SA-1B skipped (docs/setup_cscs.md)"; fi
  bash "$D/download_epic_kitchens.sh"
}

step_blocklist() { bash "$HERE/build_corpus.sh" blocklist; }
step_ego4d_download() { bash "$HERE/download/corpus/download_ego4d.sh"; }
step_corpus_build() { bash "$HERE/build_corpus.sh" text_rich openimages sa1b imagenet21k cc12m ego4d epic_kitchens; }
step_corpus_finalize() { bash "$HERE/build_corpus.sh" finalize; }
step_check() { python "$REPO/scripts/check_data.py"; }

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(models benchmarks_download benchmarks_extract benchmarks_lmdb eval_sets probe_data corpus_download blocklist ego4d_download corpus_build corpus_finalize check)
for s in "${STEPS[@]}"; do
  log "===> $s"
  case "$s" in
    corpus:*) bash "$HERE/build_corpus.sh" "${s#corpus:}" ;;
    *) "step_$s" ;;
  esac
  log "<=== $s done"
done
