#!/usr/bin/env bash
# Build the OVLM-GV general-vision distillation corpus (docs/corpus.md) from the raw downloads.
#
#   step 0  download        make_datasets/download/corpus/download_*.sh          -> $RAW_ROOT
#   step 1  blocklist       fingerprints of every benchmark val/test image/frame -> $CORPUS_ROOT/eval_blocklist.json
#   step 2  build sources   filter + decontaminate + shard to LMDB (parallel)   -> $CORPUS_ROOT/<source>/
#   step 3  finalize        global exact + perceptual de-dup, dataset.json per source, manifest + CORPUS_CARD.md
#   (train / val / test splits and mixture weights are NOT baked in: the loader assigns splits by a
#    seeded content hash and the YAML sets the weights, so ablations never require a rebuild)
#
# Usage:  source scripts/paths.sh && bash make_datasets/build_corpus.sh [step...]
#         (or RAW_ROOT=... LMDB_ROOT=... CORPUS_ROOT=... EVAL_ROOT=... WORKERS=32 bash make_datasets/build_corpus.sh [step...])
#         Finished sources are skipped (<source>/.done); an unfinished one is rebuilt from an empty directory.
#         (steps: blocklist text_rich openimages sa1b imagenet21k cc12m ego4d epic_kitchens finalize; default all)
# The benchmark LMDBs ($LMDB_ROOT, make_datasets/build_all_lmdb.sh) and the evaluation image sets ($EVAL_ROOT,
# make_datasets/eval_sets/build_eval_sets.py) must exist before step 1: the blocklist covers all of them.
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
LMDB_ROOT="${LMDB_ROOT:-${ONBOARD_VLM_DATA:-./data/lmdb}}"
CORPUS_ROOT="${CORPUS_ROOT:-${ONBOARD_VLM_CORPUS:-./data/corpus}}"
EVAL_ROOT="${EVAL_ROOT:-${ONBOARD_VLM_EVAL:-./data/eval_sets}}"   # make_datasets/eval_sets/build_eval_sets.py
WORKERS="${WORKERS:-16}"
C=make_datasets/corpus
BL="$CORPUS_ROOT/eval_blocklist.json"
CAULDRON_SUBSETS="${CAULDRON_SUBSETS:-docvqa infographic_vqa chartqa plotqa dvqa figureqa st_vqa ocrvqa screen2words visualmrc iam rendered_text websight}"
# large, low-diversity synthetic subsets are capped (images per subset, split over workers)
declare -A CAULDRON_CAP=([plotqa]=50000 [dvqa]=50000 [figureqa]=50000 [ocrvqa]=50000)

parallel() {  # parallel <n_workers> <builder.py> <args...>   (one process per worker slice)
  local n="$1"; shift
  local pids=()
  for ((i = 0; i < n; i++)); do
    python -u "$@" --worker_id "$i" --num_workers "$n" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}
img() { parallel "$1" $C/make_image_corpus.py "${@:2}" --blocklist "$BL"; }

step_blocklist() {
  python -u $C/make_eval_blocklist.py \
    --lmdb_dirs "$LMDB_ROOT"/{vizwiz,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema,cvbench} \
    --corpus_dirs "$EVAL_ROOT"/* \
    --egoschema_uid_map "$RAW_ROOT/egoschema/repo/uid_to_ego4d.json" --output "$BL"
}
step_text_rich() {
  img 4 --source parquet --input_path "$RAW_ROOT/pixmo_docs/**/train-*.parquet" --image_column image \
    --name pixmo_docs --output_path "$CORPUS_ROOT/text_rich/pixmo_docs"
  img 8 --source parquet --input_path "$RAW_ROOT/synthdog_en/data/train-*.parquet" --image_column image \
    --name synthdog_en --output_path "$CORPUS_ROOT/text_rich/synthdog_en"
  for s in $CAULDRON_SUBSETS; do
    local cap="${CAULDRON_CAP[$s]:-}"
    img 2 --source parquet --input_path "$RAW_ROOT/the_cauldron/$s/*.parquet" --image_column images \
      --name "cauldron_$s" --output_path "$CORPUS_ROOT/text_rich/cauldron/$s" ${cap:+--limit $((cap / 2))}
  done
}
step_openimages() {  # 16 tars -> at most 16 useful workers; ids of TextVQA val/test images are blocked
  img 16 --source webdataset --input_path "$RAW_ROOT/openimages/train_*.tar.gz" --id_key openimages \
    --name openimages --output_path "$CORPUS_ROOT/openimages"
}
step_sa1b()        { img "$WORKERS" --source webdataset --input_path "$RAW_ROOT/sa1b/sa_*.tar" --name sa1b --output_path "$CORPUS_ROOT/sa1b"; }
step_imagenet21k() { img "$WORKERS" --source webdataset --input_path "$RAW_ROOT/imagenet21k/imagenet22k-train-*.tar" --name imagenet21k --output_path "$CORPUS_ROOT/imagenet21k"; }
step_cc12m()       { img "$WORKERS" --source webdataset --input_path "$RAW_ROOT/cc12m/cc12m-train-*.tar" --name cc12m --output_path "$CORPUS_ROOT/cc12m"; }
step_ego4d() {
  parallel "$WORKERS" $C/make_frame_corpus.py --input_path "$RAW_ROOT/ego4d/v2/video_540ss" --fps 0.5 \
    --name ego4d --output_path "$CORPUS_ROOT/egocentric/ego4d" --blocklist "$BL"
}
step_epic_kitchens() {
  parallel "$WORKERS" $C/make_frame_corpus.py --input_path "$RAW_ROOT/epic_kitchens/EPIC-KITCHENS" --fps 0.5 \
    --name epic_kitchens --output_path "$CORPUS_ROOT/egocentric/epic_kitchens" --blocklist "$BL"
}
step_finalize() { python -u $C/finalize_corpus.py --corpus_root "$CORPUS_ROOT" --sources_yaml $C/sources.yaml; }

out_dir() {  # processed output of a source step ("" = not a source)
  case "$1" in
    text_rich) echo "$CORPUS_ROOT/text_rich" ;;
    ego4d|epic_kitchens) echo "$CORPUS_ROOT/egocentric/$1" ;;
    blocklist|finalize) echo "" ;;
    *) echo "$CORPUS_ROOT/$1" ;;
  esac
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(blocklist text_rich openimages sa1b imagenet21k cc12m ego4d epic_kitchens finalize)
for s in "${STEPS[@]}"; do
  echo "===> corpus step: $s ($(date))"
  out="$(out_dir "$s")"
  if [ "$s" = blocklist ] && [ -f "$BL" ] && [ -z "${FORCE_BLOCKLIST:-}" ]; then
    echo "     $BL exists (FORCE_BLOCKLIST=1 to rebuild it; then rebuild every source)"; continue
  fi
  if [ -n "$out" ] && [ -f "$out/.done" ]; then
    echo "     $out already built (delete it to rebuild)"; continue
  fi
  [ -n "$out" ] && rm -rf "$out"  # a source is always built into an empty directory (no stale shards)
  "step_$s"
  [ -n "$out" ] && touch "$out/.done"
  echo "<=== corpus step: $s done ($(date))"
done
