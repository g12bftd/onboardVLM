#!/usr/bin/env bash
# Build the OVLM-GV general-vision distillation corpus (docs/corpus.md) from the raw downloads.
#
#   step 0  download        make_datasets/download/corpus/download_*.sh          -> $RAW_ROOT
#   step 1  blocklist       fingerprints of every benchmark val/test image/frame -> $CORPUS_ROOT/eval_blocklist.json
#   step 2  build sources   filter + decontaminate + shard to LMDB (parallel)   -> $CORPUS_ROOT/<source>/
#   step 3  finalize        global exact + perceptual de-dup, held-out split, manifest + CORPUS_CARD.md
#
# Usage:  RAW_ROOT=/data/raw LMDB_ROOT=/data/lmdb CORPUS_ROOT=/data/corpus WORKERS=32 bash make_datasets/build_corpus.sh [step...]
#         (steps: blocklist text_rich openimages sa1b imagenet21k cc12m ego4d epic_kitchens finalize; default all)
# The benchmark LMDBs ($LMDB_ROOT, make_datasets/build_all_lmdb.sh) must exist before step 1.
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
LMDB_ROOT="${LMDB_ROOT:-${ONBOARD_VLM_DATA:-./data/lmdb}}"
CORPUS_ROOT="${CORPUS_ROOT:-${ONBOARD_VLM_CORPUS:-./data/corpus}}"
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
step_finalize() { python -u $C/finalize_corpus.py --corpus_root "$CORPUS_ROOT" --sources_yaml $C/sources.yaml --holdout 2000; }

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(blocklist text_rich openimages sa1b imagenet21k cc12m ego4d epic_kitchens finalize)
for s in "${STEPS[@]}"; do
  echo "===> corpus step: $s ($(date))"
  "step_$s"
done
