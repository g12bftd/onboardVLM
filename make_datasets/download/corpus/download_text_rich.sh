#!/usr/bin/env bash
# Text-rich images: PixMo-Docs (255,261; ODC-BY), SynthDoG-en (500,000; MIT) and the document / chart /
# scene-text / UI subsets of The Cauldron (train splits; per-subset licences in docs/corpus.md).
source "$(dirname "$0")/common.sh"
hf_get allenai/pixmo-docs "$RAW_ROOT/pixmo_docs" "*/train-*.parquet" "README.md"
hf_get naver-clova-ix/synthdog-en "$RAW_ROOT/synthdog_en" "data/train-*.parquet" "README.md"
CAULDRON_SUBSETS="${CAULDRON_SUBSETS:-docvqa infographic_vqa chartqa plotqa dvqa figureqa st_vqa ocrvqa screen2words visualmrc iam rendered_text websight}"
inc=(); for s in $CAULDRON_SUBSETS; do inc+=("$s/*"); done
hf_get HuggingFaceM4/the_cauldron "$RAW_ROOT/the_cauldron" "${inc[@]}" "README.md"
