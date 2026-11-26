#!/usr/bin/env bash
# Convert raw downloads ($RAW_ROOT) to LMDB ($LMDB_ROOT, the `data_root` used by the configs).
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
LMDB_ROOT="${LMDB_ROOT:-${ONBOARD_VLM_DATA:-./data/lmdb}}"
L=make_datasets/lmdb

python $L/make_vizwiz.py  --input_path "$RAW_ROOT/vizwiz"  --output_path "$LMDB_ROOT/vizwiz"
python $L/make_textvqa.py --input_path "$RAW_ROOT/textvqa" --output_path "$LMDB_ROOT/textvqa"
python $L/make_wearvqa.py --input_path "$RAW_ROOT/wearvqa" --output_path "$LMDB_ROOT/wearvqa"
python $L/make_cvbench.py --input_path "$RAW_ROOT/cvbench" --output_path "$LMDB_ROOT/cvbench"

# EgoBlind: frames sampled up to the question time; test.csv (full test set) and train.csv from the GitHub repo.
python $L/make_egoblind.py --input_path "$RAW_ROOT/egoblind" --video_dir "$RAW_ROOT/egoblind" \
  --output_path "$LMDB_ROOT/egoblind" --num_frames 16 --max_side 1280 \
  ${EGOBLIND_TRAIN_CSV:+--train_csv "$EGOBLIND_TRAIN_CSV"}

# EgoTextVQA: Outdoor at high (1920x1080) and low (960x540) resolution, Indoor as released.
# Outdoor videos from the authors' Hugging Face copy; Indoor videos are EgoSchema clips (README of EgoTextVQA).
python $L/make_egotextvqa.py --subset outdoor --input_path "$RAW_ROOT/egotextvqa" --video_dir "$RAW_ROOT/egotextvqa/videos/EgoTextVQA-Outdoor" \
  --output_path "$LMDB_ROOT/egotextvqa_outdoor_high" --resize 1920x1080 --num_frames 16
python $L/make_egotextvqa.py --subset outdoor --input_path "$RAW_ROOT/egotextvqa" --video_dir "$RAW_ROOT/egotextvqa/videos/EgoTextVQA-Outdoor" \
  --output_path "$LMDB_ROOT/egotextvqa_outdoor_low" --resize 960x540 --num_frames 16
python $L/make_egotextvqa.py --subset indoor --input_path "$RAW_ROOT/egotextvqa" --video_dir "$RAW_ROOT/egoschema/videos" \
  --output_path "$LMDB_ROOT/egotextvqa_indoor" --num_frames 16 --max_side 1280

# EgoSchema: 32 segment-centred frames per 3-minute clip; writes subset.lmdb (500) and test.lmdb (5,031).
python $L/make_egoschema.py --input_path "$RAW_ROOT/egoschema" --video_dir "$RAW_ROOT/egoschema/videos" \
  --output_path "$LMDB_ROOT/egoschema" --num_frames 32 --max_side 1280
