#!/usr/bin/env bash
# Ego4D v2, 540p-downscaled videos (video_540ss) for a scenario-stratified subset of ~$EGO4D_HOURS hours.
# 1) Sign the Ego4D license at https://ego4d-data.org/docs/start-here/ ; you receive AWS credentials:
#    `aws configure --profile ego4d`.   2) pip install ego4d
# 3) Build the benchmark blocklist first (make_datasets/build_corpus.sh step 1), so that every video
#    behind EgoSchema / EgoTextVQA-Indoor is excluded from the selection.
source "$(dirname "$0")/common.sh"
CORPUS_ROOT="${CORPUS_ROOT:-./data/corpus}"
EGO4D_HOURS="${EGO4D_HOURS:-1000}"
OUT="$RAW_ROOT/ego4d"
# metadata only (ego4d.json); the annotations dataset is the smallest one that brings it along
ego4d --output_directory "$OUT" --datasets annotations --metadata --version v2 --aws_profile_name ego4d -y
python make_datasets/download/corpus/select_ego4d_videos.py --metadata "$OUT/ego4d.json" \
  --blocklist "$CORPUS_ROOT/eval_blocklist.json" --hours "$EGO4D_HOURS" --output "$OUT/corpus_video_uids.txt"
ego4d --output_directory "$OUT" --datasets video_540ss --video_uid_file "$OUT/corpus_video_uids.txt" \
  --version v2 --aws_profile_name ego4d -y
