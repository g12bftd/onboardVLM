#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_egoschema.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# EgoSchema (https://github.com/egoschema/EgoSchema): 5,031 3-minute clips, 500 public answers.
D="$RAW_ROOT/egoschema"; mkdir -p "$D/videos"
[ -d "$D/repo" ] || git clone --depth 1 https://github.com/egoschema/EgoSchema "$D/repo"
cp -n "$D/repo/questions.json" "$D/repo/subset_answers.json" "$D/"
# Videos: Kaggle, option A of the EgoSchema README. Needs an API token (~/.kaggle/access_token) AND the
# competition rules accepted at https://www.kaggle.com/competitions/egoschema-public (else HTTP 403).
# Every command on its own line: a failed download must fail this job (set -e).
if [ ! -f "$D/videos/egoschema-public.zip.extracted" ]; then
  kaggle competitions download egoschema-public -p "$D/videos"
  (cd "$D/videos" && unzip -qn egoschema-public.zip)
  touch "$D/videos/egoschema-public.zip.extracted"
fi
# complete? every clip of the 500-question subset (scored), and the full test set
python -I "$HERE/verify_videos.py" --name egoschema --video_dir "$D/videos" --ids "$D/questions.json" --key q_uid \
  --required "$D/subset_answers.json" --max_missing 0.01
echo "EgoSchema ready in $D"
