#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_egoschema.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# EgoSchema (https://github.com/egoschema/EgoSchema): 5,031 3-minute clips, 500 public answers.
D="$RAW_ROOT/egoschema"; mkdir -p "$D"
[ -d "$D/repo" ] || git clone --depth 1 https://github.com/egoschema/EgoSchema "$D/repo"
cp -n "$D/repo/questions.json" "$D/repo/subset_answers.json" "$D/"
# Videos: Kaggle (accept the rules at https://www.kaggle.com/competitions/egoschema-public first; needs an API token in
# ~/.kaggle/access_token, or a legacy ~/.kaggle/kaggle.json)
kaggle competitions download -c egoschema-public -p "$D/videos" && (cd "$D/videos" && unzip -qn egoschema-public.zip)
# Alternatives: the Google Drive folder in the EgoSchema README, or the lmms-lab/egoschema HF mirror:
#   hf download lmms-lab/egoschema --repo-type dataset --local-dir "$D/videos"
echo "EgoSchema ready in $D"
