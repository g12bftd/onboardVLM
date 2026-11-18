#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_egotextvqa.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# EgoTextVQA (CVPR 2025, https://github.com/zhousheng97/EgoTextVQA): Outdoor 694 videos / 4,848 QA,
# Indoor 813 Ego4D videos / 2,216 QA. Test-only.
D="$RAW_ROOT/egotextvqa"; mkdir -p "$D"
[ -d "$D/repo" ] || git clone --depth 1 https://github.com/zhousheng97/EgoTextVQA "$D/repo"
cp -n "$D/repo/data/egotextvqa_outdoor_annotation.csv" "$D/repo/data/egotextvqa_indoor_annotation.csv" "$D/"
# Videos (README of the repo): Hugging Face mirror ...
hf download ShengZhou97/EgoTextVQA --repo-type dataset --local-dir "$D/videos" || {
  # ... or Google Drive: outdoor folder + indoor archive (indoor videos are EgoSchema/Ego4D clips)
  gdown --folder https://drive.google.com/drive/folders/1uGE-dObMFpmuhewHi9K_I8pdCbV8efm1 -O "$D/videos/outdoor"
  gdown 12TZyzoJVZ4GxPKMTZWZzIFnwetECPpI2 -O "$D/videos/indoor_archive"
}
echo "EgoTextVQA ready in $D (extract any archives under $D/videos)"
