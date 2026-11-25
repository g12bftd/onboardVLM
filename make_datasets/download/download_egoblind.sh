#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_egoblind.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# EgoBlind (NeurIPS 2025 D&B, https://github.com/doc-doc/EgoBlind): 1,392 videos by blind users.
# Annotations from the GitHub repo: test.csv (full test set, 2,565 Q, answers released), test_half_release.csv
# (the half released first), finetune/train.csv (2,746 Q). Videos: the authors' Google Drive folders (Train,
# Test), zipped in parts; bench-extract unpacks them into $D/videos. Needs `pip install gdown`.
D="$RAW_ROOT/egoblind"; mkdir -p "$D"
if [ -d "$D/repo/.git" ]; then git -C "$D/repo" pull -q --ff-only || true; else git clone --depth 1 https://github.com/doc-doc/EgoBlind "$D/repo"; fi
cp -f "$D/repo/test.csv" "$D/repo/test_half_release.csv" "$D/"
cp -f "$D/repo/finetune/train.csv" "$D/train.csv"
# Google Drive limits downloads per file ("Too many users have viewed or downloaded this file recently"):
# if that happens, re-submit later (finished files are skipped, -c), or put the cookies of a browser
# session that can open the folders in ~/.cache/gdown/cookies.txt (gdown FAQ).
status=0
gdown -c --folder https://drive.google.com/drive/folders/1zW8waLCS_8Ay8kOWpCsvF77IeIG9TzOq -O "$D/train" || status=1
gdown -c --folder https://drive.google.com/drive/folders/1gLcqwKrJcZ7tTbaBI8aWEhPImRdDsGQx -O "$D/test" || status=1
# the "video source" file of the README (sources to cite when redistributing); not needed to build the benchmark
gdown -c --fuzzy https://drive.google.com/file/d/1Wob42tJ95DL1VvKad7BPc5QIlHtiuttI/view -O "$D/video_sources" || true
if [ "$status" -ne 0 ]; then
  echo "EgoBlind: some Google Drive files did not download (see above). Re-submit later; finished files are kept." >&2
  exit 1
fi
echo "EgoBlind ready in $D (bench-extract unpacks the video zips into $D/videos)"
