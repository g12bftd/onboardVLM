#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_egotextvqa.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# EgoTextVQA (CVPR 2025, https://github.com/zhousheng97/EgoTextVQA): Outdoor 694 videos / 4,848 QA,
# Indoor 813 Ego4D videos / 2,216 QA. Test-only.
D="$RAW_ROOT/egotextvqa"; mkdir -p "$D"
[ -d "$D/repo" ] || git clone --depth 1 https://github.com/zhousheng97/EgoTextVQA "$D/repo"
cp -n "$D/repo/data/egotextvqa_outdoor_annotation.csv" "$D/repo/data/egotextvqa_indoor_annotation.csv" "$D/"
# Videos: the authors' Hugging Face copy (README of the repo; EgoTextVQA-Indoor/, EgoTextVQA-Outdoor/).
# Google Drive is not used: it rate-limits bulk downloads, and a second copy of a video under another
# folder would make the builders' lookup by file name ambiguous.
if [ -d "$D/videos/outdoor" ] || compgen -G "$D/videos/indoor_archive*" >/dev/null; then
  echo "remove the earlier Google Drive copy first: rm -rf $D/videos/outdoor $D/videos/indoor_archive*" >&2; exit 1
fi
for attempt in 1 2 3 4 5; do  # hf download resumes and skips finished files; retry transient errors / rate limits
  hf download ShengZhou97/EgoTextVQA --repo-type dataset --local-dir "$D/videos" && break
  [ "$attempt" -lt 5 ] || exit 1
  echo "hf download failed (attempt $attempt); retrying in $((attempt * 120)) s" >&2; sleep $((attempt * 120))
done
for s in outdoor indoor; do
  python -I "$HERE/verify_videos.py" --name "egotextvqa_$s" --video_dir "$D/videos" --ids "$D/egotextvqa_${s}_annotation.csv" --key video_id
done
echo "EgoTextVQA ready in $D"
