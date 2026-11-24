#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_egoblind.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# EgoBlind (https://github.com/doc-doc/EgoBlind): 1,392 videos / 5.3K questions by blind users.
# Annotations live in the GitHub repo; train data + videos on Google Drive (needs `pip install gdown`).
D="$RAW_ROOT/egoblind"; mkdir -p "$D"
[ -d "$D/repo" ] || git clone --depth 1 https://github.com/doc-doc/EgoBlind "$D/repo"
cp -n "$D/repo/test_half_release.csv" "$D/"                      # publicly released half of the test set
gdown -c --folder https://drive.google.com/drive/folders/1zW8waLCS_8Ay8kOWpCsvF77IeIG9TzOq -O "$D/train"
gdown -c --folder https://drive.google.com/drive/folders/1gLcqwKrJcZ7tTbaBI8aWEhPImRdDsGQx -O "$D/test"
gdown -c --fuzzy https://drive.google.com/file/d/1Wob42tJ95DL1VvKad7BPc5QIlHtiuttI/view -O "$D/videos_archive"
echo "Extract the downloaded video archive(s) under $D/videos and locate the train CSV under $D/train."
