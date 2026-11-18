#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_vizwiz.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# VizWiz-VQA (https://vizwiz.org/tasks-and-datasets/vqa/): 20,523 train / 4,319 val / 8,000 test images.
D="$RAW_ROOT/vizwiz"; mkdir -p "$D"; cd "$D"
for f in train val test; do wget -c "https://vizwiz.cs.colorado.edu/VizWiz_final/images/$f.zip"; done
wget -c https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip
for f in train val test; do unzip -qn "$f.zip"; done
unzip -qn Annotations.zip -d Annotations
echo "VizWiz ready in $D (images: train/ val/ test/, annotations: Annotations/)"
