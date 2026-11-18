#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_textvqa.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# TextVQA v0.5.1 (https://textvqa.org/dataset/): 34,602 train / 5,000 val / 5,734 test questions.
D="$RAW_ROOT/textvqa"; mkdir -p "$D"; cd "$D"
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget -c https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
for s in train val test; do wget -c "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_$s.json"; done
unzip -qn train_val_images.zip   # -> train_images/ (train + val)
unzip -qn test_images.zip        # -> test_images/
echo "TextVQA ready in $D"
