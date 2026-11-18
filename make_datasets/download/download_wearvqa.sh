#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_wearvqa.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# WearVQA (NeurIPS 2025 D&B, arXiv 2511.22154): ~2.5K image-question-answer triplets, test only.
# Dataset URL from the paper's OpenReview page. If the repo is gated: `hf auth login` and accept the terms first.
D="$RAW_ROOT/wearvqa"; mkdir -p "$D"
hf download tonyliao-meta/WearVQA --repo-type dataset --local-dir "$D"
echo "WearVQA ready in $D -- inspect columns with: python make_datasets/lmdb/make_wearvqa.py --input_path $D --output_path /tmp/x --dry_run"
