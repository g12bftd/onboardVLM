#!/usr/bin/env bash
# Usage: RAW_ROOT=/path/to/raw bash download_cvbench.sh
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
# CV-Bench (Cambrian-1, https://huggingface.co/datasets/nyu-visionx/CV-Bench): 2,638 MCQs (2D: ADE20K+COCO, 3D: Omni3D).
D="$RAW_ROOT/cvbench"; mkdir -p "$D"
hf download nyu-visionx/CV-Bench --repo-type dataset --local-dir "$D"
echo "CV-Bench ready in $D"
