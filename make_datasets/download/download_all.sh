#!/usr/bin/env bash
# Download every benchmark into $RAW_ROOT (default ./data/raw). Requires: wget, unzip, git,
# `pip install -U "huggingface_hub[cli]" gdown kaggle`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for d in vizwiz textvqa wearvqa egoblind egotextvqa egoschema cvbench; do
  echo "=== $d ==="; bash "$HERE/download_$d.sh"
done
