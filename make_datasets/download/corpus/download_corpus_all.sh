#!/usr/bin/env bash
# Download every source of the OVLM-GV corpus (see docs/corpus.md for sizes and licences).
# Gated sources need prior steps: ImageNet-21k (HF terms), SA-1B (link file), Ego4D (license + AWS creds).
set -euo pipefail
D="$(dirname "$0")"
bash "$D/download_text_rich.sh"
bash "$D/download_openimages.sh"
bash "$D/download_cc12m.sh"
bash "$D/download_imagenet21k.sh"
bash "$D/download_sa1b.sh"
bash "$D/download_epic_kitchens.sh"
bash "$D/download_ego4d.sh"     # needs $CORPUS_ROOT/eval_blocklist.json (build_corpus.sh step 1)
