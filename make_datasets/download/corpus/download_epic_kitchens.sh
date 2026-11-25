#!/usr/bin/env bash
# EPIC-KITCHENS-100 videos (700 videos, 100 h, 45 kitchens; CC BY-NC 4.0) via the official downloader.
source "$(dirname "$0")/common.sh"
DL="$RAW_ROOT/epic_kitchens/epic-kitchens-download-scripts"
[ -d "$DL" ] || git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts "$DL"
# the downloader reads its split lists from data/ relative to the working directory: run it from its clone
OUT="$(cd "$RAW_ROOT" && pwd)/epic_kitchens"
(cd "$DL" && python epic_downloader.py --videos --output-path "$OUT")
