#!/usr/bin/env bash
# Conceptual 12M as a webdataset snapshot (2,176 tars, 10,968,539 image-text pairs, ~1.18 TB).
# The images were fetched once in 2023 (pixparse); no URL scraping and no link rot.
source "$(dirname "$0")/common.sh"
hf_get pixparse/cc12m-wds "$RAW_ROOT/cc12m" "cc12m-train-*.tar" "_info.json" "LICENSE"
