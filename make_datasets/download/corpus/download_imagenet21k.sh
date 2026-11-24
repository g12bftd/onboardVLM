#!/usr/bin/env bash
# ImageNet-21k (fall11 release, 13,673,551 train images, 4,096 tars, ~1.24 TB).
# Gated: accept the ImageNet terms on https://huggingface.co/datasets/timm/imagenet-22k-wds first,
# then `hf auth login` (or export HF_TOKEN).
source "$(dirname "$0")/common.sh"
hf_get timm/imagenet-22k-wds "$RAW_ROOT/imagenet21k" "imagenet22k-train-*.tar" "_info.json" "LICENSE"
