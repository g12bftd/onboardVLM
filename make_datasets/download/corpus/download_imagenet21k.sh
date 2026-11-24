#!/usr/bin/env bash
# OPTIONAL (objects_source=imagenet21k): ImageNet-21k (fall11, 13,673,551 train images, 4,096 tars, ~1.19 TB).
# The default corpus uses ImageNet-1k instead (download_imagenet1k.sh).
# Gated: accept the ImageNet terms on https://huggingface.co/datasets/timm/imagenet-22k-wds first,
# then `hf auth login` (or export HF_TOKEN).
source "$(dirname "$0")/common.sh"
hf_get timm/imagenet-22k-wds "$RAW_ROOT/imagenet21k" "imagenet22k-train-*.tar" "_info.json" "LICENSE"
