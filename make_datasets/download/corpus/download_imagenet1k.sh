#!/usr/bin/env bash
# ImageNet-1k (ILSVRC-2012) train: 1,281,167 images in 1,024 webdataset tars (~154 GB).
# One download serves two purposes: the corpus "objects" component (default objects_source) and the
# E2 ImageNet probes (run_probe.py reads $RAW_ROOT/imagenet1k/imagenet1k-train-*.tar). The validation
# split is downloaded separately as an evaluation set (make_datasets/eval_sets) and is blocklisted.
# Gated: accept the terms on https://huggingface.co/datasets/timm/imagenet-1k-wds, then `hf auth login`.
source "$(dirname "$0")/common.sh"
hf_get timm/imagenet-1k-wds "$RAW_ROOT/imagenet1k" "imagenet1k-train-*.tar" "_info.json" "README.md"
