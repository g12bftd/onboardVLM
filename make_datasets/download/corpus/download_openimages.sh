#!/usr/bin/env bash
# Open Images V7: the 1,743,042 train images that carry box annotations (multi-object scenes),
# hosted by the CVDF on a public S3 bucket (16 tars, ~550 GB). No account needed.
source "$(dirname "$0")/common.sh"
for s in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  http_get "https://open-images-dataset.s3.amazonaws.com/tar/train_${s}.tar.gz" "$RAW_ROOT/openimages/train_${s}.tar.gz"
done
