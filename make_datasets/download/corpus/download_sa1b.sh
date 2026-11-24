#!/usr/bin/env bash
# SA-1B: accept the SA-1B Dataset Research License at
#   https://ai.meta.com/datasets/segment-anything-downloads/
# and save the provided link list (tab-separated "file_name<TAB>cdn_link", 1,000 tars of ~11k images)
# as $SA1B_LINKS. We use the first $SA1B_TARS tars (default 100 = ~1.1M images, ~1 TB).
source "$(dirname "$0")/common.sh"
SA1B_LINKS="${SA1B_LINKS:?set SA1B_LINKS to the link file from the SA-1B download page}"
SA1B_TARS="${SA1B_TARS:-100}"
tail -n +2 "$SA1B_LINKS" | sort | head -n "$SA1B_TARS" | while IFS=$'\t' read -r name url; do
  http_get "$url" "$RAW_ROOT/sa1b/$name"
done
