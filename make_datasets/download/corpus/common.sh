# Sourced by the corpus download scripts.
set -euo pipefail
export RAW_ROOT="${RAW_ROOT:-./data/raw}"
mkdir -p "$RAW_ROOT"
# Hugging Face downloads (resumable; set HF_TOKEN or run `hf auth login` for gated sets).
hf_get() {  # hf_get <repo_id> <local_dir> <include_pattern>...   (each pattern becomes one --include)
  local repo="$1" dir="$2"; shift 2
  local args=(); for p in "$@"; do args+=(--include "$p"); done
  hf download "$repo" --repo-type dataset --local-dir "$dir" "${args[@]}"
}
# Plain HTTPS downloads, resumable.
http_get() {  # http_get <url> <out_file>
  mkdir -p "$(dirname "$2")"
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$2" "$1"
}
