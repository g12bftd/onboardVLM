#!/usr/bin/env bash
# Convert raw downloads ($RAW_ROOT) to LMDB ($LMDB_ROOT, the `data_root` used by the configs).
#
# Every benchmark is built on its own into $LMDB_ROOT/<name> and marked finished with <name>/.done:
# finished benchmarks are skipped, an unfinished one is rebuilt from an empty directory. Benchmarks
# whose name starts with a word of SKIP_BENCHMARKS (e.g. "egoblind egotextvqa_outdoor") are not built,
# e.g. while their download is incomplete; build them later by running this script again without them.
#
# A benchmark that fails is removed (no half-built LMDB is ever read), listed in $LMDB_ROOT/.failed, and the others
# are still built. The script fails only if one of REQUIRED_BENCHMARKS failed (default: textvqa cvbench, the
# per-checkpoint downstream benchmarks); the application-check benchmarks never block the data pipeline.
set -euo pipefail
RAW_ROOT="${RAW_ROOT:-./data/raw}"
LMDB_ROOT="${LMDB_ROOT:-${ONBOARD_VLM_DATA:-./data/lmdb}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
L="$HERE/lmdb"
read -r -a SKIP <<< "${SKIP_BENCHMARKS:-}"
read -r -a ONLY <<< "${ONLY_BENCHMARKS:-}"   # build only these (prefixes), e.g. "vizwiz" or "egotextvqa_indoor"
read -r -a REQUIRED <<< "${REQUIRED_BENCHMARKS:-textvqa cvbench}"
BUILT=() SKIPPED=() FAILED=()

build() {  # build <name> <command...>
  local name="$1" s o want=""; shift
  if [ ${#ONLY[@]} -gt 0 ]; then
    for o in "${ONLY[@]}"; do [[ "$name" == "$o"* ]] && want=1; done
    [ -n "$want" ] || return 0
  fi
  for s in "${SKIP[@]}"; do
    if [[ "$name" == "$s"* ]]; then echo "[lmdb] $name: skipped (SKIP_BENCHMARKS)"; SKIPPED+=("$name"); return; fi
  done
  if [ -f "$LMDB_ROOT/$name/.done" ]; then echo "[lmdb] $name: already built"; BUILT+=("$name"); return; fi
  echo "[lmdb] $name: building"
  rm -rf "${LMDB_ROOT:?}/$name"   # LMDB writers refuse duplicate keys: always build into an empty directory
  local rc
  set +e
  ( set -e; "$@" )                # its own shell: one benchmark failing does not stop the others
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[lmdb] $name: FAILED (exit $rc); its partial output is removed"
    rm -rf "${LMDB_ROOT:?}/$name"
    FAILED+=("$name")
    return
  fi
  touch "$LMDB_ROOT/$name/.done"
  BUILT+=("$name")
}

build vizwiz  python "$L/make_vizwiz.py"  --input_path "$RAW_ROOT/vizwiz"  --output_path "$LMDB_ROOT/vizwiz"
build textvqa python "$L/make_textvqa.py" --input_path "$RAW_ROOT/textvqa" --output_path "$LMDB_ROOT/textvqa"
build wearvqa python "$L/make_wearvqa.py" --input_path "$RAW_ROOT/wearvqa" --output_path "$LMDB_ROOT/wearvqa"
build cvbench python "$L/make_cvbench.py" --input_path "$RAW_ROOT/cvbench" --output_path "$LMDB_ROOT/cvbench"

# EgoBlind: frames sampled up to the question time; test.csv (full test set) and train.csv from the GitHub repo.
build egoblind python "$L/make_egoblind.py" --input_path "$RAW_ROOT/egoblind" --video_dir "$RAW_ROOT/egoblind" \
  --output_path "$LMDB_ROOT/egoblind" --num_frames 16 --max_side 1280 ${EGOBLIND_TRAIN_CSV:+--train_csv "$EGOBLIND_TRAIN_CSV"}

# EgoTextVQA: Outdoor at high (1920x1080) and low (960x540) resolution, Indoor as released.
# Outdoor videos from the authors' Hugging Face copy; Indoor videos are EgoSchema clips (README of EgoTextVQA).
OUTDOOR="$RAW_ROOT/egotextvqa/videos/EgoTextVQA-Outdoor"
build egotextvqa_outdoor_high python "$L/make_egotextvqa.py" --subset outdoor --input_path "$RAW_ROOT/egotextvqa" --video_dir "$OUTDOOR" \
  --output_path "$LMDB_ROOT/egotextvqa_outdoor_high" --resize 1920x1080 --num_frames 16
build egotextvqa_outdoor_low python "$L/make_egotextvqa.py" --subset outdoor --input_path "$RAW_ROOT/egotextvqa" --video_dir "$OUTDOOR" \
  --output_path "$LMDB_ROOT/egotextvqa_outdoor_low" --resize 960x540 --num_frames 16
indoor() {  # every Indoor video must be among the EgoSchema clips
  python -I "$HERE/download/verify_videos.py" --name egotextvqa_indoor --video_dir "$RAW_ROOT/egoschema/videos" \
    --ids "$RAW_ROOT/egotextvqa/egotextvqa_indoor_annotation.csv" --key video_id
  python "$L/make_egotextvqa.py" --subset indoor --input_path "$RAW_ROOT/egotextvqa" --video_dir "$RAW_ROOT/egoschema/videos" \
    --output_path "$LMDB_ROOT/egotextvqa_indoor" --num_frames 16 --max_side 1280
}
build egotextvqa_indoor indoor

# EgoSchema: 32 segment-centred frames per 3-minute clip; writes subset.lmdb (500) and test.lmdb (5,031).
build egoschema python "$L/make_egoschema.py" --input_path "$RAW_ROOT/egoschema" --video_dir "$RAW_ROOT/egoschema/videos" \
  --output_path "$LMDB_ROOT/egoschema" --num_frames 32 --max_side 1280

echo "[lmdb] built: ${BUILT[*]:-none} | skipped: ${SKIPPED[*]:-none} | failed: ${FAILED[*]:-none}"
mkdir -p "$LMDB_ROOT"
printf '%s\n' "${FAILED[@]}" | sed '/^$/d' > "$LMDB_ROOT/.failed"   # read by prepare_all.sh: the step stays unfinished
for b in "${FAILED[@]}"; do
  for r in "${REQUIRED[@]}"; do
    [ "$b" = "$r" ] && { echo "[lmdb] required benchmark $b failed (see above)" >&2; exit 1; }
  done
done
[ ${#FAILED[@]} -eq 0 ] || echo "[lmdb] WARNING: ${FAILED[*]} failed (application checks only): the rest of the pipeline goes on; re-run to retry them"
