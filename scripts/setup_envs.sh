#!/usr/bin/env bash
# (Re)create the conda environments `onboardvlm` (training, data) and `onboardvlm-lmms` (lmms-eval) in an
# environments directory of your choice, with exactly the package versions of a lock file, and register that
# directory with conda (envs_dirs), so `conda activate onboardvlm` and every batch job find them by name.
#
#   source scripts/paths.sh
#   bash scripts/setup_envs.sh --envs-dir /capstor/scratch/cscs/$USER/miniforge3/envs [--from <old env prefix or name>]
#
# Steps (each skipped when already done, so the script can be re-run):
#   1. lock   --from: write the exact package list of an existing env to env/lock/ in this repository
#             (conda explicit spec + pip versions; small text files: commit them, they define the environment)
#   2. config prepend --envs-dir to conda's envs_dirs (~/.condarc): environments are found there first, by name
#   3. create onboardvlm from the lock (+ this repository, editable), then onboardvlm-lmms = the same + lmms-eval
#             (with its audio extra: its Gemma 4 model imports librosa), torch / torchvision / transformers / numpy
#             held at the locked versions; an existing lmms env whose bridge import fails is repaired in place
#   4. check  both envs import what they need, with identical torch / transformers, and resolve to --envs-dir
#
# Scratch deletes files not accessed for 30 days (docs.cscs.ch/storage/filesystems): if an environment breaks after a
# pause, delete it and re-run this script; the lock files (in $HOME, not purged) give back the same versions.
set -euo pipefail
: "${ONBOARD_VLM_CODE:?run 'source scripts/paths.sh' first}"
ENVS_DIR="" FROM="" LOCK_DIR="$ONBOARD_VLM_CODE/env/lock" ONLY="onboardvlm onboardvlm-lmms" LMMS_VERSION="0.7.4"
while [ $# -gt 0 ]; do
  case "$1" in
    --envs-dir) ENVS_DIR="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;                 # env prefix or name to lock (default: none, use the existing lock)
    --lock-dir) LOCK_DIR="$2"; shift 2 ;;
    --only) ONLY="${2//,/ }"; shift 2 ;;          # onboardvlm | onboardvlm-lmms
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$ENVS_DIR" ] || { echo "--envs-dir <directory> is required (e.g. /capstor/scratch/cscs/$USER/miniforge3/envs)" >&2; exit 2; }
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[ -n "$CONDA" ] && [ -x "$CONDA" ] || { echo "conda not found: initialise conda in this shell first" >&2; exit 1; }
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ONBOARD_VLM_ROOT/cache/pip}"   # wheels (torch: GBs) off $HOME
ARCH="$(uname -m)"
CONDA_LOCK="$LOCK_DIR/conda-explicit-$ARCH.txt" PIP_LOCK="$LOCK_DIR/pip-$ARCH.txt"
mkdir -p "$ENVS_DIR" "$LOCK_DIR" "$PIP_CACHE_DIR"
ENVS_DIR="$(cd "$ENVS_DIR" && pwd)"

# ---- 1. lock ----------------------------------------------------------------------------------------------------
if [ -n "$FROM" ]; then
  if [ -d "$FROM" ]; then SRC="$(cd "$FROM" && pwd)"; else SRC="$("$CONDA" run -n "$FROM" python -c 'import sys; print(sys.prefix)' | tail -1)"; fi
  [ -x "$SRC/bin/python" ] || { echo "--from $FROM: no environment there" >&2; exit 1; }
  echo "[setup_envs] locking $SRC -> $LOCK_DIR"
  "$CONDA" list -p "$SRC" --explicit > "$CONDA_LOCK"
  "$SRC/bin/python" -m pip list --format=freeze --exclude-editable > "$PIP_LOCK"
  echo "  $(grep -c '^http' "$CONDA_LOCK") conda packages, $(wc -l < "$PIP_LOCK") pip packages; torch / transformers:" \
       "$(grep -iE '^(torch|transformers)==' "$PIP_LOCK" | tr '\n' ' ')"
fi
[ -f "$CONDA_LOCK" ] && [ -f "$PIP_LOCK" ] || { echo "no lock in $LOCK_DIR: pass --from <existing env> once" >&2; exit 1; }
# a torch built for a CUDA version PyPI does not carry (torch==X+cuNNN) comes from PyTorch's own index
CU="$(sed -nE 's/^torch==[^+]*\+cu([0-9]+)$/\1/p' "$PIP_LOCK" | head -1)"
PIP_INDEX=(); [ -n "$CU" ] && PIP_INDEX=(--extra-index-url "https://download.pytorch.org/whl/cu$CU")

# ---- 2. config ----------------------------------------------------------------------------------------------------
if ! "$CONDA" config --show envs_dirs | grep -qF -- "- $ENVS_DIR"; then
  "$CONDA" config --prepend envs_dirs "$ENVS_DIR"
  echo "[setup_envs] conda envs_dirs: $ENVS_DIR first (~/.condarc)"
fi

# ---- 3. create ----------------------------------------------------------------------------------------------------
ready() { [ -x "$ENVS_DIR/$1/bin/python" ] && "$ENVS_DIR/$1/bin/python" -c 'import onboard_vlm' 2>/dev/null; }
create() {  # create <name>: conda packages from the explicit spec, pip packages at the locked versions, this repo
  local p="$ENVS_DIR/$1"
  echo "[setup_envs] creating $1 at $p"
  rm -rf "$p"
  "$CONDA" create -y -p "$p" --file "$CONDA_LOCK"
  "$p/bin/python" -m pip install "${PIP_INDEX[@]}" -r "$PIP_LOCK"
  "$p/bin/python" -m pip install --no-deps -e "$ONBOARD_VLM_CODE"
}
for name in $ONLY; do
  case "$name" in onboardvlm|onboardvlm-lmms) ;; *) echo "unknown environment $name" >&2; exit 2 ;; esac
  if ready "$name"; then echo "[setup_envs] $name exists at $ENVS_DIR/$name"; else create "$name"; fi
  # lmms-eval's Gemma 4 model imports librosa (its `audio` extra); the bridge import is the real test
  if [ "$name" = onboardvlm-lmms ] && ! "$ENVS_DIR/$name/bin/python" -c 'import lmms_eval, onboard_vlm.eval.lmms_eval_model' 2>/dev/null; then
    pins="$(mktemp)"
    grep -iE '^(torch|torchvision|transformers|numpy)==' "$PIP_LOCK" > "$pins" || true
    echo "[setup_envs] lmms-eval[audio]==$LMMS_VERSION into $name, holding: $(tr '\n' ' ' < "$pins")"
    "$ENVS_DIR/$name/bin/python" -m pip install "${PIP_INDEX[@]}" "lmms-eval[audio]==$LMMS_VERSION" -c "$pins"
    rm -f "$pins"
  fi
done

# ---- 4. check -----------------------------------------------------------------------------------------------------
echo "[setup_envs] check"
fail=0
declare -A V
versions() { "$CONDA" run -n "$1" python -c "import sys, torch, transformers, onboard_vlm; print(sys.prefix, torch.__version__, transformers.__version__)" 2>&1 | tail -1; }
for name in $ONLY; do
  if v="$(versions "$name")" && [[ "$v" == /* ]]; then
    V[$name]="${v#* }"
    echo "  $name: $v"
    [[ "$v" == "$ENVS_DIR/$name "* ]] || { echo "  !! $name resolves to ${v%% *}, not $ENVS_DIR/$name (an env of that name comes first in envs_dirs)"; fail=1; }
  else
    echo "  !! $name: cannot import torch / transformers / onboard_vlm: $v"; fail=1
  fi
done
if [[ " $ONLY " == *" onboardvlm-lmms "* ]] && [ -n "${V[onboardvlm-lmms]:-}" ]; then
  # the import the general suite makes: lmms-eval's Gemma 4 model subclassed by our bridge
  if out="$("$CONDA" run -n onboardvlm-lmms python -c "import lmms_eval, onboard_vlm.eval.lmms_eval_model" 2>&1)"; then
    echo "  onboardvlm-lmms: lmms-eval bridge imports"
  else
    echo "  !! onboardvlm-lmms cannot import the lmms-eval bridge: $(echo "$out" | tail -1)"; fail=1
  fi
  if [ -n "${V[onboardvlm]:-}" ] && [ "${V[onboardvlm]}" != "${V[onboardvlm-lmms]}" ]; then
    echo "  !! torch / transformers differ (${V[onboardvlm]} vs ${V[onboardvlm-lmms]}): the general suite would not run the model as trained"; fail=1
  fi
fi
if [ "$fail" -eq 0 ]; then
  echo "[setup_envs] OK. Once you are happy, remove the old environments (e.g. conda env remove -p <old prefix>) and commit $LOCK_DIR."
  [[ " $ONLY " != *" onboardvlm-lmms "* ]] || echo "  (pip's warning that omegaconf / hydra-core need antlr4 4.9 in onboardvlm-lmms is expected: lmms-eval needs" \
    "antlr4 4.13, and that env never loads Hydra configs)"
fi
exit "$fail"
