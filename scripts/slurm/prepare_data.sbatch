#!/bin/bash
# One or more make_datasets/prepare_all.sh steps as a batch job (downloads and processing alike):
#   sbatch --account=<project> scripts/slurm/prepare_data.sbatch <step> [step ...]
#   sbatch --account=<project> --time=24:00:00 scripts/slurm/prepare_data.sbatch download:cc12m
# Every step is idempotent (finished outputs skipped, downloads resume), so a job that timed out can
# simply be resubmitted. scripts/slurm/submit_data_pipeline.sh submits the whole pipeline.
#SBATCH --job-name=onboardvlm-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
set -euo pipefail
ROOT="${ONBOARD_VLM_ROOT:-/capstor/scratch/cscs/${USER}/onboardvlm}"
CODE="${ONBOARD_VLM_CODE:-$HOME/repos/onboardVLM}"   # set by scripts/paths.sh / the submit script
# conda: CONDA_BASE as exported by the submit script, else the installation of the submitting shell
# (sbatch passes its environment on, including CONDA_EXE), else $ROOT/miniforge3. Env: $ONBOARD_VLM_ENV.
if [ -z "${CONDA_BASE:-}" ] && [ -n "${CONDA_EXE:-}" ]; then CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"; fi
CONDA_BASE="${CONDA_BASE:-$ROOT/miniforge3}"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "no conda at $CONDA_BASE: export CONDA_BASE=\$(conda info --base) before submitting" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ONBOARD_VLM_ENV:-onboardvlm}"
cd "$CODE"
source scripts/paths.sh
export WORKERS="${WORKERS:-128}"   # builder processes (GH200 node: 4 x 72 Grace cores)
echo "job $SLURM_JOB_ID on $(hostname): $*"
srun --cpu-bind=none bash make_datasets/prepare_all.sh "$@"
