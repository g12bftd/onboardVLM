#!/bin/bash
# CPU-heavy data preparation as a batch job:   sbatch --account=<project> scripts/slurm/prepare_data.sbatch <step> [step ...]
# e.g.  sbatch --account=<project> scripts/slurm/prepare_data.sbatch benchmarks_lmdb eval_sets
#       sbatch --account=<project> --time=24:00:00 scripts/slurm/prepare_data.sbatch corpus:cc12m
# (steps: make_datasets/prepare_all.sh; finished steps are skipped, so a timed-out job can simply be resubmitted)
#SBATCH --job-name=onboardvlm-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT="${ONBOARD_VLM_ROOT:-/capstor/scratch/cscs/${USER}/onboardvlm}"
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate onboardvlm
cd "$ROOT/code"
source scripts/paths.sh
export WORKERS="${WORKERS:-128}"
srun --cpu-bind=none bash make_datasets/prepare_all.sh "$@"
