#!/bin/bash
# Run one entry point on one GH200 node (4 GPUs), training or evaluation:
#   sbatch --account=<project> scripts/slurm/run.sbatch run_distill.py +experiment=embedding/gemma4_vit_l8_d384_e2b seed=0
#   sbatch --account=<project> --time=06:00:00 scripts/slurm/run.sbatch run_fidelity.py run_name=<name> student_dir=<export>
# run_distill.py runs one process per GPU (Lightning DDP over the node's 4 GPUs; one job per seed, no -m).
# Every other entry point runs one process on one GPU (NTASKS=<n> to change). Submit from a shell where the
# onboardvlm conda env is active (its installation is used here, as in prepare_data.sbatch).
# Logs: $SLURM_LOG_DIR/<job-name>-<id>.out (pass --job-name=<name> to tell runs apart).
#SBATCH --job-name=ovlm-run
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
set -euo pipefail
ROOT="${ONBOARD_VLM_ROOT:-/capstor/scratch/cscs/${USER}/onboardvlm}"
CODE="${ONBOARD_VLM_CODE:-$HOME/repos/onboardVLM}"
if [ -z "${CONDA_BASE:-}" ] && [ -n "${CONDA_EXE:-}" ]; then CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"; fi
CONDA_BASE="${CONDA_BASE:-$ROOT/miniforge3}"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "no conda at $CONDA_BASE: export CONDA_BASE=\$(conda info --base) before submitting" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ONBOARD_VLM_ENV:-onboardvlm}"
cd "$CODE"
source scripts/paths.sh
ENTRY="$1"; shift
case "$ENTRY" in run_distill.py) NT="${NTASKS:-4}" ;; *) NT="${NTASKS:-1}" ;; esac
CPUS=$(( ${SLURM_CPUS_ON_NODE:-288} / NT ))
export OMP_NUM_THREADS=$(( CPUS < 16 ? CPUS : 16 ))
echo "job ${SLURM_JOB_ID:-?} on $(hostname): $ENTRY $* | $NT process(es), $CPUS CPUs each | commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
srun --ntasks="$NT" --ntasks-per-node="$NT" --cpus-per-task="$CPUS" --cpu-bind=none python -u "$ENTRY" "$@"
