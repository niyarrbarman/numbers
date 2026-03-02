#!/bin/bash
#SBATCH -J np-emb-v9
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm_logs/np_emb_v9_%j.log
#SBATCH -e slurm_logs/np_emb_v9_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="${REPO_DIR}/np_emb_v9.py"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

MAX_STEPS="${MAX_STEPS:-500000}"
SCALE_DIMS="${SCALE_DIMS:-16}"
RESIDUE_PERIODS="${RESIDUE_PERIODS:-10,100,1000,10000,100000}"

module load gnu/11.2.0

apptainer exec \
  --nv \
  --cleanenv \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    REPO_DIR="${REPO_DIR}" \
    SCRIPT_PATH="${SCRIPT_PATH}" \
    MAX_STEPS="${MAX_STEPS}" \
    SCALE_DIMS="${SCALE_DIMS}" \
    RESIDUE_PERIODS="${RESIDUE_PERIODS}" \
  bash -c '
    set -euo pipefail
    cd "${REPO_DIR}"
    python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
    python3 "${SCRIPT_PATH}" \
      --max-steps ${MAX_STEPS} \
      --scale-dims ${SCALE_DIMS} \
      --residue-periods ${RESIDUE_PERIODS}
  '
