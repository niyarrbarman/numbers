#!/bin/bash
#SBATCH -J gpt2-base
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_base_%j.log
#SBATCH -e slurm_logs/gpt2_base_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAIN_DIR="${REPO_DIR}/GPT2/base"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

module load gnu/11.2.0

apptainer exec \
  --nv \
  --cleanenv \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    TRAIN_DIR="${TRAIN_DIR}" \
  bash -c '
    set -euo pipefail
    cd "${TRAIN_DIR}"
    python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
    python3 train.py
  '
