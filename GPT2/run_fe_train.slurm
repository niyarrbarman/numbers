#!/bin/bash
#SBATCH -J gpt2-fe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_fe_%j.log
#SBATCH -e slurm_logs/gpt2_fe_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAIN_DIR="${REPO_DIR}/GPT2/fe"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Path to pretrained NumberEncoder checkpoint
NUM_EMB_CHECKPOINT="/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v8_500k_model.pt"

# Dataset: 'sorting' for synthetic sorting data, 'openwebtext' for OWT
DATASET="${DATASET:-sorting}"

# Data directory: absolute path to the folder containing {train,val}.bin
# Default: GPT2/fe/data/{DATASET}
DATA_DIR="/tmpdir/m24047brmn/numbers/data/"

module load gnu/11.2.0

apptainer exec \
  --nv \
  --cleanenv \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    TRAIN_DIR="${TRAIN_DIR}" \
    NUM_EMB_CHECKPOINT="${NUM_EMB_CHECKPOINT}" \
    DATASET="${DATASET}" \
    DATA_DIR="${DATA_DIR}" \
  bash -c '
    set -euo pipefail
    cd "${TRAIN_DIR}"
    python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
    python3 train.py \
      dataset="${DATASET}" \
      data_dir="${DATA_DIR}" \
      num_emb_checkpoint="${NUM_EMB_CHECKPOINT}"
  '
