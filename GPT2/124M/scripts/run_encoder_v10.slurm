#!/bin/bash
# Train NumberEncoder v10 (high-fidelity, 1B range).
# 1 node, 1 GPU, batch=1024, 2M steps.
#
# Usage:
#   sbatch 124M/scripts/run_encoder_v10.slurm
#   MAX_STEPS=5000000 sbatch 124M/scripts/run_encoder_v10.slurm

#SBATCH -J np-emb-v10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=12:00:00
#SBATCH -o slurm_logs/np_emb_v10_%j.log
#SBATCH -e slurm_logs/np_emb_v10_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

MAX_STEPS="${MAX_STEPS:-2000000}"
SEED="${SEED:-42}"

module load gnu/11.2.0

echo "=========================================="
echo "NumberEncoder v10 Training (1B range)"
echo "  Max steps:   ${MAX_STEPS}"
echo "  Seed:        ${SEED}"
echo "  Node:        $(hostname)"
echo "  GPUs:        1"
if command -v nvidia-smi &>/dev/null; then
  echo "  GPU type:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
fi
echo "=========================================="

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${REPO_DIR}/../np_emb_v10.py" \
    --max-steps "${MAX_STEPS}" \
    --seed "${SEED}"

status=$?
echo "=========================================="
echo "Training finished with status $status"
echo "=========================================="
exit $status
