#!/bin/bash
# Train base GPT-2 on 5-digit dataset (plain text numbers, no number embeddings).
#
# Usage:
#   sbatch run_base_train_5dig.slurm
#   MAX_ITERS=25000 sbatch run_base_train_5dig.slurm

#SBATCH -J gpt2-base-5dig
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_base_5dig_%j.log
#SBATCH -e slurm_logs/gpt2_base_5dig_%j.log

mkdir -p slurm_logs
set -euo pipefail

TRAIN_DIR="/work/m24047/m24047brmn/numbers/GPT2/base"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Dataset
DATASET="${DATASET:-numtasks_base_5dig}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_base_5dig}"

# Training hyperparameters (matching the FE-SME 5-digit run)
MAX_ITERS="${MAX_ITERS:-20000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"

# Init mode
INIT_FROM="${INIT_FROM:-scratch}"

# Output directory for checkpoints
OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/model_checkpoints/base_5dig}"

GPUS_PER_NODE=2

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

module load gnu/11.2.0

echo "=========================================="
echo "GPT-2 BASE Training — 5-digit (plain text)"
echo "  Nodes:          ${SLURM_NNODES}"
echo "  GPUs/node:      ${GPUS_PER_NODE}"
echo "  Total GPUs:     $((SLURM_NNODES * GPUS_PER_NODE))"
echo "  Dataset:        ${DATASET}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Max iters:      ${MAX_ITERS}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Init from:      ${INIT_FROM}"
echo "  Out dir:        ${OUT_DIR}"
echo "  Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================================="

srun apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --env "MASTER_ADDR=${MASTER_ADDR}" \
  --env "MASTER_PORT=${MASTER_PORT}" \
  --env "SLURM_NNODES=${SLURM_NNODES}" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    ${TRAIN_DIR}/train.py \
      dataset="${DATASET}" \
      data_dir="${DATA_DIR}" \
      out_dir="${OUT_DIR}" \
      learning_rate="${LEARNING_RATE}" \
      max_iters="${MAX_ITERS}" \
      lr_decay_iters="${MAX_ITERS}" \
      init_from="${INIT_FROM}"

status=$?
echo "=========================================="
echo "Training finished with status $status"
echo "=========================================="
exit $status
