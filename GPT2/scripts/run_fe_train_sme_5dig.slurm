#!/bin/bash
# Train GPT-2 SME on 5-digit dataset (single phase, no curriculum).
#
# Usage:
#   sbatch run_fe_train_sme_5dig.slurm
#   MAX_ITERS=25000 sbatch run_fe_train_sme_5dig.slurm

#SBATCH -J gpt2-sme-5dig
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_sme_5dig_%j.log
#SBATCH -e slurm_logs/gpt2_sme_5dig_%j.log

mkdir -p slurm_logs
set -euo pipefail

TRAIN_DIR="/work/m24047/m24047brmn/numbers/GPT2/fe"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Path to pretrained NumberEncoder checkpoint
NUM_EMB_CHECKPOINT="${NUM_EMB_CHECKPOINT:-/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v8_500k_model.pt}"

# Dataset
DATASET="${DATASET:-numtasks_sme_5dig}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig}"

# Training hyperparameters (matching the successful 3-digit run)
MAX_ITERS="${MAX_ITERS:-20000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.5}"

# Init mode
INIT_FROM="${INIT_FROM:-scratch}"
RESUME_CKPT="${RESUME_CKPT:-}"

# Output directory for checkpoints
OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig}"

GPUS_PER_NODE=2

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

module load gnu/11.2.0

echo "=========================================="
echo "GPT-2 SME Training — 5-digit (no curriculum)"
echo "  Nodes:          ${SLURM_NNODES}"
echo "  GPUs/node:      ${GPUS_PER_NODE}"
echo "  Total GPUs:     $((SLURM_NNODES * GPUS_PER_NODE))"
echo "  Dataset:        ${DATASET}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Checkpoint:     ${NUM_EMB_CHECKPOINT}"
echo "  Max iters:      ${MAX_ITERS}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Adapter lr x:   ${ADAPTER_LR_SCALE}"
echo "  Init from:      ${INIT_FROM}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "  Resume ckpt:    ${RESUME_CKPT}"
fi
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
      num_emb_checkpoint="${NUM_EMB_CHECKPOINT}" \
      out_dir="${OUT_DIR}" \
      learning_rate="${LEARNING_RATE}" \
      max_iters="${MAX_ITERS}" \
      lr_decay_iters="${MAX_ITERS}" \
      adapter_lr_scale="${ADAPTER_LR_SCALE}" \
      resume_ckpt="${RESUME_CKPT}" \
      init_from="${INIT_FROM}"

status=$?
echo "=========================================="
echo "Training finished with status $status"
echo "=========================================="
exit $status
