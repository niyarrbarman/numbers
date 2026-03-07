#!/bin/bash
# Train 124M FE GPT-2 with NumberEncoder v9 (multi-GPU).
# Tasks: ADD, SUB, SUM, MIN, MAX
# Text output decoding, blend schedule, norm matching.
#
# Usage:
#   sbatch 124M/scripts/run_fe_train.slurm
#   MAX_ITERS=60000 sbatch 124M/scripts/run_fe_train.slurm

#SBATCH -J gpt2-124M-fe
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_124M_fe_%j.log
#SBATCH -e slurm_logs/gpt2_124M_fe_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAIN_DIR="${REPO_DIR}/124M/fe"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Path to pretrained NumberEncoder checkpoint
NUM_EMB_CHECKPOINT="${NUM_EMB_CHECKPOINT:-/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v9_500k_model.pt}"

# Dataset
DATASET="${DATASET:-numtasks_124M_fe}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_124M_fe}"

# Training
MAX_ITERS="${MAX_ITERS:-40000}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.2}"

# Init mode
INIT_FROM="${INIT_FROM:-scratch}"
RESUME_CKPT="${RESUME_CKPT:-}"

# Output
OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/model_checkpoints/124M_fe}"

GPUS_PER_NODE=2

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

module load gnu/11.2.0

echo "=========================================="
echo "GPT-2 124M FE Training (NumberEncoder v9)"
echo "  Nodes:          ${SLURM_NNODES}"
echo "  GPUs/node:      ${GPUS_PER_NODE}"
echo "  Total GPUs:     $((SLURM_NNODES * GPUS_PER_NODE))"
echo "  Dataset:        ${DATASET}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Encoder ckpt:   ${NUM_EMB_CHECKPOINT}"
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
