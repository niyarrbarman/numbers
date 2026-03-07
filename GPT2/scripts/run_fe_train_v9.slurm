#!/bin/bash
# Train GPT-2 FE-SME with v9 math-aware encoder (UNFROZEN) on 5-digit dataset.
#
# Usage:
#   sbatch run_fe_train_v9.slurm

#SBATCH -J sme-v9
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_sme_v9_%j.log
#SBATCH -e slurm_logs/gpt2_sme_v9_%j.log

mkdir -p slurm_logs
set -euo pipefail

TRAIN_DIR="/work/m24047/m24047brmn/numbers/GPT2/fe_v9"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

NUM_EMB_CHECKPOINT="${NUM_EMB_CHECKPOINT:-/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v9_500k_model.pt}"

DATASET="${DATASET:-numtasks_sme_5dig_5m}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig_5m}"

MAX_ITERS="${MAX_ITERS:-35000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.2}"
NUM_NORM_MATCH="${NUM_NORM_MATCH:-True}"
NUM_BLEND_BETA_START="${NUM_BLEND_BETA_START:-0.0}"
NUM_BLEND_BETA_END="${NUM_BLEND_BETA_END:-1.0}"
NUM_BLEND_WARMUP_ITERS="${NUM_BLEND_WARMUP_ITERS:-2000}"
NUM_BLEND_RAMP_ITERS="${NUM_BLEND_RAMP_ITERS:-18000}"

# Optional: digit-position weighted loss for SME digit tokens (d0..dK).
DIGIT_LOSS_ENABLE="${DIGIT_LOSS_ENABLE:-False}"
# Use colon or space separators here to avoid sbatch --export comma parsing.
# Example: "0.385:0.864:1.335:1.416" (mean normalized in train.py by default).
DIGIT_LOSS_POS_WEIGHTS="${DIGIT_LOSS_POS_WEIGHTS:-}"
DIGIT_LOSS_NORMALIZE_MEAN="${DIGIT_LOSS_NORMALIZE_MEAN:-True}"

INIT_FROM="${INIT_FROM:-scratch}"
BASE_OUT_DIR="${BASE_OUT_DIR:-/tmpdir/m24047brmn/numbers/model_checkpoints}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID}}"
OUT_DIR="${OUT_DIR:-${BASE_OUT_DIR}/sme_5dig_v9_${RUN_TAG}}"

GPUS_PER_NODE=2

export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

module load gnu/11.2.0

echo "=========================================="
echo "GPT-2 FE-SME Training — v9 ENCODER (UNFROZEN)"
echo "  Nodes:          ${SLURM_NNODES}"
echo "  GPUs/node:      ${GPUS_PER_NODE}"
echo "  Total GPUs:     $((SLURM_NNODES * GPUS_PER_NODE))"
echo "  Dataset:        ${DATASET}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Checkpoint:     ${NUM_EMB_CHECKPOINT}"
echo "  Max iters:      ${MAX_ITERS}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Adapter lr x:   ${ADAPTER_LR_SCALE}"
echo "  Num norm match: ${NUM_NORM_MATCH}"
echo "  Num beta:       ${NUM_BLEND_BETA_START} -> ${NUM_BLEND_BETA_END}"
echo "  Num beta ramp:  warmup=${NUM_BLEND_WARMUP_ITERS}, ramp=${NUM_BLEND_RAMP_ITERS}"
echo "  Digit loss:     enable=${DIGIT_LOSS_ENABLE} weights='${DIGIT_LOSS_POS_WEIGHTS}' norm_mean=${DIGIT_LOSS_NORMALIZE_MEAN}"
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
      num_emb_checkpoint="${NUM_EMB_CHECKPOINT}" \
      out_dir="${OUT_DIR}" \
      learning_rate="${LEARNING_RATE}" \
      max_iters="${MAX_ITERS}" \
      lr_decay_iters="${MAX_ITERS}" \
      adapter_lr_scale="${ADAPTER_LR_SCALE}" \
      num_norm_match="${NUM_NORM_MATCH}" \
      num_blend_beta_start="${NUM_BLEND_BETA_START}" \
      num_blend_beta_end="${NUM_BLEND_BETA_END}" \
      num_blend_warmup_iters="${NUM_BLEND_WARMUP_ITERS}" \
      num_blend_ramp_iters="${NUM_BLEND_RAMP_ITERS}" \
      digit_loss_enable="${DIGIT_LOSS_ENABLE}" \
      digit_loss_pos_weights="${DIGIT_LOSS_POS_WEIGHTS}" \
      digit_loss_normalize_mean="${DIGIT_LOSS_NORMALIZE_MEAN}" \
      init_from="${INIT_FROM}"

status=$?
echo "=========================================="
echo "Training finished with status $status"
echo "=========================================="
exit $status
