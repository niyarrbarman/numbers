#!/bin/bash
# Train one curriculum phase for FE SME with explicit resume behavior.
#
# Usage:
#   PHASE=1 sbatch run_fe_train_sme_phase.slurm
#   PHASE=2 sbatch run_fe_train_sme_phase.slurm
#   PHASE=3 sbatch run_fe_train_sme_phase.slurm
#
# Phase behavior:
#   1: scratch on phase-1 data, up to P1_MAX_ITERS
#   2: resume on phase-2 data, up to P2_MAX_ITERS (cumulative)
#   3: resume on phase-3 data, up to P3_MAX_ITERS (cumulative)

#SBATCH -J gpt2-sme-phase
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/gpt2_sme_phase_%j.log
#SBATCH -e slurm_logs/gpt2_sme_phase_%j.log

mkdir -p slurm_logs
set -euo pipefail

TRAIN_DIR="/work/m24047/m24047brmn/numbers/GPT2/fe"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Path to pretrained NumberEncoder checkpoint
NUM_EMB_CHECKPOINT="${NUM_EMB_CHECKPOINT:-/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v8_500k_model.pt}"

PHASE="${PHASE:-1}"  # 1,2,3

BASE_DATA_DIR="${BASE_DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_vardig_e9_numw2_curr_stage_v1}"
PHASE1_DATA_DIR="${PHASE1_DATA_DIR:-${BASE_DATA_DIR}_p1_d1_4}"
PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-${BASE_DATA_DIR}_p2_d5_8}"
PHASE3_DATA_DIR="${PHASE3_DATA_DIR:-${BASE_DATA_DIR}_p3_d1_15}"

OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_vardig_e9_numw2_curr_stage_v1}"
DATASET="${DATASET:-numtasks_sme_vardig_e9_numw2_curr_stage_v1}"

# Cumulative max iters by phase (override as needed).
P1_MAX_ITERS="${P1_MAX_ITERS:-12000}"
P2_MAX_ITERS="${P2_MAX_ITERS:-26000}"
P3_MAX_ITERS="${P3_MAX_ITERS:-40000}"

MAX_ITERS="${MAX_ITERS:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
DATA_DIR="${DATA_DIR:-}"

LEARNING_RATE="${LEARNING_RATE:-4e-4}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.5}"

case "${PHASE}" in
  1)
    phase_name="phase1_short"
    default_data_dir="${PHASE1_DATA_DIR}"
    default_init_from="scratch"
    default_max_iters="${P1_MAX_ITERS}"
    ;;
  2)
    phase_name="phase2_medium"
    default_data_dir="${PHASE2_DATA_DIR}"
    default_init_from="resume"
    default_max_iters="${P2_MAX_ITERS}"
    ;;
  3)
    phase_name="phase3_full"
    default_data_dir="${PHASE3_DATA_DIR}"
    default_init_from="resume"
    default_max_iters="${P3_MAX_ITERS}"
    ;;
  *)
    echo "ERROR: PHASE must be 1, 2, or 3 (got: ${PHASE})" >&2
    exit 2
    ;;
esac

if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${default_data_dir}"
fi
if [[ -z "${INIT_FROM}" ]]; then
  INIT_FROM="${default_init_from}"
fi
if [[ -z "${MAX_ITERS}" ]]; then
  MAX_ITERS="${default_max_iters}"
fi
if [[ -z "${RESUME_CKPT}" && "${INIT_FROM}" == "resume" ]]; then
  RESUME_CKPT="${OUT_DIR}/ckpt_best.pt"
fi

GPUS_PER_NODE=2

# Multi-node coordination
export MASTER_PORT=$(echo "${SLURM_JOB_ID:-0} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)

module load gnu/11.2.0

echo "=========================================="
echo "GPT-2 SME Curriculum Training (${phase_name})"
echo "  Phase:          ${PHASE}"
echo "  Nodes:          ${SLURM_NNODES}"
echo "  GPUs/node:      ${GPUS_PER_NODE}"
echo "  Total GPUs:     $((SLURM_NNODES * GPUS_PER_NODE))"
echo "  Dataset label:  ${DATASET}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Checkpoint:     ${NUM_EMB_CHECKPOINT}"
echo "  Init from:      ${INIT_FROM}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "  Resume ckpt:    ${RESUME_CKPT}"
fi
echo "  Max iters:      ${MAX_ITERS}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Adapter lr x:   ${ADAPTER_LR_SCALE}"
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

