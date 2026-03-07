#!/bin/bash
# Submit an FE-v9 fixed-length SME mantissa experiment (SME only) with
# digit-position weighted loss (d0..dK).
#
# Pipeline:
#   1) Generate ID dataset with fixed sig digits + fixed SME output length
#   2) Train FE-v9 (digit-weighted loss enabled)
#   3) Validate (standard + extended)
#   4) Error breakdown
#
# Usage:
#   bash submit_sme_v9_fixedlen_weighted.sh
#
# Common overrides:
#   TAG=... TRAIN_RANGE=9999 MAX_ITERS=25000 bash submit_sme_v9_fixedlen_weighted.sh
#   SME_MIN_DIGITS=4 SIG_DIGITS_MIN=4 SIG_DIGITS_MAX=4 bash submit_sme_v9_fixedlen_weighted.sh
#   DIGIT_LOSS_POS_WEIGHTS="0.3:0.8:1.3:1.6" bash submit_sme_v9_fixedlen_weighted.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node." >&2
  exit 1
fi

DATA_ROOT="/tmpdir/m24047brmn/numbers/data"
MODEL_ROOT="/tmpdir/m24047brmn/numbers/model_checkpoints"

TAG="${TAG:-v9_4dig_r10k_fixlen_wdig_a05_$(date +%Y%m%d_%H%M%S)}"

# Reuse an existing fixed-length (fixlen) dataset to avoid regeneration.
# Set EXISTING_DATA_DIR="" to force regeneration.
EXISTING_DATA_DIR="${EXISTING_DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_v9_4dig_r10k_fixlen_20260304_035452_id_d4}"

# Data config (ID)
TRAIN_RANGE="${TRAIN_RANGE:-9999}"      # [-9999, 9999]
N_TRAIN="${N_TRAIN:-5000000}"
N_VAL="${N_VAL:-10000}"
MAX_LEN="${MAX_LEN:-10}"
REASONING_WEIGHT="${REASONING_WEIGHT:-1}"
NUMERIC_WEIGHT="${NUMERIC_WEIGHT:-2}"

# Significant digits for sampling floats (also caps output precision in generator)
SIG_DIGITS_MIN="${SIG_DIGITS_MIN:-4}"
SIG_DIGITS_MAX="${SIG_DIGITS_MAX:-4}"

# Fixed-length SME mantissa for *outputs* (pads trailing zeros).
SME_MIN_DIGITS="${SME_MIN_DIGITS:-4}"

# Digit-loss weights (alpha=0.5 from the fixlen baseline digit accuracies).
# IMPORTANT: use ':' (or spaces) not commas, because sbatch --export uses commas.
DIGIT_LOSS_ENABLE="${DIGIT_LOSS_ENABLE:-True}"
DIGIT_LOSS_POS_WEIGHTS="${DIGIT_LOSS_POS_WEIGHTS:-0.385:0.864:1.335:1.416}"
DIGIT_LOSS_NORMALIZE_MEAN="${DIGIT_LOSS_NORMALIZE_MEAN:-True}"

# Training config
MAX_ITERS="${MAX_ITERS:-35000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.2}"

# Paths
if [ -n "${EXISTING_DATA_DIR}" ]; then
  SME_TRAIN_DIR="${EXISTING_DATA_DIR}"
  DATASET="${DATASET:-$(basename "${SME_TRAIN_DIR}")}"
  SKIP_GEN=1
else
  SME_TRAIN_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_id_d4"
  DATASET="${DATASET:-numtasks_sme_${TAG}_id_d4}"
  SKIP_GEN=0
fi
SME_OUT_DIR="${MODEL_ROOT}/sme_v9_${TAG}"

echo "=========================================="
echo "FE-v9 fixed-length + weighted digit loss"
echo "  TAG:              ${TAG}"
echo "  Skip gen:         ${SKIP_GEN} (existing='${EXISTING_DATA_DIR}')"
echo "  ID range:         [-${TRAIN_RANGE}, ${TRAIN_RANGE}]"
echo "  Sig digits:       ${SIG_DIGITS_MIN}-${SIG_DIGITS_MAX}"
echo "  SME min digits:   ${SME_MIN_DIGITS}"
echo "  Digit loss:       enable=${DIGIT_LOSS_ENABLE}"
echo "  Digit weights:    ${DIGIT_LOSS_POS_WEIGHTS} (normalize_mean=${DIGIT_LOSS_NORMALIZE_MEAN})"
echo "  Train examples:   ${N_TRAIN}"
echo "  Val examples:     ${N_VAL}"
echo "  Max iters:        ${MAX_ITERS}"
echo "  Data dir:         ${SME_TRAIN_DIR}"
echo "  Dataset:          ${DATASET}"
echo "  Out dir:          ${SME_OUT_DIR}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1) Data generation
# ---------------------------------------------------------------------------
GEN_SME_ID=""
if [ "${SKIP_GEN}" -eq 0 ]; then
  GEN_SME_ID=$(sbatch --parsable \
    --job-name=gen-sme-id4 \
    --output=slurm_logs/gen_sme_${TAG}_id_%j.log \
    --error=slurm_logs/gen_sme_${TAG}_id_%j.log \
    --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${TRAIN_RANGE},REASONING_WEIGHT=${REASONING_WEIGHT},NUMERIC_WEIGHT=${NUMERIC_WEIGHT},SIG_DIGITS_MIN=${SIG_DIGITS_MIN},SIG_DIGITS_MAX=${SIG_DIGITS_MAX},SME_MIN_DIGITS=${SME_MIN_DIGITS},OUT_DIR=${SME_TRAIN_DIR} \
    run_generate_data_sme_5dig.slurm)
  echo "GEN SME ID:    ${GEN_SME_ID}"
else
  echo "GEN SME ID:    (skipped; reusing ${SME_TRAIN_DIR})"
fi

# ---------------------------------------------------------------------------
# 2) Training
# ---------------------------------------------------------------------------
TRAIN_DEP=()
if [ -n "${GEN_SME_ID}" ]; then
  TRAIN_DEP=(--dependency=afterok:${GEN_SME_ID})
fi
TRAIN_SME=$(sbatch --parsable \
  "${TRAIN_DEP[@]}" \
  --job-name=train-sme-v9 \
  --output=slurm_logs/gpt2_sme_v9_${TAG}_%j.log \
  --error=slurm_logs/gpt2_sme_v9_${TAG}_%j.log \
  --export=ALL,MAX_ITERS=${MAX_ITERS},LEARNING_RATE=${LEARNING_RATE},ADAPTER_LR_SCALE=${ADAPTER_LR_SCALE},DATASET=${DATASET},DATA_DIR=${SME_TRAIN_DIR},OUT_DIR=${SME_OUT_DIR},RUN_TAG=${TAG},DIGIT_LOSS_ENABLE=${DIGIT_LOSS_ENABLE},DIGIT_LOSS_POS_WEIGHTS=${DIGIT_LOSS_POS_WEIGHTS},DIGIT_LOSS_NORMALIZE_MEAN=${DIGIT_LOSS_NORMALIZE_MEAN} \
  run_fe_train_v9.slurm)
if [ -n "${GEN_SME_ID}" ]; then
  echo "TRAIN SME-v9:  ${TRAIN_SME}  (after ${GEN_SME_ID})"
else
  echo "TRAIN SME-v9:  ${TRAIN_SME}"
fi

# ---------------------------------------------------------------------------
# 3) Validation
# ---------------------------------------------------------------------------
VAL_SME_ID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME} \
  --job-name=val-sme-v9-id4 \
  --output=slurm_logs/validate_v9_${TAG}_id_%j.log \
  --error=slurm_logs/validate_v9_${TAG}_id_%j.log \
  --export=ALL,CKPT_PATH=${SME_OUT_DIR}/ckpt_best.pt,DATA_DIR=${SME_TRAIN_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_id_d4.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_id_d4.json,SUM_GEN_COUNT=200 \
  run_validate_v9.slurm)
echo "VAL SME ID:    ${VAL_SME_ID}"

# ---------------------------------------------------------------------------
# 4) Error breakdown (post-train)
# ---------------------------------------------------------------------------
ERR_SME_ID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME} \
  --job-name=err-sme-v9-id4 \
  --output=slurm_logs/error_breakdown_v9_${TAG}_id_%j.log \
  --error=slurm_logs/error_breakdown_v9_${TAG}_id_%j.log \
  --export=ALL,CKPT_PATH=${SME_OUT_DIR}/ckpt_best.pt,DATA_DIR=${SME_TRAIN_DIR},OUTPUT_JSON=${SME_OUT_DIR}/error_breakdown_id_d4.json,BASE_EXACT_RATE= \
  run_error_breakdown_v9.slurm)
echo "ERR SME ID:    ${ERR_SME_ID}"

echo ""
echo "=========================================="
echo "Queued FE-v9 weighted fixlen run: TAG=${TAG}"
echo "  GEN=${GEN_SME_ID}"
echo "  TRAIN=${TRAIN_SME}"
echo "  VAL=${VAL_SME_ID}"
echo "  ERR=${ERR_SME_ID}"
echo "Track with:"
echo "  squeue -j ${GEN_SME_ID},${TRAIN_SME},${VAL_SME_ID},${ERR_SME_ID}"
