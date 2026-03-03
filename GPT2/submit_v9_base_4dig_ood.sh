#!/bin/bash
# Submit a full BASE vs FE-v9 experiment:
#   - Train both on fixed 4-digit data
#   - Evaluate on ID + multiple OOD splits
#
# OOD splits:
#   1) Digit OOD low:  1-3 significant digits
#   2) Digit OOD high: 5-8 significant digits
#   3) Magnitude OOD:  4 digits, larger numeric range
#
# Usage:
#   bash submit_v9_base_4dig_ood.sh
#   TAG=myrun TRAIN_RANGE=9999 MAX_ITERS=25000 bash submit_v9_base_4dig_ood.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node."
  exit 1
fi

DATA_ROOT="/tmpdir/m24047brmn/numbers/data"
MODEL_ROOT="/tmpdir/m24047brmn/numbers/model_checkpoints"

TAG="${TAG:-4dig_r10k_$(date +%Y%m%d_%H%M%S)}"

# Training data config (ID)
# Use 9999 by default to keep strict 4-digit integer bounds.
TRAIN_RANGE="${TRAIN_RANGE:-9999}"      # [-9999, 9999]
TRAIN_N="${TRAIN_N:-5000000}"
TRAIN_VAL_N="${TRAIN_VAL_N:-10000}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-10}"
TRAIN_SIG_MIN="${TRAIN_SIG_MIN:-4}"
TRAIN_SIG_MAX="${TRAIN_SIG_MAX:-4}"

# OOD eval data config
EVAL_N="${EVAL_N:-20000}"
EVAL_VAL_N="${EVAL_VAL_N:-10000}"
OOD_MAX_LEN="${OOD_MAX_LEN:-10}"
OOD_LOW_SIG_MIN="${OOD_LOW_SIG_MIN:-1}"
OOD_LOW_SIG_MAX="${OOD_LOW_SIG_MAX:-3}"
OOD_HIGH_SIG_MIN="${OOD_HIGH_SIG_MIN:-5}"
OOD_HIGH_SIG_MAX="${OOD_HIGH_SIG_MAX:-8}"
OOD_HIGH_RANGE="${OOD_HIGH_RANGE:-10000}"   # keep same range by default
OOD_MAG_RANGE="${OOD_MAG_RANGE:-100000000}" # larger magnitude OOD (1e8)

# Training config
MAX_ITERS="${MAX_ITERS:-35000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
ADAPTER_LR_SCALE="${ADAPTER_LR_SCALE:-0.2}"

# Paths
SME_TRAIN_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_id_d4"
BASE_TRAIN_DIR="${DATA_ROOT}/numtasks_base_${TAG}_id_d4"

SME_OOD_LOW_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_d1_3"
BASE_OOD_LOW_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_d1_3"

SME_OOD_HIGH_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_d5_8"
BASE_OOD_HIGH_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_d5_8"

SME_OOD_MAG_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_mag"
BASE_OOD_MAG_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_mag"

SME_OUT_DIR="${MODEL_ROOT}/sme_v9_${TAG}"
BASE_OUT_DIR="${MODEL_ROOT}/base_${TAG}"

echo "=========================================="
echo "BASE vs FE-v9 (4-digit ID + OOD) pipeline"
echo "  TAG:                  ${TAG}"
echo "  ID range:             [-${TRAIN_RANGE}, ${TRAIN_RANGE}]"
echo "  ID sig digits:        ${TRAIN_SIG_MIN}-${TRAIN_SIG_MAX}"
echo "  OOD low sig digits:   ${OOD_LOW_SIG_MIN}-${OOD_LOW_SIG_MAX}"
echo "  OOD high sig digits:  ${OOD_HIGH_SIG_MIN}-${OOD_HIGH_SIG_MAX}"
echo "  OOD mag range:        [-${OOD_MAG_RANGE}, ${OOD_MAG_RANGE}]"
echo "  Train examples:       ${TRAIN_N}"
echo "  Train val examples:   ${TRAIN_VAL_N}"
echo "  Eval examples:        ${EVAL_N}"
echo "  Eval val examples:    ${EVAL_VAL_N}"
echo "  Max iters:            ${MAX_ITERS}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1) Data generation
# ---------------------------------------------------------------------------

# ID train data
GEN_SME_ID=$(sbatch --parsable \
  --job-name=gen-sme-id4 \
  --output=slurm_logs/gen_sme_${TAG}_id_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_id_%j.log \
  --export=ALL,N_TRAIN=${TRAIN_N},N_VAL=${TRAIN_VAL_N},MAX_LEN=${TRAIN_MAX_LEN},NUMBER_RANGE=${TRAIN_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${TRAIN_SIG_MIN},SIG_DIGITS_MAX=${TRAIN_SIG_MAX},OUT_DIR=${SME_TRAIN_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME ID:      ${GEN_SME_ID}"

GEN_BASE_ID=$(sbatch --parsable \
  --job-name=gen-base-id4 \
  --output=slurm_logs/gen_base_${TAG}_id_%j.log \
  --error=slurm_logs/gen_base_${TAG}_id_%j.log \
  --export=ALL,N_TRAIN=${TRAIN_N},N_VAL=${TRAIN_VAL_N},MAX_LEN=${TRAIN_MAX_LEN},NUMBER_RANGE=${TRAIN_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${TRAIN_SIG_MIN},SIG_DIGITS_MAX=${TRAIN_SIG_MAX},OUT_DIR=${BASE_TRAIN_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE ID:     ${GEN_BASE_ID}"

# OOD eval data: low digits
GEN_SME_OOD_LOW=$(sbatch --parsable \
  --job-name=gen-sme-o1 \
  --output=slurm_logs/gen_sme_${TAG}_oodlow_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_oodlow_%j.log \
  --export=ALL,N_TRAIN=${EVAL_N},N_VAL=${EVAL_VAL_N},MAX_LEN=${OOD_MAX_LEN},NUMBER_RANGE=${TRAIN_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_LOW_SIG_MIN},SIG_DIGITS_MAX=${OOD_LOW_SIG_MAX},OUT_DIR=${SME_OOD_LOW_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME OOD LOW: ${GEN_SME_OOD_LOW}"

GEN_BASE_OOD_LOW=$(sbatch --parsable \
  --job-name=gen-base-o1 \
  --output=slurm_logs/gen_base_${TAG}_oodlow_%j.log \
  --error=slurm_logs/gen_base_${TAG}_oodlow_%j.log \
  --export=ALL,N_TRAIN=${EVAL_N},N_VAL=${EVAL_VAL_N},MAX_LEN=${OOD_MAX_LEN},NUMBER_RANGE=${TRAIN_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_LOW_SIG_MIN},SIG_DIGITS_MAX=${OOD_LOW_SIG_MAX},OUT_DIR=${BASE_OOD_LOW_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE OOD LOW:${GEN_BASE_OOD_LOW}"

# OOD eval data: high digits
GEN_SME_OOD_HIGH=$(sbatch --parsable \
  --job-name=gen-sme-o2 \
  --output=slurm_logs/gen_sme_${TAG}_oodhigh_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_oodhigh_%j.log \
  --export=ALL,N_TRAIN=${EVAL_N},N_VAL=${EVAL_VAL_N},MAX_LEN=${OOD_MAX_LEN},NUMBER_RANGE=${OOD_HIGH_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_HIGH_SIG_MIN},SIG_DIGITS_MAX=${OOD_HIGH_SIG_MAX},OUT_DIR=${SME_OOD_HIGH_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME OOD HIGH:${GEN_SME_OOD_HIGH}"

GEN_BASE_OOD_HIGH=$(sbatch --parsable \
  --job-name=gen-base-o2 \
  --output=slurm_logs/gen_base_${TAG}_oodhigh_%j.log \
  --error=slurm_logs/gen_base_${TAG}_oodhigh_%j.log \
  --export=ALL,N_TRAIN=${EVAL_N},N_VAL=${EVAL_VAL_N},MAX_LEN=${OOD_MAX_LEN},NUMBER_RANGE=${OOD_HIGH_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_HIGH_SIG_MIN},SIG_DIGITS_MAX=${OOD_HIGH_SIG_MAX},OUT_DIR=${BASE_OOD_HIGH_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE OOD HIGH:${GEN_BASE_OOD_HIGH}"

# OOD eval data: magnitude
GEN_SME_OOD_MAG=$(sbatch --parsable \
  --job-name=gen-sme-o3 \
  --output=slurm_logs/gen_sme_${TAG}_oodmag_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_oodmag_%j.log \
  --export=ALL,N_TRAIN=${EVAL_N},N_VAL=${EVAL_VAL_N},MAX_LEN=${OOD_MAX_LEN},NUMBER_RANGE=${OOD_MAG_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${TRAIN_SIG_MIN},SIG_DIGITS_MAX=${TRAIN_SIG_MAX},OUT_DIR=${SME_OOD_MAG_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME OOD MAG: ${GEN_SME_OOD_MAG}"

GEN_BASE_OOD_MAG=$(sbatch --parsable \
  --job-name=gen-base-o3 \
  --output=slurm_logs/gen_base_${TAG}_oodmag_%j.log \
  --error=slurm_logs/gen_base_${TAG}_oodmag_%j.log \
  --export=ALL,N_TRAIN=${EVAL_N},N_VAL=${EVAL_VAL_N},MAX_LEN=${OOD_MAX_LEN},NUMBER_RANGE=${OOD_MAG_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${TRAIN_SIG_MIN},SIG_DIGITS_MAX=${TRAIN_SIG_MAX},OUT_DIR=${BASE_OOD_MAG_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE OOD MAG:${GEN_BASE_OOD_MAG}"

# ---------------------------------------------------------------------------
# 2) Training
# ---------------------------------------------------------------------------

TRAIN_SME=$(sbatch --parsable \
  --dependency=afterok:${GEN_SME_ID} \
  --job-name=train-sme4 \
  --output=slurm_logs/gpt2_sme_v9_${TAG}_%j.log \
  --error=slurm_logs/gpt2_sme_v9_${TAG}_%j.log \
  --export=ALL,MAX_ITERS=${MAX_ITERS},LEARNING_RATE=${LEARNING_RATE},ADAPTER_LR_SCALE=${ADAPTER_LR_SCALE},DATASET=numtasks_sme_${TAG}_id_d4,DATA_DIR=${SME_TRAIN_DIR},OUT_DIR=${SME_OUT_DIR},RUN_TAG=${TAG} \
  run_fe_train_v9.slurm)
echo "TRAIN SME-v9:    ${TRAIN_SME}  (after ${GEN_SME_ID})"

TRAIN_BASE=$(sbatch --parsable \
  --dependency=afterok:${GEN_BASE_ID} \
  --job-name=train-base4 \
  --output=slurm_logs/gpt2_base_${TAG}_%j.log \
  --error=slurm_logs/gpt2_base_${TAG}_%j.log \
  --export=ALL,MAX_ITERS=${MAX_ITERS},LEARNING_RATE=${LEARNING_RATE},DATASET=numtasks_base_${TAG}_id_d4,DATA_DIR=${BASE_TRAIN_DIR},OUT_DIR=${BASE_OUT_DIR} \
  run_base_train_5dig.slurm)
echo "TRAIN BASE:      ${TRAIN_BASE}  (after ${GEN_BASE_ID})"

# ---------------------------------------------------------------------------
# 3) Validation (ID + OOD)
# ---------------------------------------------------------------------------

# ID validation
VAL_SME_ID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME} \
  --job-name=val-sme-id4 \
  --output=slurm_logs/validate_v9_${TAG}_id_%j.log \
  --error=slurm_logs/validate_v9_${TAG}_id_%j.log \
  --export=ALL,CKPT_PATH=${SME_OUT_DIR}/ckpt_best.pt,DATA_DIR=${SME_TRAIN_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_id_d4.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_id_d4.json,SUM_GEN_COUNT=200 \
  run_validate_v9.slurm)
echo "VAL SME ID:      ${VAL_SME_ID}"

VAL_BASE_ID=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_BASE} \
  --job-name=val-base-id4 \
  --output=slurm_logs/validate_base_${TAG}_id_%j.log \
  --error=slurm_logs/validate_base_${TAG}_id_%j.log \
  --export=ALL,CKPT_PATH=${BASE_OUT_DIR}/ckpt.pt,DATA_DIR=${BASE_TRAIN_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_id_d4.json \
  run_base_validate_5dig.slurm)
echo "VAL BASE ID:     ${VAL_BASE_ID}"

# OOD low digits
VAL_SME_OOD_LOW=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME}:${GEN_SME_OOD_LOW} \
  --job-name=val-sme-o1 \
  --output=slurm_logs/validate_v9_${TAG}_oodlow_%j.log \
  --error=slurm_logs/validate_v9_${TAG}_oodlow_%j.log \
  --export=ALL,CKPT_PATH=${SME_OUT_DIR}/ckpt_best.pt,DATA_DIR=${SME_OOD_LOW_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_ood_d1_3.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_d1_3.json,SUM_GEN_COUNT=100 \
  run_validate_v9.slurm)
echo "VAL SME OOD LOW: ${VAL_SME_OOD_LOW}"

VAL_BASE_OOD_LOW=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_BASE}:${GEN_BASE_OOD_LOW} \
  --job-name=val-base-o1 \
  --output=slurm_logs/validate_base_${TAG}_oodlow_%j.log \
  --error=slurm_logs/validate_base_${TAG}_oodlow_%j.log \
  --export=ALL,CKPT_PATH=${BASE_OUT_DIR}/ckpt.pt,DATA_DIR=${BASE_OOD_LOW_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_d1_3.json \
  run_base_validate_5dig.slurm)
echo "VAL BASE OOD LOW:${VAL_BASE_OOD_LOW}"

# OOD high digits
VAL_SME_OOD_HIGH=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME}:${GEN_SME_OOD_HIGH} \
  --job-name=val-sme-o2 \
  --output=slurm_logs/validate_v9_${TAG}_oodhigh_%j.log \
  --error=slurm_logs/validate_v9_${TAG}_oodhigh_%j.log \
  --export=ALL,CKPT_PATH=${SME_OUT_DIR}/ckpt_best.pt,DATA_DIR=${SME_OOD_HIGH_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_ood_d5_8.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_d5_8.json,SUM_GEN_COUNT=100 \
  run_validate_v9.slurm)
echo "VAL SME OOD HIGH:${VAL_SME_OOD_HIGH}"

VAL_BASE_OOD_HIGH=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_BASE}:${GEN_BASE_OOD_HIGH} \
  --job-name=val-base-o2 \
  --output=slurm_logs/validate_base_${TAG}_oodhigh_%j.log \
  --error=slurm_logs/validate_base_${TAG}_oodhigh_%j.log \
  --export=ALL,CKPT_PATH=${BASE_OUT_DIR}/ckpt.pt,DATA_DIR=${BASE_OOD_HIGH_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_d5_8.json \
  run_base_validate_5dig.slurm)
echo "VAL BASE OOD HIGH:${VAL_BASE_OOD_HIGH}"

# OOD magnitude
VAL_SME_OOD_MAG=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME}:${GEN_SME_OOD_MAG} \
  --job-name=val-sme-o3 \
  --output=slurm_logs/validate_v9_${TAG}_oodmag_%j.log \
  --error=slurm_logs/validate_v9_${TAG}_oodmag_%j.log \
  --export=ALL,CKPT_PATH=${SME_OUT_DIR}/ckpt_best.pt,DATA_DIR=${SME_OOD_MAG_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_ood_mag.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_mag.json,SUM_GEN_COUNT=100 \
  run_validate_v9.slurm)
echo "VAL SME OOD MAG: ${VAL_SME_OOD_MAG}"

VAL_BASE_OOD_MAG=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_BASE}:${GEN_BASE_OOD_MAG} \
  --job-name=val-base-o3 \
  --output=slurm_logs/validate_base_${TAG}_oodmag_%j.log \
  --error=slurm_logs/validate_base_${TAG}_oodmag_%j.log \
  --export=ALL,CKPT_PATH=${BASE_OUT_DIR}/ckpt.pt,DATA_DIR=${BASE_OOD_MAG_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_mag.json \
  run_base_validate_5dig.slurm)
echo "VAL BASE OOD MAG:${VAL_BASE_OOD_MAG}"

echo ""
echo "=========================================="
echo "Queued pipeline for TAG=${TAG}"
echo "  ID data jobs:        SME ${GEN_SME_ID}, BASE ${GEN_BASE_ID}"
echo "  OOD data jobs:       SME ${GEN_SME_OOD_LOW}/${GEN_SME_OOD_HIGH}/${GEN_SME_OOD_MAG}"
echo "                       BASE ${GEN_BASE_OOD_LOW}/${GEN_BASE_OOD_HIGH}/${GEN_BASE_OOD_MAG}"
echo "  Train jobs:          SME ${TRAIN_SME}, BASE ${TRAIN_BASE}"
echo "  ID val jobs:         SME ${VAL_SME_ID}, BASE ${VAL_BASE_ID}"
echo "  OOD val jobs:        SME ${VAL_SME_OOD_LOW}/${VAL_SME_OOD_HIGH}/${VAL_SME_OOD_MAG}"
echo "                       BASE ${VAL_BASE_OOD_LOW}/${VAL_BASE_OOD_HIGH}/${VAL_BASE_OOD_MAG}"
echo "=========================================="
