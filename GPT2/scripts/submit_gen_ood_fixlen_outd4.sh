#!/bin/bash
# Generate OOD evaluation datasets where:
# - Input significant-digit distribution is shifted (1-3 or 5-8), OR magnitude is shifted
# - Output precision is FIXED to 4 sig digits (teacher-forced compatibility with d4 models)
# - SME outputs are padded to 4 mantissa digits (fixlen), so END is aligned in eval
#
# This produces matched BASE and SME datasets (same seed -> same underlying examples).
#
# Usage:
#   bash submit_gen_ood_fixlen_outd4.sh
#   TAG=mytag bash submit_gen_ood_fixlen_outd4.sh
#
# Outputs (under /tmpdir/.../numbers/data):
#   numtasks_sme_${TAG}_ood_d1_3
#   numtasks_base_${TAG}_ood_d1_3
#   numtasks_sme_${TAG}_ood_d5_8
#   numtasks_base_${TAG}_ood_d5_8
#   numtasks_sme_${TAG}_ood_mag
#   numtasks_base_${TAG}_ood_mag

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node."
  exit 1
fi

DATA_ROOT="/tmpdir/m24047brmn/numbers/data"

TAG="${TAG:-ood_fixlen_outd4_$(date +%Y%m%d_%H%M%S)}"

# Dataset sizes (keep OOD generation cheap; val is what we actually score).
N_TRAIN="${N_TRAIN:-20000}"
N_VAL="${N_VAL:-10000}"
MAX_LEN="${MAX_LEN:-10}"

# Input distributions
ID_RANGE="${ID_RANGE:-9999}"
OOD_LOW_SIG_MIN="${OOD_LOW_SIG_MIN:-1}"
OOD_LOW_SIG_MAX="${OOD_LOW_SIG_MAX:-3}"
OOD_HIGH_SIG_MIN="${OOD_HIGH_SIG_MIN:-5}"
OOD_HIGH_SIG_MAX="${OOD_HIGH_SIG_MAX:-8}"
OOD_MAG_RANGE="${OOD_MAG_RANGE:-100000000}"

# Output formatting (must match your fixlen d4 model)
OUT_SIG_DIGITS_MAX="${OUT_SIG_DIGITS_MAX:-4}"
SME_MIN_DIGITS="${SME_MIN_DIGITS:-4}"

SME_OOD_LOW_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_d1_3"
BASE_OOD_LOW_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_d1_3"

SME_OOD_HIGH_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_d5_8"
BASE_OOD_HIGH_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_d5_8"

SME_OOD_MAG_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_mag"
BASE_OOD_MAG_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_mag"

echo "=========================================="
echo "Generate OOD datasets (fixlen outputs, out-digits=4)"
echo "  TAG:                 ${TAG}"
echo "  N_TRAIN/N_VAL:       ${N_TRAIN}/${N_VAL}"
echo "  MAX_LEN:             ${MAX_LEN}"
echo "  ID_RANGE:            ${ID_RANGE}"
echo "  OOD low sig digits:  ${OOD_LOW_SIG_MIN}-${OOD_LOW_SIG_MAX} (inputs)"
echo "  OOD high sig digits: ${OOD_HIGH_SIG_MIN}-${OOD_HIGH_SIG_MAX} (inputs)"
echo "  OOD mag range:       [-${OOD_MAG_RANGE}, ${OOD_MAG_RANGE}] (inputs)"
echo "  OUT_SIG_DIGITS_MAX:  ${OUT_SIG_DIGITS_MAX} (outputs)"
echo "  SME_MIN_DIGITS:      ${SME_MIN_DIGITS} (outputs, padded)"
echo "------------------------------------------"
echo "  SME_OOD_LOW_DIR:     ${SME_OOD_LOW_DIR}"
echo "  BASE_OOD_LOW_DIR:    ${BASE_OOD_LOW_DIR}"
echo "  SME_OOD_HIGH_DIR:    ${SME_OOD_HIGH_DIR}"
echo "  BASE_OOD_HIGH_DIR:   ${BASE_OOD_HIGH_DIR}"
echo "  SME_OOD_MAG_DIR:     ${SME_OOD_MAG_DIR}"
echo "  BASE_OOD_MAG_DIR:    ${BASE_OOD_MAG_DIR}"
echo "=========================================="

# OOD low: input sig digits 1-3, output digits fixed at 4, SME padded to 4.
GEN_SME_OOD_LOW=$(sbatch --parsable \
  --job-name=gen-sme-o1 \
  --output=slurm_logs/gen_sme_${TAG}_oodlow_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_oodlow_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${ID_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_LOW_SIG_MIN},SIG_DIGITS_MAX=${OOD_LOW_SIG_MAX},OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX},SME_MIN_DIGITS=${SME_MIN_DIGITS},OUT_DIR=${SME_OOD_LOW_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME OOD LOW:  ${GEN_SME_OOD_LOW}"

GEN_BASE_OOD_LOW=$(sbatch --parsable \
  --job-name=gen-base-o1 \
  --output=slurm_logs/gen_base_${TAG}_oodlow_%j.log \
  --error=slurm_logs/gen_base_${TAG}_oodlow_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${ID_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_LOW_SIG_MIN},SIG_DIGITS_MAX=${OOD_LOW_SIG_MAX},OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX},OUT_DIR=${BASE_OOD_LOW_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE OOD LOW: ${GEN_BASE_OOD_LOW}"

# OOD high: input sig digits 5-8, but keep output digits fixed at 4 (matches model).
GEN_SME_OOD_HIGH=$(sbatch --parsable \
  --job-name=gen-sme-o2 \
  --output=slurm_logs/gen_sme_${TAG}_oodhigh_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_oodhigh_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${ID_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_HIGH_SIG_MIN},SIG_DIGITS_MAX=${OOD_HIGH_SIG_MAX},OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX},SME_MIN_DIGITS=${SME_MIN_DIGITS},OUT_DIR=${SME_OOD_HIGH_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME OOD HIGH: ${GEN_SME_OOD_HIGH}"

GEN_BASE_OOD_HIGH=$(sbatch --parsable \
  --job-name=gen-base-o2 \
  --output=slurm_logs/gen_base_${TAG}_oodhigh_%j.log \
  --error=slurm_logs/gen_base_${TAG}_oodhigh_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${ID_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=${OOD_HIGH_SIG_MIN},SIG_DIGITS_MAX=${OOD_HIGH_SIG_MAX},OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX},OUT_DIR=${BASE_OOD_HIGH_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE OOD HIGH:${GEN_BASE_OOD_HIGH}"

# OOD magnitude: larger number range, keep input digits at 4 (ID), output digits fixed at 4.
GEN_SME_OOD_MAG=$(sbatch --parsable \
  --job-name=gen-sme-o3 \
  --output=slurm_logs/gen_sme_${TAG}_oodmag_%j.log \
  --error=slurm_logs/gen_sme_${TAG}_oodmag_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${OOD_MAG_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=4,SIG_DIGITS_MAX=4,OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX},SME_MIN_DIGITS=${SME_MIN_DIGITS},OUT_DIR=${SME_OOD_MAG_DIR} \
  run_generate_data_sme_5dig.slurm)
echo "GEN SME OOD MAG:  ${GEN_SME_OOD_MAG}"

GEN_BASE_OOD_MAG=$(sbatch --parsable \
  --job-name=gen-base-o3 \
  --output=slurm_logs/gen_base_${TAG}_oodmag_%j.log \
  --error=slurm_logs/gen_base_${TAG}_oodmag_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},MAX_LEN=${MAX_LEN},NUMBER_RANGE=${OOD_MAG_RANGE},REASONING_WEIGHT=1,NUMERIC_WEIGHT=2,SIG_DIGITS_MIN=4,SIG_DIGITS_MAX=4,OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX},OUT_DIR=${BASE_OOD_MAG_DIR} \
  run_generate_data_base_5dig.slurm)
echo "GEN BASE OOD MAG: ${GEN_BASE_OOD_MAG}"

echo
echo "Submitted."
echo "When jobs finish, datasets will be in:"
echo "  ${SME_OOD_LOW_DIR}"
echo "  ${SME_OOD_HIGH_DIR}"
echo "  ${SME_OOD_MAG_DIR}"
echo "  ${BASE_OOD_LOW_DIR}"
echo "  ${BASE_OOD_HIGH_DIR}"
echo "  ${BASE_OOD_MAG_DIR}"

