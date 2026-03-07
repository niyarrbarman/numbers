#!/bin/bash
# One-shot OOD pipeline for your fixed-length (4-digit) SME model:
#   1) Generate OOD eval datasets with:
#      - shifted input sig-digits (1-3 and 5-8) and magnitude (1e8)
#      - outputs capped to 4 sig digits
#      - SME outputs padded to 4 mantissa digits (fixlen)
#   2) Validate BOTH:
#      - FE-v9 fixlen+weighted checkpoint (SME) via run_validate_v9.slurm
#      - Base checkpoint via run_base_validate_5dig.slurm
#
# Usage (run on a SLURM login node):
#   TAG=ood_fixlen_outd4_20260305_120000 bash submit_ood_fixlen_outd4_pipeline.sh
#   # or let TAG auto-generate:
#   bash submit_ood_fixlen_outd4_pipeline.sh

set -euo pipefail

mkdir -p slurm_logs

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-/tmpdir/m24047brmn/numbers/data}"
MODEL_ROOT="${MODEL_ROOT:-/tmpdir/m24047brmn/numbers/model_checkpoints}"

# Dataset tag (used in generated DATA_DIR names)
TAG="${TAG:-ood_fixlen_outd4_$(date +%Y%m%d_%H%M%S)}"

# Existing model checkpoints (override if needed)
SME_MODEL_TAG="${SME_MODEL_TAG:-v9_4dig_r10k_fixlen_wdig_a05_20260304_191943}"
SME_OUT_DIR="${SME_OUT_DIR:-${MODEL_ROOT}/sme_v9_${SME_MODEL_TAG}}"
SME_CKPT="${SME_CKPT:-${SME_OUT_DIR}/ckpt_best.pt}"

BASE_MODEL_TAG="${BASE_MODEL_TAG:-4dig_r10k_20260303_205237}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${MODEL_ROOT}/base_${BASE_MODEL_TAG}}"
BASE_CKPT="${BASE_CKPT:-${BASE_OUT_DIR}/ckpt.pt}"

# OOD dataset sizes (cheap; val is what we score)
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

# Output formatting (match your fixlen d4 model)
OUT_SIG_DIGITS_MAX="${OUT_SIG_DIGITS_MAX:-4}"
SME_MIN_DIGITS="${SME_MIN_DIGITS:-4}"

# Validate knobs
SUM_GEN_COUNT_OOD="${SUM_GEN_COUNT_OOD:-100}"
BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
TOP_K_ERRORS="${TOP_K_ERRORS:-25}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"

need_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }; }
need_dir()  { [[ -d "$1" ]] || { echo "ERROR: missing dir:  $1" >&2; exit 1; }; }

need_file run_generate_data_sme_5dig.slurm
need_file run_generate_data_base_5dig.slurm
need_file run_validate_v9.slurm
need_file run_base_validate_5dig.slurm
need_file "${SME_CKPT}"
need_file "${BASE_CKPT}"
need_dir "${SME_OUT_DIR}"
need_dir "${BASE_OUT_DIR}"

# Dataset output dirs
SME_OOD_LOW_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_d1_3"
BASE_OOD_LOW_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_d1_3"

SME_OOD_HIGH_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_d5_8"
BASE_OOD_HIGH_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_d5_8"

SME_OOD_MAG_DIR="${DATA_ROOT}/numtasks_sme_${TAG}_ood_mag"
BASE_OOD_MAG_DIR="${DATA_ROOT}/numtasks_base_${TAG}_ood_mag"

echo "=========================================="
echo "OOD FIXLEN OUTD4 PIPELINE"
echo "  DATA TAG:           ${TAG}"
echo "  SME CKPT:           ${SME_CKPT}"
echo "  SME OUT_DIR:        ${SME_OUT_DIR}"
echo "  BASE CKPT:          ${BASE_CKPT}"
echo "  BASE OUT_DIR:       ${BASE_OUT_DIR}"
echo "------------------------------------------"
echo "  N_TRAIN/N_VAL:      ${N_TRAIN}/${N_VAL}"
echo "  MAX_LEN:            ${MAX_LEN}"
echo "  ID_RANGE:           ${ID_RANGE}"
echo "  OOD low sig:        ${OOD_LOW_SIG_MIN}-${OOD_LOW_SIG_MAX} (inputs)"
echo "  OOD high sig:       ${OOD_HIGH_SIG_MIN}-${OOD_HIGH_SIG_MAX} (inputs)"
echo "  OOD mag range:      [-${OOD_MAG_RANGE}, ${OOD_MAG_RANGE}] (inputs)"
echo "  OUT_SIG_DIGITS_MAX: ${OUT_SIG_DIGITS_MAX} (outputs)"
echo "  SME_MIN_DIGITS:     ${SME_MIN_DIGITS} (outputs padded)"
echo "=========================================="

submit_job() {
  local jobname="$1"; shift
  local script="$1"; shift
  local logfile="slurm_logs/${jobname}_%j.log"
  local export_arg="ALL"
  for kv in "$@"; do
    export_arg+=",${kv}"
  done
  sbatch --parsable \
    --job-name="${jobname}" \
    --output="${logfile}" \
    --error="${logfile}" \
    --export="${export_arg}" \
    "${script}"
}

echo ""
echo "Submitting data-generation jobs..."

# OOD low digits (inputs 1-3 sig digits)
GEN_SME_OOD_LOW="$(submit_job "gen_sme_${TAG}_oodlow" run_generate_data_sme_5dig.slurm \
  "N_TRAIN=${N_TRAIN}" "N_VAL=${N_VAL}" "MAX_LEN=${MAX_LEN}" "NUMBER_RANGE=${ID_RANGE}" \
  "REASONING_WEIGHT=1" "NUMERIC_WEIGHT=2" \
  "SIG_DIGITS_MIN=${OOD_LOW_SIG_MIN}" "SIG_DIGITS_MAX=${OOD_LOW_SIG_MAX}" \
  "OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX}" \
  "SME_MIN_DIGITS=${SME_MIN_DIGITS}" \
  "OUT_DIR=${SME_OOD_LOW_DIR}")"
GEN_BASE_OOD_LOW="$(submit_job "gen_base_${TAG}_oodlow" run_generate_data_base_5dig.slurm \
  "N_TRAIN=${N_TRAIN}" "N_VAL=${N_VAL}" "MAX_LEN=${MAX_LEN}" "NUMBER_RANGE=${ID_RANGE}" \
  "REASONING_WEIGHT=1" "NUMERIC_WEIGHT=2" \
  "SIG_DIGITS_MIN=${OOD_LOW_SIG_MIN}" "SIG_DIGITS_MAX=${OOD_LOW_SIG_MAX}" \
  "OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX}" \
  "OUT_DIR=${BASE_OOD_LOW_DIR}")"

# OOD high digits (inputs 5-8 sig digits)
GEN_SME_OOD_HIGH="$(submit_job "gen_sme_${TAG}_oodhigh" run_generate_data_sme_5dig.slurm \
  "N_TRAIN=${N_TRAIN}" "N_VAL=${N_VAL}" "MAX_LEN=${MAX_LEN}" "NUMBER_RANGE=${ID_RANGE}" \
  "REASONING_WEIGHT=1" "NUMERIC_WEIGHT=2" \
  "SIG_DIGITS_MIN=${OOD_HIGH_SIG_MIN}" "SIG_DIGITS_MAX=${OOD_HIGH_SIG_MAX}" \
  "OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX}" \
  "SME_MIN_DIGITS=${SME_MIN_DIGITS}" \
  "OUT_DIR=${SME_OOD_HIGH_DIR}")"
GEN_BASE_OOD_HIGH="$(submit_job "gen_base_${TAG}_oodhigh" run_generate_data_base_5dig.slurm \
  "N_TRAIN=${N_TRAIN}" "N_VAL=${N_VAL}" "MAX_LEN=${MAX_LEN}" "NUMBER_RANGE=${ID_RANGE}" \
  "REASONING_WEIGHT=1" "NUMERIC_WEIGHT=2" \
  "SIG_DIGITS_MIN=${OOD_HIGH_SIG_MIN}" "SIG_DIGITS_MAX=${OOD_HIGH_SIG_MAX}" \
  "OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX}" \
  "OUT_DIR=${BASE_OOD_HIGH_DIR}")"

# OOD magnitude (inputs range 1e8, still 4-digit inputs)
GEN_SME_OOD_MAG="$(submit_job "gen_sme_${TAG}_oodmag" run_generate_data_sme_5dig.slurm \
  "N_TRAIN=${N_TRAIN}" "N_VAL=${N_VAL}" "MAX_LEN=${MAX_LEN}" "NUMBER_RANGE=${OOD_MAG_RANGE}" \
  "REASONING_WEIGHT=1" "NUMERIC_WEIGHT=2" \
  "SIG_DIGITS_MIN=4" "SIG_DIGITS_MAX=4" \
  "OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX}" \
  "SME_MIN_DIGITS=${SME_MIN_DIGITS}" \
  "OUT_DIR=${SME_OOD_MAG_DIR}")"
GEN_BASE_OOD_MAG="$(submit_job "gen_base_${TAG}_oodmag" run_generate_data_base_5dig.slurm \
  "N_TRAIN=${N_TRAIN}" "N_VAL=${N_VAL}" "MAX_LEN=${MAX_LEN}" "NUMBER_RANGE=${OOD_MAG_RANGE}" \
  "REASONING_WEIGHT=1" "NUMERIC_WEIGHT=2" \
  "SIG_DIGITS_MIN=4" "SIG_DIGITS_MAX=4" \
  "OUTPUT_SIG_DIGITS_MAX=${OUT_SIG_DIGITS_MAX}" \
  "OUT_DIR=${BASE_OOD_MAG_DIR}")"

echo "  GEN SME OOD LOW:   ${GEN_SME_OOD_LOW}"
echo "  GEN BASE OOD LOW:  ${GEN_BASE_OOD_LOW}"
echo "  GEN SME OOD HIGH:  ${GEN_SME_OOD_HIGH}"
echo "  GEN BASE OOD HIGH: ${GEN_BASE_OOD_HIGH}"
echo "  GEN SME OOD MAG:   ${GEN_SME_OOD_MAG}"
echo "  GEN BASE OOD MAG:  ${GEN_BASE_OOD_MAG}"

echo ""
echo "Submitting validation jobs (dependent on data gen)..."

VAL_SME_OOD_LOW="$(sbatch --parsable \
  --dependency=afterok:${GEN_SME_OOD_LOW} \
  --job-name="val_sme_${TAG}_oodlow" \
  --output="slurm_logs/val_sme_ood_d1_3_${TAG}_%j.log" \
  --error="slurm_logs/val_sme_ood_d1_3_${TAG}_%j.log" \
  --export=ALL,CKPT_PATH=${SME_CKPT},DATA_DIR=${SME_OOD_LOW_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_ood_d1_3_${TAG}.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_d1_3_${TAG}.json,SUM_GEN_COUNT=${SUM_GEN_COUNT_OOD},BATCH_BLOCKS=${BATCH_BLOCKS},TOP_K_ERRORS=${TOP_K_ERRORS},MAX_BLOCKS=${MAX_BLOCKS} \
  run_validate_v9.slurm)"

VAL_SME_OOD_HIGH="$(sbatch --parsable \
  --dependency=afterok:${GEN_SME_OOD_HIGH} \
  --job-name="val_sme_${TAG}_oodhigh" \
  --output="slurm_logs/val_sme_ood_d5_8_${TAG}_%j.log" \
  --error="slurm_logs/val_sme_ood_d5_8_${TAG}_%j.log" \
  --export=ALL,CKPT_PATH=${SME_CKPT},DATA_DIR=${SME_OOD_HIGH_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_ood_d5_8_${TAG}.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_d5_8_${TAG}.json,SUM_GEN_COUNT=${SUM_GEN_COUNT_OOD},BATCH_BLOCKS=${BATCH_BLOCKS},TOP_K_ERRORS=${TOP_K_ERRORS},MAX_BLOCKS=${MAX_BLOCKS} \
  run_validate_v9.slurm)"

VAL_SME_OOD_MAG="$(sbatch --parsable \
  --dependency=afterok:${GEN_SME_OOD_MAG} \
  --job-name="val_sme_${TAG}_oodmag" \
  --output="slurm_logs/val_sme_ood_mag_${TAG}_%j.log" \
  --error="slurm_logs/val_sme_ood_mag_${TAG}_%j.log" \
  --export=ALL,CKPT_PATH=${SME_CKPT},DATA_DIR=${SME_OOD_MAG_DIR},STANDARD_JSON=${SME_OUT_DIR}/val_ood_mag_${TAG}.json,EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_mag_${TAG}.json,SUM_GEN_COUNT=${SUM_GEN_COUNT_OOD},BATCH_BLOCKS=${BATCH_BLOCKS},TOP_K_ERRORS=${TOP_K_ERRORS},MAX_BLOCKS=${MAX_BLOCKS} \
  run_validate_v9.slurm)"

VAL_BASE_OOD_LOW="$(sbatch --parsable \
  --dependency=afterok:${GEN_BASE_OOD_LOW} \
  --job-name="val_base_${TAG}_oodlow" \
  --output="slurm_logs/val_base_ood_d1_3_${TAG}_%j.log" \
  --error="slurm_logs/val_base_ood_d1_3_${TAG}_%j.log" \
  --export=ALL,CKPT_PATH=${BASE_CKPT},DATA_DIR=${BASE_OOD_LOW_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_d1_3_${TAG}.json \
  run_base_validate_5dig.slurm)"

VAL_BASE_OOD_HIGH="$(sbatch --parsable \
  --dependency=afterok:${GEN_BASE_OOD_HIGH} \
  --job-name="val_base_${TAG}_oodhigh" \
  --output="slurm_logs/val_base_ood_d5_8_${TAG}_%j.log" \
  --error="slurm_logs/val_base_ood_d5_8_${TAG}_%j.log" \
  --export=ALL,CKPT_PATH=${BASE_CKPT},DATA_DIR=${BASE_OOD_HIGH_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_d5_8_${TAG}.json \
  run_base_validate_5dig.slurm)"

VAL_BASE_OOD_MAG="$(sbatch --parsable \
  --dependency=afterok:${GEN_BASE_OOD_MAG} \
  --job-name="val_base_${TAG}_oodmag" \
  --output="slurm_logs/val_base_ood_mag_${TAG}_%j.log" \
  --error="slurm_logs/val_base_ood_mag_${TAG}_%j.log" \
  --export=ALL,CKPT_PATH=${BASE_CKPT},DATA_DIR=${BASE_OOD_MAG_DIR},OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_mag_${TAG}.json \
  run_base_validate_5dig.slurm)"

echo "  VAL SME OOD LOW:   ${VAL_SME_OOD_LOW}"
echo "  VAL SME OOD HIGH:  ${VAL_SME_OOD_HIGH}"
echo "  VAL SME OOD MAG:   ${VAL_SME_OOD_MAG}"
echo "  VAL BASE OOD LOW:  ${VAL_BASE_OOD_LOW}"
echo "  VAL BASE OOD HIGH: ${VAL_BASE_OOD_HIGH}"
echo "  VAL BASE OOD MAG:  ${VAL_BASE_OOD_MAG}"

echo ""
echo "Datasets will be in:"
echo "  ${SME_OOD_LOW_DIR}"
echo "  ${SME_OOD_HIGH_DIR}"
echo "  ${SME_OOD_MAG_DIR}"
echo "  ${BASE_OOD_LOW_DIR}"
echo "  ${BASE_OOD_HIGH_DIR}"
echo "  ${BASE_OOD_MAG_DIR}"

echo ""
echo "Track jobs with:"
echo "  squeue -j ${GEN_SME_OOD_LOW},${GEN_BASE_OOD_LOW},${GEN_SME_OOD_HIGH},${GEN_BASE_OOD_HIGH},${GEN_SME_OOD_MAG},${GEN_BASE_OOD_MAG},${VAL_SME_OOD_LOW},${VAL_SME_OOD_HIGH},${VAL_SME_OOD_MAG},${VAL_BASE_OOD_LOW},${VAL_BASE_OOD_HIGH},${VAL_BASE_OOD_MAG}"

