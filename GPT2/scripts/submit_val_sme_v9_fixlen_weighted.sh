#!/bin/bash
# Submit FE-v9 (SME) validations for the "fixlen + weighted digits (alpha=0.5)" run.
# Runs `run_validate_v9.slurm` (standard + extended) on:
#   - ID (fixlen) split
#   - OOD low digits (1-3 sig digits)
#   - OOD high digits (5-8 sig digits)
#   - OOD magnitude (larger numeric range)
#
# Defaults are wired to your fixlen-weighted checkpoint and the existing OOD datasets
# from the earlier 4dig_r10k_20260303_205237 run.
#
# Usage (on a SLURM login node):
#   bash submit_val_sme_v9_fixlen_weighted.sh
#
# Common overrides:
#   RUN_ID=0 bash submit_val_sme_v9_fixlen_weighted.sh
#   SME_CKPT=... SME_OUT_DIR=... bash submit_val_sme_v9_fixlen_weighted.sh

set -euo pipefail

mkdir -p slurm_logs

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH. Run this from a SLURM login node." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-/tmpdir/m24047brmn/numbers/data}"
MODEL_ROOT="${MODEL_ROOT:-/tmpdir/m24047brmn/numbers/model_checkpoints}"

# Fixlen + weighted run
TAG="${TAG:-v9_4dig_r10k_fixlen_wdig_a05_20260304_191943}"
SME_OUT_DIR="${SME_OUT_DIR:-${MODEL_ROOT}/sme_v9_${TAG}}"
SME_CKPT="${SME_CKPT:-${SME_OUT_DIR}/ckpt_best.pt}"

# ID dataset (fixlen outputs)
SME_ID_DIR="${SME_ID_DIR:-${DATA_ROOT}/numtasks_sme_v9_4dig_r10k_fixlen_20260304_035452_id_d4}"

# OOD datasets (defaults reuse existing OOD data from the older run tag)
OOD_TAG="${OOD_TAG:-4dig_r10k_20260303_205237}"
SME_OOD_LOW_DIR="${SME_OOD_LOW_DIR:-${DATA_ROOT}/numtasks_sme_${OOD_TAG}_ood_d1_3}"
SME_OOD_HIGH_DIR="${SME_OOD_HIGH_DIR:-${DATA_ROOT}/numtasks_sme_${OOD_TAG}_ood_d5_8}"
SME_OOD_MAG_DIR="${SME_OOD_MAG_DIR:-${DATA_ROOT}/numtasks_sme_${OOD_TAG}_ood_mag}"

# Validate knobs (forwarded to run_validate_v9.slurm)
BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
TOP_K_ERRORS="${TOP_K_ERRORS:-25}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
SUM_GEN_COUNT_ID="${SUM_GEN_COUNT_ID:-200}"
SUM_GEN_COUNT_OOD="${SUM_GEN_COUNT_OOD:-100}"

# Which splits to run
RUN_ID="${RUN_ID:-1}"
RUN_OOD_LOW="${RUN_OOD_LOW:-1}"
RUN_OOD_HIGH="${RUN_OOD_HIGH:-1}"
RUN_OOD_MAG="${RUN_OOD_MAG:-1}"

need_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }; }
need_dir()  { [[ -d "$1" ]] || { echo "ERROR: missing dir:  $1" >&2; exit 1; }; }

need_file run_validate_v9.slurm
need_file "${SME_CKPT}"
mkdir -p "${SME_OUT_DIR}"

if [[ "${RUN_ID}" == "1" ]]; then need_dir "${SME_ID_DIR}"; fi
if [[ "${RUN_OOD_LOW}" == "1" ]]; then need_dir "${SME_OOD_LOW_DIR}"; fi
if [[ "${RUN_OOD_HIGH}" == "1" ]]; then need_dir "${SME_OOD_HIGH_DIR}"; fi
if [[ "${RUN_OOD_MAG}" == "1" ]]; then need_dir "${SME_OOD_MAG_DIR}"; fi

echo "=========================================="
echo "SUBMIT FE-v9 VALIDATIONS (SME only)"
echo "  TAG:          ${TAG}"
echo "  CKPT:         ${SME_CKPT}"
echo "  OUT_DIR:      ${SME_OUT_DIR}"
echo "  ID dir:       ${SME_ID_DIR}"
echo "  OOD low dir:  ${SME_OOD_LOW_DIR}"
echo "  OOD high dir: ${SME_OOD_HIGH_DIR}"
echo "  OOD mag dir:  ${SME_OOD_MAG_DIR}"
echo "  blocks:       batch=${BATCH_BLOCKS} max=${MAX_BLOCKS}"
echo "=========================================="

submit_job() {
  local jobname="$1"; shift
  local data_dir="$1"; shift
  local standard_json="$1"; shift
  local extended_json="$1"; shift
  local sum_gen_count="$1"; shift

  local logfile="slurm_logs/${jobname}_%j.log"

  local export_arg="ALL"
  export_arg+=",CKPT_PATH=${SME_CKPT}"
  export_arg+=",DATA_DIR=${data_dir}"
  export_arg+=",STANDARD_JSON=${standard_json}"
  export_arg+=",EXTENDED_JSON=${extended_json}"
  export_arg+=",SUM_GEN_COUNT=${sum_gen_count}"
  export_arg+=",BATCH_BLOCKS=${BATCH_BLOCKS}"
  export_arg+=",TOP_K_ERRORS=${TOP_K_ERRORS}"
  export_arg+=",MAX_BLOCKS=${MAX_BLOCKS}"

  echo ""
  echo "---- SBATCH: ${jobname}"
  echo "     data:   ${data_dir}"

  sbatch --parsable \
    --job-name="${jobname}" \
    --output="${logfile}" \
    --error="${logfile}" \
    --export="${export_arg}" \
    run_validate_v9.slurm
}

jobids=()

if [[ "${RUN_ID}" == "1" ]]; then
  jobids+=( "$(submit_job "val_sme_id_d4_${TAG}" \
    "${SME_ID_DIR}" \
    "${SME_OUT_DIR}/val_id_d4.json" \
    "${SME_OUT_DIR}/eval_id_d4.json" \
    "${SUM_GEN_COUNT_ID}")" )
fi

if [[ "${RUN_OOD_LOW}" == "1" ]]; then
  jobids+=( "$(submit_job "val_sme_ood_d1_3_${TAG}" \
    "${SME_OOD_LOW_DIR}" \
    "${SME_OUT_DIR}/val_ood_d1_3.json" \
    "${SME_OUT_DIR}/eval_ood_d1_3.json" \
    "${SUM_GEN_COUNT_OOD}")" )
fi

if [[ "${RUN_OOD_HIGH}" == "1" ]]; then
  jobids+=( "$(submit_job "val_sme_ood_d5_8_${TAG}" \
    "${SME_OOD_HIGH_DIR}" \
    "${SME_OUT_DIR}/val_ood_d5_8.json" \
    "${SME_OUT_DIR}/eval_ood_d5_8.json" \
    "${SUM_GEN_COUNT_OOD}")" )
fi

if [[ "${RUN_OOD_MAG}" == "1" ]]; then
  jobids+=( "$(submit_job "val_sme_ood_mag_${TAG}" \
    "${SME_OOD_MAG_DIR}" \
    "${SME_OUT_DIR}/val_ood_mag.json" \
    "${SME_OUT_DIR}/eval_ood_mag.json" \
    "${SUM_GEN_COUNT_OOD}")" )
fi

if [[ "${#jobids[@]}" -eq 0 ]]; then
  echo "No jobs submitted (all RUN_* flags are 0)."
  exit 0
fi

echo ""
echo "Submitted job IDs:"
printf "  %s\n" "${jobids[@]}"

job_csv="$(IFS=,; echo "${jobids[*]}")"
echo ""
echo "Track with:"
echo "  squeue -j ${job_csv}"

