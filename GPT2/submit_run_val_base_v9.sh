#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs

# ----------------------------
# User-configurable parameters
# ----------------------------
DATA_ROOT="${DATA_ROOT:-/tmpdir/m24047brmn/numbers/data}"
MODEL_ROOT="${MODEL_ROOT:-/tmpdir/m24047brmn/numbers/model_checkpoints}"

TAG="${TAG:-4dig_r10k_20260303_205237}"

# Dataset dirs (must already exist)
SME_TRAIN_DIR="${SME_TRAIN_DIR:-${DATA_ROOT}/numtasks_sme_${TAG}_id_d4}"
BASE_TRAIN_DIR="${BASE_TRAIN_DIR:-${DATA_ROOT}/numtasks_base_${TAG}_id_d4}"

SME_OOD_LOW_DIR="${SME_OOD_LOW_DIR:-${DATA_ROOT}/numtasks_sme_${TAG}_ood_d1_3}"
BASE_OOD_LOW_DIR="${BASE_OOD_LOW_DIR:-${DATA_ROOT}/numtasks_base_${TAG}_ood_d1_3}"

SME_OOD_HIGH_DIR="${SME_OOD_HIGH_DIR:-${DATA_ROOT}/numtasks_sme_${TAG}_ood_d5_8}"
BASE_OOD_HIGH_DIR="${BASE_OOD_DIR:-${DATA_ROOT}/numtasks_base_${TAG}_ood_d5_8}"

SME_OOD_MAG_DIR="${SME_OOD_MAG_DIR:-${DATA_ROOT}/numtasks_sme_${TAG}_ood_mag}"
BASE_OOD_MAG_DIR="${BASE_OOD_MAG_DIR:-${DATA_ROOT}/numtasks_base_${TAG}_ood_mag}"

# Model output dirs (must already exist)
SME_OUT_DIR="${SME_OUT_DIR:-${MODEL_ROOT}/sme_v9_${TAG}}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${MODEL_ROOT}/base_${TAG}}"

# Checkpoints
SME_CKPT="${SME_CKPT:-${SME_OUT_DIR}/ckpt_best.pt}"
BASE_CKPT="${BASE_CKPT:-${BASE_OUT_DIR}/ckpt.pt}"

# Validate config
SUM_GEN_COUNT_ID="${SUM_GEN_COUNT_ID:-200}"
SUM_GEN_COUNT_OOD="${SUM_GEN_COUNT_OOD:-100}"

echo "=========================================="
echo "SUBMIT VALIDATIONS (via sbatch)"
echo "  TAG:         ${TAG}"
echo "  DATA_ROOT:   ${DATA_ROOT}"
echo "  MODEL_ROOT:  ${MODEL_ROOT}"
echo "  SME_CKPT:    ${SME_CKPT}"
echo "  BASE_CKPT:   ${BASE_CKPT}"
echo "=========================================="

# ----------------------------
# Sanity checks
# ----------------------------
need_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }; }
need_dir()  { [[ -d "$1" ]] || { echo "ERROR: missing dir:  $1" >&2; exit 1; }; }

need_file run_validate_v9.slurm
need_file run_base_validate_5dig.slurm

need_file "${SME_CKPT}"
need_file "${BASE_CKPT}"

need_dir "${SME_TRAIN_DIR}"
need_dir "${BASE_TRAIN_DIR}"
need_dir "${SME_OOD_LOW_DIR}"
need_dir "${BASE_OOD_LOW_DIR}"
need_dir "${SME_OOD_HIGH_DIR}"
need_dir "${BASE_OOD_HIGH_DIR}"
need_dir "${SME_OOD_MAG_DIR}"
need_dir "${BASE_OOD_MAG_DIR}"

need_dir "${SME_OUT_DIR}"
need_dir "${BASE_OUT_DIR}"

# ----------------------------
# Helper: sbatch with env
# ----------------------------
submit_job() {
  local jobname="$1"; shift
  local script="$1"; shift
  local logfile="slurm_logs/${jobname}_%j.log"

  # Build --export argument (ALL plus provided KEY=VAL pairs)
  local export_arg="ALL"
  for kv in "$@"; do
    export_arg+=",${kv}"
  done

  echo ""
  echo "---- SBATCH: ${jobname}"
  echo "     script: ${script}"
  echo "     export: ${export_arg}"

  sbatch --parsable \
    --job-name="${jobname}" \
    --output="${logfile}" \
    --error="${logfile}" \
    --export="${export_arg}" \
    "${script}"
}

jobids=()

# 1) ID
jobids+=( "$(submit_job "val_sme_id_d4_${TAG}"  run_validate_v9.slurm \
  "CKPT_PATH=${SME_CKPT}" \
  "DATA_DIR=${SME_TRAIN_DIR}" \
  "STANDARD_JSON=${SME_OUT_DIR}/val_id_d4.json" \
  "EXTENDED_JSON=${SME_OUT_DIR}/eval_id_d4.json" \
  "SUM_GEN_COUNT=${SUM_GEN_COUNT_ID}")" )

jobids+=( "$(submit_job "val_base_id_d4_${TAG}" run_base_validate_5dig.slurm \
  "CKPT_PATH=${BASE_CKPT}" \
  "DATA_DIR=${BASE_TRAIN_DIR}" \
  "OUTPUT_JSON=${BASE_OUT_DIR}/val_id_d4.json")" )

# 2) OOD low digits
jobids+=( "$(submit_job "val_sme_ood_d1_3_${TAG}"  run_validate_v9.slurm \
  "CKPT_PATH=${SME_CKPT}" \
  "DATA_DIR=${SME_OOD_LOW_DIR}" \
  "STANDARD_JSON=${SME_OUT_DIR}/val_ood_d1_3.json" \
  "EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_d1_3.json" \
  "SUM_GEN_COUNT=${SUM_GEN_COUNT_OOD}")" )

jobids+=( "$(submit_job "val_base_ood_d1_3_${TAG}" run_base_validate_5dig.slurm \
  "CKPT_PATH=${BASE_CKPT}" \
  "DATA_DIR=${BASE_OOD_LOW_DIR}" \
  "OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_d1_3.json")" )

# 3) OOD high digits
jobids+=( "$(submit_job "val_sme_ood_d5_8_${TAG}"  run_validate_v9.slurm \
  "CKPT_PATH=${SME_CKPT}" \
  "DATA_DIR=${SME_OOD_HIGH_DIR}" \
  "STANDARD_JSON=${SME_OUT_DIR}/val_ood_d5_8.json" \
  "EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_d5_8.json" \
  "SUM_GEN_COUNT=${SUM_GEN_COUNT_OOD}")" )

jobids+=( "$(submit_job "val_base_ood_d5_8_${TAG}" run_base_validate_5dig.slurm \
  "CKPT_PATH=${BASE_CKPT}" \
  "DATA_DIR=${BASE_OOD_HIGH_DIR}" \
  "OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_d5_8.json")" )

# 4) OOD magnitude
jobids+=( "$(submit_job "val_sme_ood_mag_${TAG}"  run_validate_v9.slurm \
  "CKPT_PATH=${SME_CKPT}" \
  "DATA_DIR=${SME_OOD_MAG_DIR}" \
  "STANDARD_JSON=${SME_OUT_DIR}/val_ood_mag.json" \
  "EXTENDED_JSON=${SME_OUT_DIR}/eval_ood_mag.json" \
  "SUM_GEN_COUNT=${SUM_GEN_COUNT_OOD}")" )

jobids+=( "$(submit_job "val_base_ood_mag_${TAG}" run_base_validate_5dig.slurm \
  "CKPT_PATH=${BASE_CKPT}" \
  "DATA_DIR=${BASE_OOD_MAG_DIR}" \
  "OUTPUT_JSON=${BASE_OUT_DIR}/val_ood_mag.json")" )

echo ""
echo "Submitted job IDs:"
printf "  %s\n" "${jobids[@]}"

# ----------------------------
# Optional: wait for completion
# ----------------------------
echo ""
echo "Waiting for all jobs to finish (polling squeue)..."
while true; do
  # If squeue returns nothing for all jobids, we're done
  if ! squeue -h -j "$(IFS=,; echo "${jobids[*]}")" >/dev/null 2>&1; then
    break
  fi

  # Some clusters return exit 0 even if empty; handle that too:
  if [[ -z "$(squeue -h -j "$(IFS=,; echo "${jobids[*]}")")" ]]; then
    break
  fi

  sleep 15
done

echo "=========================================="
echo "DONE: all validations finished for TAG=${TAG}"
echo " SME outputs in:  ${SME_OUT_DIR}"
echo " BASE outputs in: ${BASE_OUT_DIR}"
echo " logs in:         slurm_logs/"
echo "==========================================
