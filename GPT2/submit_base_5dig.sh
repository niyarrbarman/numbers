#!/bin/bash
# Submit 5-digit base GPT-2 pipeline: generate data → train → validate.
#
# Usage:
#   bash submit_base_5dig.sh
#   SUBMIT_VALIDATE=0 bash submit_base_5dig.sh   # skip validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEN_SCRIPT="${SCRIPT_DIR}/run_generate_data_base_5dig.slurm"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_base_train_5dig.slurm"
VAL_SCRIPT="${SCRIPT_DIR}/run_base_validate_5dig.slurm"
SUBMIT_VALIDATE="${SUBMIT_VALIDATE:-1}"

if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "ERROR: missing generator script: ${GEN_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: missing training script: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

echo "Submitting 5-digit base GPT-2 pipeline..."
echo "  Generate: ${GEN_SCRIPT}"
echo "  Train:    ${TRAIN_SCRIPT}"
if [[ "${SUBMIT_VALIDATE}" == "1" ]]; then
  echo "  Validate: ${VAL_SCRIPT}"
fi
echo

# Submit data generation
G1="$(sbatch --parsable --export=ALL "${GEN_SCRIPT}")"
echo "  G1=${G1} (generate 5-digit base data)"

# Submit training after data generation completes
T1="$(sbatch --parsable --dependency="afterok:${G1}" --export=ALL "${TRAIN_SCRIPT}")"
echo "  T1=${T1} (train base GPT-2, after G1)"

V1=""
if [[ "${SUBMIT_VALIDATE}" == "1" && -f "${VAL_SCRIPT}" ]]; then
  V1="$(sbatch --parsable --dependency="afterok:${T1}" --export=ALL "${VAL_SCRIPT}")"
  echo "  V1=${V1} (validate base, after T1)"
fi

TRACK="${G1},${T1}"
if [[ -n "${V1}" ]]; then
  TRACK="${TRACK},${V1}"
fi

echo
echo "Track with:"
echo "  squeue -j ${TRACK}"
