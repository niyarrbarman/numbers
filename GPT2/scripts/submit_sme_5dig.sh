#!/bin/bash
# Submit 5-digit SME pipeline: generate data → train model.
#
# Usage:
#   bash submit_sme_5dig.sh
#
# Optional env overrides:
#   N_TRAIN=2000000 bash submit_sme_5dig.sh
#   MAX_ITERS=25000 bash submit_sme_5dig.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEN_SCRIPT="${SCRIPT_DIR}/run_generate_data_sme_5dig.slurm"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_fe_train_sme_5dig.slurm"

if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "ERROR: missing generator script: ${GEN_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: missing training script: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

echo "Submitting 5-digit SME pipeline..."
echo "  Generate: ${GEN_SCRIPT}"
echo "  Train:    ${TRAIN_SCRIPT}"
echo

# Submit data generation
G1="$(sbatch --parsable --export=ALL "${GEN_SCRIPT}")"
echo "  G1=${G1} (generate 5-digit data)"

# Submit training after data generation completes
T1="$(sbatch --parsable --dependency="afterok:${G1}" --export=ALL "${TRAIN_SCRIPT}")"
echo "  T1=${T1} (train on 5-digit data, after G1)"

echo
echo "Track with:"
echo "  squeue -j ${G1},${T1}"
