#!/bin/bash
# Submit FE-v9 error breakdown job via sbatch.
#
# Usage:
#   bash run_error_breakdown_v9.sh
#
# Optional overrides:
#   CKPT_PATH=... DATA_DIR=... OUTPUT_JSON=... BASE_EXACT_RATE=... bash run_error_breakdown_v9.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_error_breakdown_v9.slurm"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "ERROR: missing script: ${SLURM_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs"

echo "Submitting FE-v9 error breakdown..."
echo "  Script: ${SLURM_SCRIPT}"
if [[ -n "${CKPT_PATH:-}" ]]; then
  echo "  CKPT_PATH=${CKPT_PATH}"
fi
if [[ -n "${DATA_DIR:-}" ]]; then
  echo "  DATA_DIR=${DATA_DIR}"
fi
if [[ -n "${OUTPUT_JSON:-}" ]]; then
  echo "  OUTPUT_JSON=${OUTPUT_JSON}"
fi
if [[ -n "${BASE_EXACT_RATE:-}" ]]; then
  echo "  BASE_EXACT_RATE=${BASE_EXACT_RATE}"
fi
echo

JOB_ID="$(sbatch --parsable --export=ALL "${SLURM_SCRIPT}")"
echo "Submitted job: ${JOB_ID}"
echo "Track with: squeue -j ${JOB_ID}"
