#!/bin/bash
# Submit full FE-SME staged curriculum pipeline with SLURM dependencies.
#
# Pipeline:
#   G1 -> G2 -> G3
#   T1 (after G1)
#   T2 (after T1 and G2)
#   T3 (after T2 and G3)
#   V3 (after T3, optional)
#
# Usage:
#   bash submit_sme_curriculum_pipeline.sh
#
# Optional env overrides (propagated to child jobs via --export=ALL):
#   SUBMIT_VALIDATE=0|1
#   GEN_SCRIPT, TRAIN_SCRIPT, VAL_SCRIPT
#   and any vars consumed by phase scripts (N_TRAIN, NUMBER_RANGE, P1_MAX_ITERS, ...)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEN_SCRIPT="${GEN_SCRIPT:-${SCRIPT_DIR}/run_generate_data_sme_phase.slurm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/run_fe_train_sme_phase.slurm}"
VAL_SCRIPT="${VAL_SCRIPT:-${SCRIPT_DIR}/run_sme_validate_phase3.slurm}"
SUBMIT_VALIDATE="${SUBMIT_VALIDATE:-1}"

if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "ERROR: missing generator script: ${GEN_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: missing training script: ${TRAIN_SCRIPT}" >&2
  exit 2
fi
if [[ "${SUBMIT_VALIDATE}" == "1" && ! -f "${VAL_SCRIPT}" ]]; then
  echo "ERROR: missing validation script: ${VAL_SCRIPT}" >&2
  exit 2
fi

submit_job() {
  local deps="$1"
  local exports="$2"
  local script="$3"

  local dep_args=()
  if [[ -n "${deps}" ]]; then
    dep_args=(--dependency="afterok:${deps}")
  fi

  local job_raw
  if [[ -n "${exports}" ]]; then
    job_raw="$(sbatch --parsable "${dep_args[@]}" --export="ALL,${exports}" "${script}")"
  else
    job_raw="$(sbatch --parsable "${dep_args[@]}" --export="ALL" "${script}")"
  fi
  echo "${job_raw%%;*}"
}

echo "Submitting SME curriculum pipeline..."
echo "  GEN script:     ${GEN_SCRIPT}"
echo "  TRAIN script:   ${TRAIN_SCRIPT}"
if [[ "${SUBMIT_VALIDATE}" == "1" ]]; then
  echo "  VALIDATE script:${VAL_SCRIPT}"
else
  echo "  VALIDATE script:disabled"
fi
echo

# Generate phase datasets
G1="$(submit_job "" "PHASE=1" "${GEN_SCRIPT}")"
G2="$(submit_job "${G1}" "PHASE=2" "${GEN_SCRIPT}")"
G3="$(submit_job "${G2}" "PHASE=3" "${GEN_SCRIPT}")"

# Train staged model
T1="$(submit_job "${G1}" "PHASE=1" "${TRAIN_SCRIPT}")"
T2="$(submit_job "${T1}:${G2}" "PHASE=2" "${TRAIN_SCRIPT}")"
T3="$(submit_job "${T2}:${G3}" "PHASE=3" "${TRAIN_SCRIPT}")"

V3=""
if [[ "${SUBMIT_VALIDATE}" == "1" ]]; then
  V3="$(submit_job "${T3}" "" "${VAL_SCRIPT}")"
fi

echo "Submitted job IDs:"
echo "  G1=${G1} (generate phase 1)"
echo "  G2=${G2} (generate phase 2, after G1)"
echo "  G3=${G3} (generate phase 3, after G2)"
echo "  T1=${T1} (train phase 1, after G1)"
echo "  T2=${T2} (train phase 2, after T1 and G2)"
echo "  T3=${T3} (train phase 3, after T2 and G3)"
if [[ -n "${V3}" ]]; then
  echo "  V3=${V3} (validate phase 3, after T3)"
fi

TRACK_IDS="${G1},${G2},${G3},${T1},${T2},${T3}"
if [[ -n "${V3}" ]]; then
  TRACK_IDS="${TRACK_IDS},${V3}"
fi
echo
echo "Track with:"
echo "  squeue -j ${TRACK_IDS}"

