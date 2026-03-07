#!/bin/bash
# Generate one curriculum phase dataset for FE SME training.
#
# Usage examples:
#   sbatch run_generate_data_sme_phase.slurm
#   PHASE=2 sbatch run_generate_data_sme_phase.slurm
#   PHASE=3 N_TRAIN=2000000 sbatch run_generate_data_sme_phase.slurm
#
# Phase mapping:
#   1 -> sig digits 1-4
#   2 -> sig digits 5-8
#   3 -> sig digits 1-15

#SBATCH -J gen-numtasks-sme-phase
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm_logs/generate_data_sme_phase_%j.log
#SBATCH -e slurm_logs/generate_data_sme_phase_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPT2_DIR="${REPO_DIR}"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

PHASE="${PHASE:-1}"  # 1,2,3
N_TRAIN="${N_TRAIN:-1666667}"
N_VAL="${N_VAL:-50000}"
MAX_LEN="${MAX_LEN:-10}"
NUMBER_RANGE="${NUMBER_RANGE:-1000000000}"
REASONING_WEIGHT="${REASONING_WEIGHT:-1}"
NUMERIC_WEIGHT="${NUMERIC_WEIGHT:-2}"

BASE_OUT_DIR="${BASE_OUT_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_vardig_e9_numw2_curr_stage_v1}"
OUT_DIR="${OUT_DIR:-}"
SIG_DIGITS_MIN="${SIG_DIGITS_MIN:-}"
SIG_DIGITS_MAX="${SIG_DIGITS_MAX:-}"
SME_MIN_DIGITS="${SME_MIN_DIGITS:-}"  # optional: pad SME outputs to at least this many digits

case "${PHASE}" in
  1)
    phase_tag="p1_d1_4"
    default_sig_min=1
    default_sig_max=4
    ;;
  2)
    phase_tag="p2_d5_8"
    default_sig_min=5
    default_sig_max=8
    ;;
  3)
    phase_tag="p3_d1_15"
    default_sig_min=1
    default_sig_max=15
    ;;
  *)
    echo "ERROR: PHASE must be 1, 2, or 3 (got: ${PHASE})" >&2
    exit 2
    ;;
esac

if [[ -z "${SIG_DIGITS_MIN}" ]]; then
  SIG_DIGITS_MIN="${default_sig_min}"
fi
if [[ -z "${SIG_DIGITS_MAX}" ]]; then
  SIG_DIGITS_MAX="${default_sig_max}"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${BASE_OUT_DIR}_${phase_tag}"
fi

mkdir -p "${OUT_DIR}"

module load gnu/11.2.0

echo "=========================================="
echo "Generating SME curriculum phase ${PHASE} -> ${OUT_DIR}"
echo "  SIG_DIGITS=${SIG_DIGITS_MIN}-${SIG_DIGITS_MAX}"
echo "  SME_MIN_DIGITS=${SME_MIN_DIGITS:-<unset>}"
echo "  NUMBER_RANGE=${NUMBER_RANGE}"
echo "  MAX_LEN=${MAX_LEN}"
echo "  REASONING_WEIGHT=${REASONING_WEIGHT}"
echo "  NUMERIC_WEIGHT=${NUMERIC_WEIGHT}"
echo "=========================================="

apptainer exec \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    GPT2_DIR="${GPT2_DIR}" \
    N_TRAIN="${N_TRAIN}" \
    N_VAL="${N_VAL}" \
    MAX_LEN="${MAX_LEN}" \
    NUMBER_RANGE="${NUMBER_RANGE}" \
    REASONING_WEIGHT="${REASONING_WEIGHT}" \
    NUMERIC_WEIGHT="${NUMERIC_WEIGHT}" \
    SIG_DIGITS_MIN="${SIG_DIGITS_MIN}" \
    SIG_DIGITS_MAX="${SIG_DIGITS_MAX}" \
    SME_MIN_DIGITS="${SME_MIN_DIGITS}" \
    OUT_DIR="${OUT_DIR}" \
  bash -c '
    set -euo pipefail
    cd "${GPT2_DIR}"
    python3 -c "import tiktoken; print(\"tiktoken OK\")"
    SME_MIN_DIGITS_FLAG=""
    if [[ -n "${SME_MIN_DIGITS}" ]]; then
      SME_MIN_DIGITS_FLAG="--sme-min-digits ${SME_MIN_DIGITS}"
    fi
    python3 generate_data.py \
      --n-train "${N_TRAIN}" \
      --n-val "${N_VAL}" \
      --max-len "${MAX_LEN}" \
      --number-range "${NUMBER_RANGE}" \
      --reasoning-weight "${REASONING_WEIGHT}" \
      --numeric-weight "${NUMERIC_WEIGHT}" \
      --block-size 256 \
      --allow-negative \
      --allow-float \
      --no-digit-curriculum \
      --sig-digits-min "${SIG_DIGITS_MIN}" \
      --sig-digits-max "${SIG_DIGITS_MAX}" \
      ${SME_MIN_DIGITS_FLAG} \
      --out-dir "${OUT_DIR}"
  '
