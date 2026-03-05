#!/bin/bash
# Generate 5-digit SME dataset (no curriculum, single-phase).
#
# Usage:
#   sbatch run_generate_data_sme_5dig.slurm
#   N_TRAIN=2000000 sbatch run_generate_data_sme_5dig.slurm

#SBATCH -J gen-sme-5dig
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm_logs/generate_data_sme_5dig_%j.log
#SBATCH -e slurm_logs/generate_data_sme_5dig_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPT2_DIR="${REPO_DIR}"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

N_TRAIN="${N_TRAIN:-1000000}"
N_VAL="${N_VAL:-5000}"
MAX_LEN="${MAX_LEN:-10}"
NUMBER_RANGE="${NUMBER_RANGE:-100000}"
REASONING_WEIGHT="${REASONING_WEIGHT:-1}"
NUMERIC_WEIGHT="${NUMERIC_WEIGHT:-2}"
SIG_DIGITS_MIN="${SIG_DIGITS_MIN:-1}"
SIG_DIGITS_MAX="${SIG_DIGITS_MAX:-5}"
OUTPUT_SIG_DIGITS_MAX="${OUTPUT_SIG_DIGITS_MAX:-}"  # optional: decouple output precision from input SIG_DIGITS_*
SME_MIN_DIGITS="${SME_MIN_DIGITS:-}"  # optional: pad SME outputs to at least this many digits
OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig}"

mkdir -p "${OUT_DIR}"

module load gnu/11.2.0

echo "=========================================="
echo "Generating 5-digit SME dataset -> ${OUT_DIR}"
echo "  NUMBER_RANGE=${NUMBER_RANGE}"
echo "  SIG_DIGITS=${SIG_DIGITS_MIN}-${SIG_DIGITS_MAX}"
echo "  OUTPUT_SIG_DIGITS_MAX=${OUTPUT_SIG_DIGITS_MAX:-<unset>}"
echo "  SME_MIN_DIGITS=${SME_MIN_DIGITS:-<unset>}"
echo "  N_TRAIN=${N_TRAIN}"
echo "  N_VAL=${N_VAL}"
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
    OUTPUT_SIG_DIGITS_MAX="${OUTPUT_SIG_DIGITS_MAX}" \
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
    OUTPUT_SIG_DIGITS_MAX_FLAG=""
    if [[ -n "${OUTPUT_SIG_DIGITS_MAX}" ]]; then
      OUTPUT_SIG_DIGITS_MAX_FLAG="--output-sig-digits-max ${OUTPUT_SIG_DIGITS_MAX}"
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
      ${OUTPUT_SIG_DIGITS_MAX_FLAG} \
      ${SME_MIN_DIGITS_FLAG} \
      --out-dir "${OUT_DIR}"
  '
