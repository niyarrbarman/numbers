#!/bin/bash
# Generate 5-digit SME dataset with multi-position tokenization (k=5).
# Each number becomes 5 consecutive <NUM> tokens with pos_indices 0..4.
# Produces 3 binary files: {split}.bin, {split}_nums.bin, {split}_pos.bin
#
# Usage:
#   sbatch run_multipos_gen.slurm
#   N_TRAIN=2000000 sbatch run_multipos_gen.slurm

#SBATCH -J gen-multipos
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH -o slurm_logs/generate_data_multipos_%j.log
#SBATCH -e slurm_logs/generate_data_multipos_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPT2_DIR="${REPO_DIR}"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

N_TRAIN="${N_TRAIN:-5000000}"
N_VAL="${N_VAL:-10000}"
MAX_LEN="${MAX_LEN:-10}"
NUMBER_RANGE="${NUMBER_RANGE:-100000}"
REASONING_WEIGHT="${REASONING_WEIGHT:-1}"
NUMERIC_WEIGHT="${NUMERIC_WEIGHT:-2}"
SIG_DIGITS_MIN="${SIG_DIGITS_MIN:-1}"
SIG_DIGITS_MAX="${SIG_DIGITS_MAX:-5}"
OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig_multipos}"

mkdir -p "${OUT_DIR}"

module load gnu/11.2.0

echo "=========================================="
echo "Generating 5-digit SME dataset (multi-position k=5) -> ${OUT_DIR}"
echo "  NUMBER_RANGE=${NUMBER_RANGE}"
echo "  SIG_DIGITS=${SIG_DIGITS_MIN}-${SIG_DIGITS_MAX}"
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
    OUT_DIR="${OUT_DIR}" \
  bash -c '
    set -euo pipefail
    cd "${GPT2_DIR}"
    python3 -c "import tiktoken; print(\"tiktoken OK\")"
    python3 fe_multipos/generate_data.py \
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
      --out-dir "${OUT_DIR}"
  '
