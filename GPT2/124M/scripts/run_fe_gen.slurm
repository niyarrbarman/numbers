#!/bin/bash
# Generate 124M FE dataset (NUM input + text output, arithmetic tasks only).
# Tasks: ADD, SUB, SUM, MIN, MAX
#
# Usage:
#   sbatch 124M/scripts/run_fe_gen.slurm
#   N_TRAIN=5000000 sbatch 124M/scripts/run_fe_gen.slurm

#SBATCH -J gen-124M-fe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH -o slurm_logs/gen_124M_fe_%j.log
#SBATCH -e slurm_logs/gen_124M_fe_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPT2_DIR="${REPO_DIR}"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

N_TRAIN="${N_TRAIN:-50000000}"
N_VAL="${N_VAL:-500000}"
MAX_LEN="${MAX_LEN:-10}"
NUMBER_RANGE="${NUMBER_RANGE:-1000000000}"
DIGIT_CURRICULUM="${DIGIT_CURRICULUM:-1}"
OUT_DIR="${OUT_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_124M_fe}"

mkdir -p "${OUT_DIR}"

module load gnu/11.2.0

echo "=========================================="
echo "Generating 124M FE dataset -> ${OUT_DIR}"
echo "  Tasks:          ADD, SUB, SUM, MIN, MAX"
echo "  N_TRAIN:        ${N_TRAIN}"
echo "  N_VAL:          ${N_VAL}"
echo "  NUMBER_RANGE:   ${NUMBER_RANGE}"
echo "  MAX_LEN:        ${MAX_LEN}"
echo "  CURRICULUM:     ${DIGIT_CURRICULUM}"
echo "=========================================="

DIGIT_CURRICULUM_FLAG="--digit-curriculum"
if [[ "${DIGIT_CURRICULUM}" == "0" ]]; then
  DIGIT_CURRICULUM_FLAG="--no-digit-curriculum"
fi

apptainer exec \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    GPT2_DIR="${GPT2_DIR}" \
  bash -c "
    set -euo pipefail
    cd \"${GPT2_DIR}\"
    python3 -c \"import tiktoken; print('tiktoken OK')\"
    python3 124M/fe/generate_data.py \
      --n-train ${N_TRAIN} \
      --n-val ${N_VAL} \
      --max-len ${MAX_LEN} \
      --number-range ${NUMBER_RANGE} \
      --block-size 256 \
      --allow-negative \
      --allow-float \
      ${DIGIT_CURRICULUM_FLAG} \
      --out-dir \"${OUT_DIR}\"
  "

status=$?
echo "=========================================="
echo "Data generation finished with status $status"
echo "=========================================="
exit $status
