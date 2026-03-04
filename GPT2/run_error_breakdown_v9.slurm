#!/bin/bash
# FE-v9 detailed error breakdown:
#   - invalid SME decode
#   - correct mantissa, wrong exponent
#   - valid but wrong digit
#
# Usage:
#   sbatch run_error_breakdown_v9.slurm

#SBATCH -J err-v9
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/error_breakdown_v9_%j.log
#SBATCH -e slurm_logs/error_breakdown_v9_%j.log

mkdir -p slurm_logs
set -euo pipefail

IMAGE="${IMAGE:-/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif}"
PROJECT_DIR="${PROJECT_DIR:-/work/m24047/m24047brmn/numbers/GPT2}"
V9_DIR="${PROJECT_DIR}/fe_v9"

# Defaults target your current 4-digit ID run. Override via --export or env.
CKPT_PATH="${CKPT_PATH:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_v9_4dig_r10k_20260303_205237/ckpt_best.pt}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_4dig_r10k_20260303_205237_id_d4}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_v9_4dig_r10k_20260303_205237/error_breakdown_id_d4.json}"

# Optional: set the matching Base exact rate to get point-gap decomposition.
BASE_EXACT_RATE="${BASE_EXACT_RATE:-0.9233}"

DEVICE="${DEVICE:-cuda}"
BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-5}"

module load gnu/11.2.0

echo "=========================================="
echo "FE-v9 Detailed Error Breakdown"
echo "  Checkpoint:     ${CKPT_PATH}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Output JSON:    ${OUTPUT_JSON}"
echo "  Base exact:     ${BASE_EXACT_RATE:-N/A}"
echo "  Device:         ${DEVICE}"
echo "  Batch blocks:   ${BATCH_BLOCKS}"
echo "  Max blocks:     ${MAX_BLOCKS} (0 = full val set)"
echo "  Show examples:  ${SHOW_EXAMPLES}"
echo "=========================================="

cmd=(
  python3 "${V9_DIR}/error_breakdown.py"
  --ckpt "${CKPT_PATH}"
  --data-dir "${DATA_DIR}"
  --device "${DEVICE}"
  --batch-blocks "${BATCH_BLOCKS}"
  --max-blocks "${MAX_BLOCKS}"
  --show-examples "${SHOW_EXAMPLES}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${BASE_EXACT_RATE}" ]]; then
  cmd+=(--base-exact-rate "${BASE_EXACT_RATE}")
fi

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  "${cmd[@]}"

status=$?
echo "=========================================="
echo "FE-v9 error breakdown finished with status ${status}"
echo "=========================================="
exit ${status}
