#!/bin/bash
# Validate FE-v9 checkpoint with:
#   1) Standard SME checkpoint validation
#   2) Extended evaluation (conditional MAE, difficulty buckets, SUM generalization)
#
# Usage:
#   sbatch run_validate_v9.slurm

#SBATCH -J val-v9
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH -o slurm_logs/validate_v9_%j.log
#SBATCH -e slurm_logs/validate_v9_%j.log

mkdir -p slurm_logs
set -euo pipefail

IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
PROJECT_DIR="/work/m24047/m24047brmn/numbers/GPT2"
V9_DIR="${PROJECT_DIR}/fe_v9"

CKPT_PATH="${CKPT_PATH:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig_v9/ckpt_best.pt}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig_5m}"

BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
TOP_K_ERRORS="${TOP_K_ERRORS:-25}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
SUM_GEN_COUNT="${SUM_GEN_COUNT:-200}"

OUTPUT_DIR_DEFAULT="/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig_v9"
STANDARD_JSON="${STANDARD_JSON:-${OUTPUT_DIR_DEFAULT}/val_report_best.json}"
EXTENDED_JSON="${EXTENDED_JSON:-${OUTPUT_DIR_DEFAULT}/eval_extended_best.json}"

module load gnu/11.2.0

echo "=========================================="
echo "FE-v9 Validation + Extended Evaluation"
echo "  Checkpoint:      ${CKPT_PATH}"
echo "  Data dir:        ${DATA_DIR}"
echo "  Batch blocks:    ${BATCH_BLOCKS}"
echo "  Top-K errors:    ${TOP_K_ERRORS}"
echo "  Max blocks:      ${MAX_BLOCKS} (0 = full val set)"
echo "  SUM gen count:   ${SUM_GEN_COUNT} per length"
echo "  Standard JSON:   ${STANDARD_JSON}"
echo "  Extended JSON:   ${EXTENDED_JSON}"
echo "=========================================="

echo "[1/2] Running standard checkpoint validation..."
apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${V9_DIR}/validate_checkpoint.py" \
    --ckpt "${CKPT_PATH}" \
    --data-dir "${DATA_DIR}" \
    --batch-blocks "${BATCH_BLOCKS}" \
    --top-k-errors "${TOP_K_ERRORS}" \
    --max-blocks "${MAX_BLOCKS}" \
    --output-json "${STANDARD_JSON}"

echo "[2/2] Running extended evaluation (v9 only)..."
apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${PROJECT_DIR}/evaluate_extended_v9.py" \
    --ckpt "${CKPT_PATH}" \
    --data-dir "${DATA_DIR}" \
    --batch-blocks "${BATCH_BLOCKS}" \
    --max-blocks "${MAX_BLOCKS}" \
    --sum-gen-count "${SUM_GEN_COUNT}" \
    --output-json "${EXTENDED_JSON}"

status=$?
echo "=========================================="
echo "FE-v9 validation pipeline finished with status ${status}"
echo "=========================================="
exit ${status}
