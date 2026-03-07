#!/bin/bash
#SBATCH -J val-multipos
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/validate_multipos_%j.log
#SBATCH -e slurm_logs/validate_multipos_%j.log

mkdir -p slurm_logs
set -euo pipefail

IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/fe_multipos"

CKPT_PATH="${CKPT_PATH:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig_multipos/ckpt_best.pt}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig_multipos}"

BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
TOP_K_ERRORS="${TOP_K_ERRORS:-25}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig_multipos/val_report_best.json}"

module load gnu/11.2.0

echo "=========================================="
echo "Multi-Position (k=5) Checkpoint Validation"
echo "  Checkpoint:   ${CKPT_PATH}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Batch blocks: ${BATCH_BLOCKS}"
echo "  Top-K errors: ${TOP_K_ERRORS}"
echo "  Max blocks:   ${MAX_BLOCKS} (0 = full val set)"
echo "  JSON output:  ${OUTPUT_JSON}"
echo "=========================================="

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 ${SCRIPT_DIR}/validate_checkpoint.py \
    --ckpt "${CKPT_PATH}" \
    --data-dir "${DATA_DIR}" \
    --batch-blocks "${BATCH_BLOCKS}" \
    --top-k-errors "${TOP_K_ERRORS}" \
    --max-blocks "${MAX_BLOCKS}" \
    --output-json "${OUTPUT_JSON}"

status=$?
echo "=========================================="
echo "Validation finished with status ${status}"
echo "=========================================="
exit ${status}
