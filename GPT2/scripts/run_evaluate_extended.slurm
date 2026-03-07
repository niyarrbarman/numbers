#!/bin/bash
#SBATCH -J eval-extended
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH -o slurm_logs/eval_extended_%j.log
#SBATCH -e slurm_logs/eval_extended_%j.log

mkdir -p slurm_logs
set -euo pipefail

IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
SCRIPT="/work/m24047/m24047brmn/numbers/GPT2/evaluate_extended.py"

# --- Model checkpoints ---
BASE_CKPT="${BASE_CKPT:-/tmpdir/m24047brmn/numbers/model_checkpoints/base_5dig_5m/ckpt_best.pt}"
UNFREEZE_CKPT="${UNFREEZE_CKPT:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig_unfreeze/ckpt_best.pt}"
TEXTDEC_CKPT="${TEXTDEC_CKPT:-/tmpdir/m24047brmn/numbers/model_checkpoints/textdec_5dig/ckpt_best.pt}"

# --- Data directories ---
BASE_DATA="${BASE_DATA:-/tmpdir/m24047brmn/numbers/data/numtasks_base_5dig_5m}"
UNFREEZE_DATA="${UNFREEZE_DATA:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig_5m}"
TEXTDEC_DATA="${TEXTDEC_DATA:-/tmpdir/m24047brmn/numbers/data/numtasks_textdec_5dig}"

# --- Options ---
BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
SUM_GEN_COUNT="${SUM_GEN_COUNT:-200}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmpdir/m24047brmn/numbers/model_checkpoints/eval_extended_${SLURM_JOB_ID}.json}"

module load gnu/11.2.0

echo "=========================================="
echo "Extended Evaluation (3 models)"
echo "  Base ckpt:      ${BASE_CKPT}"
echo "  Unfreeze ckpt:  ${UNFREEZE_CKPT}"
echo "  TextDec ckpt:   ${TEXTDEC_CKPT}"
echo "  Base data:      ${BASE_DATA}"
echo "  Unfreeze data:  ${UNFREEZE_DATA}"
echo "  TextDec data:   ${TEXTDEC_DATA}"
echo "  Batch blocks:   ${BATCH_BLOCKS}"
echo "  Max blocks:     ${MAX_BLOCKS} (0 = full val set)"
echo "  SUM gen count:  ${SUM_GEN_COUNT} per length"
echo "  JSON output:    ${OUTPUT_JSON}"
echo "=========================================="

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${SCRIPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --base-data "${BASE_DATA}" \
    --unfreeze-ckpt "${UNFREEZE_CKPT}" \
    --unfreeze-data "${UNFREEZE_DATA}" \
    --textdec-ckpt "${TEXTDEC_CKPT}" \
    --textdec-data "${TEXTDEC_DATA}" \
    --batch-blocks "${BATCH_BLOCKS}" \
    --max-blocks "${MAX_BLOCKS}" \
    --sum-gen-count "${SUM_GEN_COUNT}" \
    --output-json "${OUTPUT_JSON}"

status=$?
echo "=========================================="
echo "Extended evaluation finished with status ${status}"
echo "=========================================="
exit ${status}
