#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V9_DIR="${PROJECT_DIR}/fe_v9"

CKPT_PATH="${CKPT_PATH:-/tmpdir/m24047brmn/numbers/model_checkpoints/sme_5dig_v9_78772/ckpt_best.pt}"
DATA_DIR="${DATA_DIR:-/tmpdir/m24047brmn/numbers/data/numtasks_sme_5dig_5m}"
DEVICE="${DEVICE:-cuda}"
BATCH_BLOCKS="${BATCH_BLOCKS:-64}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-5}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
BASE_EXACT_RATE="${BASE_EXACT_RATE:-}"
IMAGE="${IMAGE:-/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif}"
USE_APPTAINER="${USE_APPTAINER:-auto}"

echo "=========================================="
echo "FE-v9 Error Breakdown"
echo "  Checkpoint:   ${CKPT_PATH}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Device:       ${DEVICE}"
echo "  Batch blocks: ${BATCH_BLOCKS}"
echo "  Max blocks:   ${MAX_BLOCKS} (0 = full val set)"
echo "  Examples:     ${SHOW_EXAMPLES} per category"
if [[ -n "${OUTPUT_JSON}" ]]; then
  echo "  Output JSON:  ${OUTPUT_JSON}"
else
  echo "  Output JSON:  (disabled)"
fi
if [[ -n "${BASE_EXACT_RATE}" ]]; then
  echo "  Base exact:   ${BASE_EXACT_RATE}"
fi
echo "  Runtime:      ${USE_APPTAINER}"
echo "=========================================="

cmd=(
  python3 "${V9_DIR}/error_breakdown.py"
  --ckpt "${CKPT_PATH}"
  --data-dir "${DATA_DIR}"
  --device "${DEVICE}"
  --batch-blocks "${BATCH_BLOCKS}"
  --max-blocks "${MAX_BLOCKS}"
  --show-examples "${SHOW_EXAMPLES}"
)

if [[ -n "${OUTPUT_JSON}" ]]; then
  cmd+=(--output-json "${OUTPUT_JSON}")
fi
if [[ -n "${BASE_EXACT_RATE}" ]]; then
  cmd+=(--base-exact-rate "${BASE_EXACT_RATE}")
fi

if [[ "${USE_APPTAINER}" == "auto" ]]; then
  if command -v apptainer >/dev/null 2>&1 && [[ -f "${IMAGE}" ]]; then
    USE_APPTAINER="1"
  else
    USE_APPTAINER="0"
  fi
fi

if [[ "${USE_APPTAINER}" == "1" ]]; then
  apptainer exec \
    --nv \
    --env "PYTHONUSERBASE=${MYENVS:-}" \
    --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
    --bind /tmpdir,/work \
    "${IMAGE}" \
    "${cmd[@]}"
else
  "${cmd[@]}"
fi
