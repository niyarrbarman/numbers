#!/bin/bash
#SBATCH -J gendata_124M_fe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm
set -euo pipefail

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
OUT_DIR="/tmpdir/m24047brmn/numbers/data/numtasks_124M_fe"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

echo "=========================================="
echo "Generating training data for 124M FE adapt"
echo "  Script dir: $SCRIPT_DIR"
echo "  Output dir: $OUT_DIR"
echo "=========================================="

mkdir -p "${OUT_DIR}"

module load gnu/11.2.0

apptainer exec \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${SCRIPT_DIR}/generate_data.py" \
    --out-dir "$OUT_DIR" \
    --n-train 500000 \
    --n-val 5000

status=$?
echo "=========================================="
echo "Data generation finished with status $status"
echo "=========================================="
exit $status
