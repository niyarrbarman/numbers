#!/bin/bash
#SBATCH -J analytic_gendata
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm
set -euo pipefail

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
OUT_DIR="/tmpdir/m24047brmn/numbers/data/analytic_stage1"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

echo "=========================================="
echo "Stage 1 Analytic Data Generation"
echo "  Output: $OUT_DIR"
echo "=========================================="

module load gnu/11.2.0

apptainer exec \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "PYTHONUNBUFFERED=1" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${SCRIPT_DIR}/generate_data_analytic.py" \
    --out_dir "$OUT_DIR" \
    --n_train 100000 \
    --n_val 5000 \
    --n_test 3000 \
    --seed 42

status=$?
echo "=========================================="
echo "Data generation finished with status $status"
echo "=========================================="
exit $status
