#!/bin/bash
#SBATCH -J gendata_124M_fe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="/tmpdir/m24047brmn/numbers/data/numtasks_124M_fe"

echo "=========================================="
echo "Generating training data for 124M FE adapt"
echo "  Script dir: $SCRIPT_DIR"
echo "  Output dir: $OUT_DIR"
echo "=========================================="

module purge
module load gnu/11.2.0

# If using conda:
# source activate torch

cd "$SCRIPT_DIR"

python generate_data.py \
    --out-dir "$OUT_DIR" \
    --n-train 500000 \
    --n-val 5000

status=$?
echo "=========================================="
echo "Data generation finished with status $status"
echo "=========================================="
exit $status
