#!/bin/bash
#SBATCH -J convert_luciole
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm

# Convert NeMo checkpoint to simple PyTorch state dict
NEMO_CKPT="/tmpdir/m24047brmn/nemo_1b/output/baby_luciole-softmax-test/checkpoints/baby_luciole-softmax-test-step=0020998-last"
OUTPUT="/tmpdir/m24047brmn/numbers/checkpoints/baby_luciole_converted.pt"

echo "=========================================="
echo "Converting Baby Luciole NeMo checkpoint"
echo "  Input:  $NEMO_CKPT"
echo "  Output: $OUTPUT"
echo "=========================================="

mkdir -p $(dirname "$OUTPUT")

srun apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/nemo" \
    --bind /tmpdir,/work /work/conteneurs/calmip/nemo_25.04.03_arm.sif \
    python /work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt/convert_nemo_ckpt.py \
        --nemo-ckpt "$NEMO_CKPT" \
        --output "$OUTPUT"

status=$?
echo "=========================================="
echo "Conversion finished with status $status"
echo "=========================================="
exit $status
