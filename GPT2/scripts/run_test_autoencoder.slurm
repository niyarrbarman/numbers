#!/bin/bash
#SBATCH -J test-ae
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=00:30:00
#SBATCH -o slurm_logs/test_autoencoder_%j.log
#SBATCH -e slurm_logs/test_autoencoder_%j.log

mkdir -p slurm_logs
set -euo pipefail

IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/fe"
CHECKPOINT="/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v8_500k_model.pt"

module load gnu/11.2.0

echo "=========================================="
echo "Autoencoder Test: encoder → adapter → num_head → number"
echo "=========================================="

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 ${SCRIPT_DIR}/test_autoencoder.py \
    --checkpoint "${CHECKPOINT}" \
    --num-range 1000 \
    --mode both \
    --steps 5000 \
    --n-embd 256

echo "=========================================="
echo "Test complete"
echo "=========================================="
