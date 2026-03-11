#!/bin/bash
#SBATCH -J analytic_train_s1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm
set -euo pipefail

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
CONVERTED_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/baby_luciole_converted.pt"
DATA_DIR="/tmpdir/m24047brmn/numbers/data/analytic_stage1"
OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_analytic_s1"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

echo "=========================================="
echo "Stage 1: Analytic Number Integration Training"
echo "  Converted checkpoint: $CONVERTED_CKPT"
echo "  Data dir:             $DATA_DIR"
echo "  Output dir:           $OUT_DIR"
echo "=========================================="

module load gnu/11.2.0

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --env "PYTHONUNBUFFERED=1" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${SCRIPT_DIR}/train_analytic.py" \
    init_from=pretrained \
    pretrained_ckpt="$CONVERTED_CKPT" \
    data_dir="$DATA_DIR" \
    out_dir="$OUT_DIR" \
    block_size=256 \
    batch_size=4 \
    gradient_accumulation_steps=40 \
    max_iters=20000 \
    learning_rate=1e-3 \
    adapter_lr_scale=1.0 \
    warmup_iters=1000 \
    lr_decay_iters=20000 \
    min_lr=1e-4 \
    num_loss_lambda=1.0 \
    eval_interval=2000 \
    diag_interval=100 \
    sample_interval=1000 \
    log_interval=10

status=$?
echo "=========================================="
echo "Training finished with status $status"
echo "=========================================="
exit $status
