#!/bin/bash
#SBATCH -J luciole_fe_adapt
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
ENCODER_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v10_2000k_model.pt"
DATA_DIR="/tmpdir/m24047brmn/numbers/data/numtasks_124M_fe"
OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

echo "=========================================="
echo "Baby Luciole + NumberEncoder Adapter Training"
echo "  Converted checkpoint: $CONVERTED_CKPT"
echo "  Encoder checkpoint:   $ENCODER_CKPT"
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
  python3 "${SCRIPT_DIR}/train.py" \
    init_from=pretrained \
    pretrained_ckpt="$CONVERTED_CKPT" \
    num_emb_checkpoint="$ENCODER_CKPT" \
    data_dir="$DATA_DIR" \
    out_dir="$OUT_DIR" \
    freeze_base=True \
    block_size=256 \
    batch_size=4 \
    gradient_accumulation_steps=80 \
    max_iters=20000 \
    learning_rate=1e-3 \
    adapter_lr_scale=1.0 \
    warmup_iters=1000 \
    lr_decay_iters=20000 \
    min_lr=1e-4 \
    num_norm_match=True \
    num_blend_warmup_iters=0 \
    num_blend_ramp_iters=10000 \
    eval_interval=2000 \
    diag_interval=100 \
    sample_interval=1000 \
    log_interval=10

status=$?
echo "=========================================="
echo "Training finished with status $status"
echo "=========================================="
exit $status
