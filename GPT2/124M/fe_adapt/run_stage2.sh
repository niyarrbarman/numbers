#!/bin/bash
#SBATCH -J luciole_fe_s2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm
set -euo pipefail

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
STAGE1_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_fe_adapt/ckpt.pt"
DATA_DIR="/tmpdir/m24047brmn/numbers/data/numtasks_124M_fe"
OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_fe_adapt_s2"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

echo "=========================================="
echo "Stage 2: LoRA + Adapter Fine-Tuning"
echo "  Stage 1 checkpoint: $STAGE1_CKPT"
echo "  Data dir:           $DATA_DIR"
echo "  Output dir:         $OUT_DIR"
echo "=========================================="

module load gnu/11.2.0

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --env "PYTHONUNBUFFERED=1" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${SCRIPT_DIR}/train_stage2.py" \
    init_from=stage1 \
    stage1_ckpt="$STAGE1_CKPT" \
    data_dir="$DATA_DIR" \
    out_dir="$OUT_DIR" \
    lora_rank=16 \
    lora_alpha=32 \
    lora_dropout=0.05 \
    lora_targets=q_proj,v_proj,k_proj,o_proj \
    batch_size=4 \
    gradient_accumulation_steps=80 \
    block_size=256 \
    max_iters=30000 \
    learning_rate=3e-4 \
    lora_lr_scale=1.0 \
    adapter_lr_scale=0.3 \
    warmup_iters=1000 \
    lr_decay_iters=30000 \
    min_lr=3e-5 \
    num_norm_match=True \
    eval_interval=2000 \
    diag_interval=100 \
    sample_interval=1000 \
    log_interval=10

status=$?
echo "=========================================="
echo "Stage 2 finished with status $status"
echo "=========================================="
exit $status
