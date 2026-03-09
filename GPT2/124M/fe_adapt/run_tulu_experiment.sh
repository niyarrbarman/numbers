#!/bin/bash
#
# Master script for tulu-3 A/B experiment.
# Submits SLURM jobs with dependency chain:
#   1. Prepare data (CPU, no GPU)
#   2a. Train base LoRA    }  parallel
#   2b. Train adapted LoRA }
#   3. Benchmark both (after 2a + 2b)
#
# Usage:
#   bash run_tulu_experiment.sh
#
set -euo pipefail
mkdir -p slurm

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Data paths
RAW_DIR="/tmpdir/m24047brmn/numbers/data/tulu3_math_grade/raw"
DATA_DIR="/tmpdir/m24047brmn/numbers/data/tulu3_math_grade"
BASE_DATA="${DATA_DIR}/base"
ADAPTED_DATA="${DATA_DIR}/adapted"

# Model paths
PRETRAINED_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/baby_luciole_converted.pt"
STAGE1_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_fe_adapt/ckpt.pt"

# Output paths
BASE_OUT="/tmpdir/m24047brmn/numbers/model_checkpoints/tulu_base_lora"
ADAPTED_OUT="/tmpdir/m24047brmn/numbers/model_checkpoints/tulu_adapted_lora"

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

echo "=========================================="
echo "Tulu-3 Math A/B Experiment"
echo "  Raw data:        ${RAW_DIR}"
echo "  Base data:       ${BASE_DATA}"
echo "  Adapted data:    ${ADAPTED_DATA}"
echo "  Pretrained ckpt: ${PRETRAINED_CKPT}"
echo "  Stage 1 ckpt:    ${STAGE1_CKPT}"
echo "  Base output:     ${BASE_OUT}"
echo "  Adapted output:  ${ADAPTED_OUT}"
echo "=========================================="

# --- Step 1: Prepare data (CPU, short) ---
JOB_PREP=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J tulu_prepare
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/prepare_tulu.py \
  --raw_dir ${RAW_DIR} \
  --out_dir ${DATA_DIR}

echo "Data preparation complete."
EOF
)
echo "Submitted prepare job: ${JOB_PREP}"

# --- Step 2a: Train base LoRA (GPU) ---
JOB_BASE=$(sbatch --parsable --dependency=afterok:${JOB_PREP} <<EOF
#!/bin/bash
#SBATCH -J tulu_base_lora
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora.py \
  use_adapter=False \
  pretrained_ckpt=${PRETRAINED_CKPT} \
  data_dir=${BASE_DATA} \
  out_dir=${BASE_OUT} \
  block_size=512 \
  batch_size=4 \
  gradient_accumulation_steps=40 \
  lora_rank=16 \
  lora_alpha=32 \
  lora_dropout=0.05 \
  lora_targets=q_proj,v_proj,k_proj,o_proj \
  max_iters=15000 \
  learning_rate=3e-4 \
  lora_lr_scale=1.0 \
  warmup_iters=500 \
  lr_decay_iters=15000 \
  min_lr=3e-5 \
  num_norm_match=True \
  eval_interval=2000 \
  diag_interval=100 \
  sample_interval=1000 \
  log_interval=10

echo "Base LoRA training complete."
EOF
)
echo "Submitted base LoRA job: ${JOB_BASE} (depends on ${JOB_PREP})"

# --- Step 2b: Train adapted LoRA (GPU) ---
JOB_ADAPT=$(sbatch --parsable --dependency=afterok:${JOB_PREP} <<EOF
#!/bin/bash
#SBATCH -J tulu_adapt_lora
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora.py \
  use_adapter=True \
  stage1_ckpt=${STAGE1_CKPT} \
  data_dir=${ADAPTED_DATA} \
  out_dir=${ADAPTED_OUT} \
  block_size=512 \
  batch_size=4 \
  gradient_accumulation_steps=40 \
  lora_rank=16 \
  lora_alpha=32 \
  lora_dropout=0.05 \
  lora_targets=q_proj,v_proj,k_proj,o_proj \
  max_iters=15000 \
  learning_rate=3e-4 \
  lora_lr_scale=1.0 \
  adapter_lr_scale=0.3 \
  warmup_iters=500 \
  lr_decay_iters=15000 \
  min_lr=3e-5 \
  num_norm_match=True \
  eval_interval=2000 \
  diag_interval=100 \
  sample_interval=1000 \
  log_interval=10

echo "Adapted LoRA training complete."
EOF
)
echo "Submitted adapted LoRA job: ${JOB_ADAPT} (depends on ${JOB_PREP})"

# --- Step 3: Benchmark both (after training) ---
JOB_BENCH=$(sbatch --parsable --dependency=afterok:${JOB_BASE}:${JOB_ADAPT} <<EOF
#!/bin/bash
#SBATCH -J tulu_benchmark
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_tulu.py \
  --base_ckpt ${BASE_OUT}/ckpt_merged.pt \
  --adapted_ckpt ${ADAPTED_OUT}/ckpt_merged.pt \
  --base_data_dir ${BASE_DATA} \
  --adapted_data_dir ${ADAPTED_DATA} \
  --n_forward_batches 200 \
  --n_gen_samples 100

echo "Benchmark complete."
EOF
)
echo "Submitted benchmark job: ${JOB_BENCH} (depends on ${JOB_BASE}, ${JOB_ADAPT})"

echo ""
echo "=========================================="
echo "All jobs submitted:"
echo "  1. Prepare:       ${JOB_PREP}"
echo "  2a. Base LoRA:    ${JOB_BASE}"
echo "  2b. Adapted LoRA: ${JOB_ADAPT}"
echo "  3. Benchmark:     ${JOB_BENCH}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "=========================================="
