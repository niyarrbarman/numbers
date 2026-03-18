#!/bin/bash
#
# Full augmented vs baseline experiment pipeline.
#
# Flow:
#   1. S1 Data Gen (CPU)        — generate_s1_data.py
#   2. S1 Train (GPU)           — main_qwen.py --mode train_augmented --stage 1
#   3. S2 Data Gen (CPU)        — generate_s2_data.py (parallel with 1+2)
#   4. S2 Augmented (GPU)       — main_qwen.py --mode train_augmented --stage 2
#   5. S2 Baseline (GPU)        — main_qwen.py --mode train_baseline
#   6. TODO: Benchmark
#
# Usage:
#   bash run_pipeline.sh --qwen_path /path/to/Qwen2.5-0.5B-Instruct
#   bash run_pipeline.sh --qwen_path /path/to/model --gsm8k_dir /path/to/gsm8k
#
set -euo pipefail
mkdir -p slurm

# --- Parse args ---
QWEN_PATH="/work/m24047/m24047brmn/Qwen2.5-0.5B-Instruct"
GSM8K_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qwen_path) QWEN_PATH="$2"; shift 2 ;;
        --gsm8k_dir) GSM8K_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/llama3"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# --- Paths ---
S1_DATA="/tmpdir/m24047brmn/numbers/data/qwen_s1"
S2_DATA="/tmpdir/m24047brmn/numbers/data/qwen_s2"
S1_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/qwen_s1"
S2_AUG_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/qwen_s2_augmented"
S2_BASE_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/qwen_s2_baseline"

if [ -z "$GSM8K_DIR" ]; then
    GSM8K_DIR="/tmpdir/m24047brmn/numbers/data/gsm8k"
fi

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=\${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

echo "=========================================================="
echo "Pipeline: Qwen2.5-0.5B Augmented vs Baseline"
echo "  Model:        $QWEN_PATH"
echo "  S1 data:      $S1_DATA"
echo "  S2 data:      $S2_DATA"
echo "  GSM8K:        $GSM8K_DIR"
echo "  S1 ckpt:      $S1_CKPT"
echo "  S2 augmented: $S2_AUG_CKPT"
echo "  S2 baseline:  $S2_BASE_CKPT"
echo "=========================================================="

# --- Step 1: S1 Data Gen (CPU) ---
JOB_S1_DATA=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J s1_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_s1_data.py \
  --out_dir ${S1_DATA} \
  --n_train 100000 \
  --n_val 5000 \
  --n_test 3000 \
  --seed 42

echo "S1 data gen complete."
EOF
)
echo "  [1] S1 Data Gen: ${JOB_S1_DATA}"

# --- Step 2: S1 Train (GPU, afterok:1) ---
JOB_S1_TRAIN=$(sbatch --parsable --dependency=afterok:${JOB_S1_DATA} <<EOF
#!/bin/bash
#SBATCH -J s1_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/main_qwen.py \
  --mode train_augmented \
  --model_path ${QWEN_PATH} \
  --data_path ${S1_DATA}/train.jsonl \
  --val_path ${S1_DATA}/val.jsonl \
  --stage 1 \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-3 \
  --max_length 256 \
  --grad_accum_steps 4 \
  --warmup_steps 100 \
  --save_path ${S1_CKPT} \
  --device cuda

echo "S1 training complete."
EOF
)
echo "  [2] S1 Train: ${JOB_S1_TRAIN} (afterok:${JOB_S1_DATA})"

# --- Step 3: S2 Data Gen (CPU, parallel with S1) ---
JOB_S2_DATA=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J s2_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_s2_data.py \
  --gsm8k_dir ${GSM8K_DIR} \
  --out_dir ${S2_DATA} \
  --n_synth 50000 \
  --seed 123

echo "S2 data gen complete."
EOF
)
echo "  [3] S2 Data Gen: ${JOB_S2_DATA}"

# --- Step 4: S2 Augmented Train (GPU, afterok:2,3) ---
JOB_S2_AUG=$(sbatch --parsable --dependency=afterok:${JOB_S1_TRAIN}:${JOB_S2_DATA} <<EOF
#!/bin/bash
#SBATCH -J s2_augmented
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/main_qwen.py \
  --mode train_augmented \
  --model_path ${QWEN_PATH} \
  --data_path ${S2_DATA}/augmented/train.jsonl \
  --val_path ${S2_DATA}/augmented/val.jsonl \
  --stage 2 \
  --checkpoint ${S1_CKPT}/stage1_epoch3.pt \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-5 \
  --max_length 512 \
  --grad_accum_steps 8 \
  --warmup_steps 100 \
  --save_path ${S2_AUG_CKPT} \
  --device cuda

echo "S2 augmented training complete."
EOF
)
echo "  [4] S2 Augmented: ${JOB_S2_AUG} (afterok:${JOB_S1_TRAIN},${JOB_S2_DATA})"

# --- Step 5: S2 Baseline Train (GPU, afterok:3) ---
JOB_S2_BASE=$(sbatch --parsable --dependency=afterok:${JOB_S2_DATA} <<EOF
#!/bin/bash
#SBATCH -J s2_baseline
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/main_qwen.py \
  --mode train_baseline \
  --model_path ${QWEN_PATH} \
  --data_path ${S2_DATA}/baseline/train.jsonl \
  --val_path ${S2_DATA}/baseline/val.jsonl \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-5 \
  --max_length 512 \
  --grad_accum_steps 8 \
  --warmup_steps 100 \
  --save_path ${S2_BASE_CKPT} \
  --device cuda

echo "S2 baseline training complete."
EOF
)
echo "  [5] S2 Baseline: ${JOB_S2_BASE} (afterok:${JOB_S2_DATA})"

echo ""
echo "=========================================================="
echo "All jobs submitted:"
echo "  [1] S1 Data Gen:     ${JOB_S1_DATA}"
echo "  [2] S1 Train:        ${JOB_S1_TRAIN}"
echo "  [3] S2 Data Gen:     ${JOB_S2_DATA}"
echo "  [4] S2 Augmented:    ${JOB_S2_AUG}"
echo "  [5] S2 Baseline:     ${JOB_S2_BASE}"
echo ""
echo "Dependency graph:"
echo "  [1] ──→ [2] ──→ [4]"
echo "  [3] ──→ [4]"
echo "  [3] ──→ [5]"
echo ""
echo "Monitor: squeue -u \$USER"
echo "=========================================================="
