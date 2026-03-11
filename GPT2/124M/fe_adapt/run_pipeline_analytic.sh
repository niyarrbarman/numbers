#!/bin/bash
#
# Full analytic number integration experiment.
# Submits SLURM jobs with dependency chain:
#
#   Default (full pipeline):
#     1. S1 Data Gen (CPU)               — generate_data_analytic.py
#     2. S1 Train (GPU)                  — train_analytic.py         (afterok:1)
#     3. S2 Data Gen (CPU)               — generate_synth_math.py    (parallel with 1+2)
#     4a. S2 Base LoRA (GPU)             — train_tulu_lora.py        (afterok:3)
#     4b. S2 Analytic-Adapter LoRA (GPU) — train_tulu_lora.py        (afterok:2,3)
#     5. Benchmark (GPU)                 — benchmark_arithmetic.py   (afterok:4a,4b)
#
#   --resume  (skip completed steps):
#     2. S1 Train (GPU)                  — train_analytic.py
#     4b. S2 Analytic-Adapter LoRA (GPU) — train_tulu_lora.py        (afterok:2)
#     5. Benchmark (GPU)                 — benchmark_arithmetic.py   (afterok:4b)
#         Uses already-trained base LoRA checkpoint for comparison.
#
# Usage:
#   bash run_pipeline_analytic.sh          # full pipeline
#   bash run_pipeline_analytic.sh --resume # skip data gen + base LoRA
#
set -euo pipefail
mkdir -p slurm

RESUME=false
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=true ;;
    esac
done

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# --- Paths ---
# Stage 1
S1_DATA_DIR="/tmpdir/m24047brmn/numbers/data/analytic_stage1"
S1_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_analytic_s1"
PRETRAINED_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/baby_luciole_converted.pt"

# Stage 2
S2_DATA_DIR="/tmpdir/m24047brmn/numbers/data/synth_arith_analytic"
S2_BASE_DATA="${S2_DATA_DIR}/base"
S2_ADAPTED_DATA="${S2_DATA_DIR}/adapted"
S2_BASE_OUT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_base_lora"
S2_ADAPTED_OUT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_adapted_lora"

# Benchmark
ARITH_BENCH="/tmpdir/m24047brmn/numbers/data/arithmetic_bench.json"

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

if [ "$RESUME" = true ]; then
    echo "=========================================================="
    echo "RESUME: Analytic Pipeline (skip data gen + base LoRA)"
    echo "  S1 train → S2 analytic LoRA → benchmark"
    echo "=========================================================="

    # --- Step 2: S1 Training (GPU) ---
    JOB_S1_TRAIN=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J anl_s1_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_analytic.py \
  init_from=pretrained \
  pretrained_ckpt=${PRETRAINED_CKPT} \
  data_dir=${S1_DATA_DIR} \
  out_dir=${S1_OUT_DIR} \
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

echo "Stage 1 training complete."
EOF
)
    echo "  [2] S1 Training: ${JOB_S1_TRAIN}"

    # --- Step 4b: S2 Analytic-Adapter LoRA (GPU, afterok:S1) ---
    JOB_S2_ADAPT=$(sbatch --parsable --dependency=afterok:${JOB_S1_TRAIN} <<EOF
#!/bin/bash
#SBATCH -J anl_s2_adapt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora.py \
  use_adapter=True \
  stage1_ckpt=${S1_OUT_DIR}/ckpt.pt \
  data_dir=${S2_ADAPTED_DATA} \
  out_dir=${S2_ADAPTED_OUT} \
  block_size=256 \
  batch_size=8 \
  gradient_accumulation_steps=20 \
  lora_rank=16 \
  lora_alpha=32 \
  lora_dropout=0.05 \
  lora_targets=q_proj,v_proj,k_proj,o_proj \
  max_iters=10000 \
  learning_rate=3e-4 \
  lora_lr_scale=1.0 \
  adapter_lr_scale=0.3 \
  warmup_iters=500 \
  lr_decay_iters=10000 \
  min_lr=3e-5 \
  num_norm_match=True \
  eval_interval=1000 \
  diag_interval=100 \
  sample_interval=500 \
  log_interval=10

echo "Analytic-adapter LoRA training complete."
EOF
)
    echo "  [4b] S2 Analytic LoRA: ${JOB_S2_ADAPT} (afterok:${JOB_S1_TRAIN})"

    # --- Step 5: Benchmark (GPU, afterok:4b) ---
    # Uses existing base LoRA checkpoint from previous run
    JOB_BENCH=$(sbatch --parsable --dependency=afterok:${JOB_S2_ADAPT} <<EOF
#!/bin/bash
#SBATCH -J anl_bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

echo "============================================================"
echo "BENCHMARK: Arithmetic eval (few-shot)"
echo "============================================================"

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_arithmetic.py \
  --base_ckpt ${S2_BASE_OUT}/ckpt_merged.pt \
  --adapted_ckpt ${S2_ADAPTED_OUT}/ckpt_merged.pt \
  --data_path ${ARITH_BENCH} \
  --max_samples 1000 \
  --max_new_tokens 128

echo ""
echo "============================================================"
echo "BENCHMARK: Forward-pass on synth test data"
echo "============================================================"

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_tulu.py \
  --base_ckpt ${S2_BASE_OUT}/ckpt_merged.pt \
  --adapted_ckpt ${S2_ADAPTED_OUT}/ckpt_merged.pt \
  --base_data_dir ${S2_BASE_DATA} \
  --adapted_data_dir ${S2_ADAPTED_DATA} \
  --n_forward_batches 200 \
  --n_gen_samples 100

echo "All benchmarks complete."
EOF
)
    echo "  [5] Benchmark: ${JOB_BENCH} (afterok:${JOB_S2_ADAPT})"

    echo ""
    echo "=========================================================="
    echo "RESUME jobs submitted:"
    echo "  [2]  S1 Training:       ${JOB_S1_TRAIN}"
    echo "  [4b] S2 Analytic LoRA:  ${JOB_S2_ADAPT}"
    echo "  [5]  Benchmark:         ${JOB_BENCH}"
    echo ""
    echo "  [2] ──→ [4b] ──→ [5]"
    echo ""
    echo "Using existing base LoRA: ${S2_BASE_OUT}/ckpt_merged.pt"
    echo "Monitor: squeue -u \$USER"
    echo "=========================================================="

else
    echo "=========================================================="
    echo "Full Analytic Number Integration Experiment"
    echo "  Stage 1 data:    ${S1_DATA_DIR}"
    echo "  Stage 1 output:  ${S1_OUT_DIR}"
    echo "  Stage 2 data:    ${S2_DATA_DIR}"
    echo "  S2 base output:  ${S2_BASE_OUT}"
    echo "  S2 adapt output: ${S2_ADAPTED_OUT}"
    echo "=========================================================="

    # --- Step 1: S1 Data Gen (CPU) ---
    JOB_S1_DATA=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J anl_s1_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_data_analytic.py \
  --out_dir ${S1_DATA_DIR} \
  --n_train 100000 \
  --n_val 5000 \
  --n_test 3000 \
  --seed 42

echo "Stage 1 data generation complete."
EOF
)
    echo "  [1] S1 Data Gen: ${JOB_S1_DATA}"

    # --- Step 2: S1 Training (GPU, afterok:1) ---
    JOB_S1_TRAIN=$(sbatch --parsable --dependency=afterok:${JOB_S1_DATA} <<EOF
#!/bin/bash
#SBATCH -J anl_s1_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_analytic.py \
  init_from=pretrained \
  pretrained_ckpt=${PRETRAINED_CKPT} \
  data_dir=${S1_DATA_DIR} \
  out_dir=${S1_OUT_DIR} \
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

echo "Stage 1 training complete."
EOF
)
    echo "  [2] S1 Training: ${JOB_S1_TRAIN} (afterok:${JOB_S1_DATA})"

    # --- Step 3: S2 Data Gen (CPU, parallel with S1) ---
    JOB_S2_DATA=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J anl_s2_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_synth_math.py \
  --out_dir ${S2_DATA_DIR} \
  --n_train 50000 \
  --n_val 3000 \
  --n_test 3000

if [ ! -f ${ARITH_BENCH} ]; then
  ${APPTAINER} python3 ${SCRIPT_DIR}/generate_arithmetic_data.py \
    --out_path ${ARITH_BENCH} \
    --n_problems 1000 \
    --seed 42
fi

echo "Stage 2 data generation complete."
EOF
)
    echo "  [3] S2 Data Gen: ${JOB_S2_DATA}"

    # --- Step 4a: S2 Base LoRA (GPU, afterok:3) ---
    JOB_S2_BASE=$(sbatch --parsable --dependency=afterok:${JOB_S2_DATA} <<EOF
#!/bin/bash
#SBATCH -J anl_s2_base
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora.py \
  use_adapter=False \
  pretrained_ckpt=${PRETRAINED_CKPT} \
  data_dir=${S2_BASE_DATA} \
  out_dir=${S2_BASE_OUT} \
  block_size=256 \
  batch_size=8 \
  gradient_accumulation_steps=20 \
  lora_rank=16 \
  lora_alpha=32 \
  lora_dropout=0.05 \
  lora_targets=q_proj,v_proj,k_proj,o_proj \
  max_iters=10000 \
  learning_rate=3e-4 \
  lora_lr_scale=1.0 \
  warmup_iters=500 \
  lr_decay_iters=10000 \
  min_lr=3e-5 \
  num_norm_match=True \
  eval_interval=1000 \
  diag_interval=100 \
  sample_interval=500 \
  log_interval=10

echo "Base LoRA training complete."
EOF
)
    echo "  [4a] S2 Base LoRA: ${JOB_S2_BASE} (afterok:${JOB_S2_DATA})"

    # --- Step 4b: S2 Analytic-Adapter LoRA (GPU, afterok:2,3) ---
    JOB_S2_ADAPT=$(sbatch --parsable --dependency=afterok:${JOB_S1_TRAIN}:${JOB_S2_DATA} <<EOF
#!/bin/bash
#SBATCH -J anl_s2_adapt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora.py \
  use_adapter=True \
  stage1_ckpt=${S1_OUT_DIR}/ckpt.pt \
  data_dir=${S2_ADAPTED_DATA} \
  out_dir=${S2_ADAPTED_OUT} \
  block_size=256 \
  batch_size=8 \
  gradient_accumulation_steps=20 \
  lora_rank=16 \
  lora_alpha=32 \
  lora_dropout=0.05 \
  lora_targets=q_proj,v_proj,k_proj,o_proj \
  max_iters=10000 \
  learning_rate=3e-4 \
  lora_lr_scale=1.0 \
  adapter_lr_scale=0.3 \
  warmup_iters=500 \
  lr_decay_iters=10000 \
  min_lr=3e-5 \
  num_norm_match=True \
  eval_interval=1000 \
  diag_interval=100 \
  sample_interval=500 \
  log_interval=10

echo "Analytic-adapter LoRA training complete."
EOF
)
    echo "  [4b] S2 Analytic LoRA: ${JOB_S2_ADAPT} (afterok:${JOB_S1_TRAIN},${JOB_S2_DATA})"

    # --- Step 5: Benchmark (GPU, afterok:4a,4b) ---
    JOB_BENCH=$(sbatch --parsable --dependency=afterok:${JOB_S2_BASE}:${JOB_S2_ADAPT} <<EOF
#!/bin/bash
#SBATCH -J anl_bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

echo "============================================================"
echo "BENCHMARK: Arithmetic eval (few-shot)"
echo "============================================================"

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_arithmetic.py \
  --base_ckpt ${S2_BASE_OUT}/ckpt_merged.pt \
  --adapted_ckpt ${S2_ADAPTED_OUT}/ckpt_merged.pt \
  --data_path ${ARITH_BENCH} \
  --max_samples 1000 \
  --max_new_tokens 128

echo ""
echo "============================================================"
echo "BENCHMARK: Forward-pass on synth test data"
echo "============================================================"

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_tulu.py \
  --base_ckpt ${S2_BASE_OUT}/ckpt_merged.pt \
  --adapted_ckpt ${S2_ADAPTED_OUT}/ckpt_merged.pt \
  --base_data_dir ${S2_BASE_DATA} \
  --adapted_data_dir ${S2_ADAPTED_DATA} \
  --n_forward_batches 200 \
  --n_gen_samples 100

echo "All benchmarks complete."
EOF
)
    echo "  [5] Benchmark: ${JOB_BENCH} (afterok:${JOB_S2_BASE},${JOB_S2_ADAPT})"

    echo ""
    echo "=========================================================="
    echo "All jobs submitted:"
    echo "  [1]  S1 Data Gen:       ${JOB_S1_DATA}"
    echo "  [2]  S1 Training:       ${JOB_S1_TRAIN}"
    echo "  [3]  S2 Data Gen:       ${JOB_S2_DATA}"
    echo "  [4a] S2 Base LoRA:      ${JOB_S2_BASE}"
    echo "  [4b] S2 Analytic LoRA:  ${JOB_S2_ADAPT}"
    echo "  [5]  Benchmark:         ${JOB_BENCH}"
    echo ""
    echo "Dependency graph:"
    echo "  [1] ──→ [2] ──→ [4b] ──→ [5]"
    echo "  [3] ──→ [4a] ──→ [5]"
    echo "  [3] ──→ [4b]"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "=========================================================="
fi
