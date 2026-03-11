#!/bin/bash
#
# Analytic stage-2 rerun using an existing base LoRA checkpoint.
# Submits:
#   1. S2 analytic data generation
#   2. S2 analytic-adapted LoRA training
#   3. Benchmarks vs existing base LoRA checkpoint
#
# Usage:
#   bash run_analytic_s2_adapt_bench.sh
#   bash run_analytic_s2_adapt_bench.sh --base-ckpt /path/to/base/ckpt_merged.pt
#   bash run_analytic_s2_adapt_bench.sh --resume-from-data <JOBID>
#   bash run_analytic_s2_adapt_bench.sh --resume-from-adapt <JOBID>
#
set -euo pipefail
mkdir -p slurm

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

# Existing checkpoints
S1_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_analytic_s1"
S1_CKPT="${S1_OUT_DIR}/ckpt.pt"
BASE_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_base_lora/ckpt_merged.pt"

# Stage 2
S2_DATA_DIR="/tmpdir/m24047brmn/numbers/data/synth_arith_analytic"
S2_BASE_DATA="${S2_DATA_DIR}/base"
S2_ADAPTED_DATA="${S2_DATA_DIR}/adapted"
S2_ADAPTED_OUT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_adapted_lora"

# Benchmark
ARITH_BENCH="/tmpdir/m24047brmn/numbers/data/arithmetic_bench.json"
RESUME_FROM_DATA=""
RESUME_FROM_ADAPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-ckpt)
            BASE_CKPT="$2"
            shift 2
            ;;
        --resume-from-data)
            RESUME_FROM_DATA="$2"
            shift 2
            ;;
        --resume-from-adapt)
            RESUME_FROM_ADAPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${RESUME_FROM_DATA}" && -n "${RESUME_FROM_ADAPT}" ]]; then
    echo "Use only one of --resume-from-data or --resume-from-adapt." >&2
    exit 1
fi

if [[ ! -f "${S1_CKPT}" ]]; then
    echo "Missing Stage 1 checkpoint: ${S1_CKPT}" >&2
    exit 1
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing base stage-2 checkpoint: ${BASE_CKPT}" >&2
    echo "Expected the already-trained base run to have produced ckpt_merged.pt." >&2
    exit 1
fi

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

echo "=========================================================="
echo "Analytic Stage-2 Adapt + Benchmark"
echo "  Stage 1 ckpt: ${S1_CKPT}"
echo "  Base ckpt:    ${BASE_CKPT}"
echo "  Stage 2 data: ${S2_DATA_DIR}"
echo "  Adapt output: ${S2_ADAPTED_OUT}"
if [[ -n "${RESUME_FROM_DATA}" ]]; then
    echo "  Resume mode:  from data job ${RESUME_FROM_DATA}"
elif [[ -n "${RESUME_FROM_ADAPT}" ]]; then
    echo "  Resume mode:  from adapt job ${RESUME_FROM_ADAPT}"
fi
echo "=========================================================="

submit_adapt_job() {
    local dependency="$1"
    sbatch --parsable ${dependency} <<EOF
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

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora_analytic.py \
  stage1_ckpt=${S1_CKPT} \
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
  decoder_lr_scale=0.3 \
  warmup_iters=500 \
  lr_decay_iters=10000 \
  min_lr=3e-5 \
  eval_interval=1000 \
  diag_interval=100 \
  sample_interval=500 \
  log_interval=10

echo "Analytic-adapter LoRA training complete."
EOF
}

submit_bench_job() {
    local dependency="$1"
    sbatch --parsable ${dependency} <<EOF
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
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${S2_ADAPTED_OUT}/ckpt_merged.pt \
  --data_path ${ARITH_BENCH} \
  --max_samples 1000 \
  --max_new_tokens 128

echo ""
echo "============================================================"
echo "BENCHMARK: Forward-pass on synth test data"
echo "============================================================"

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_tulu.py \
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${S2_ADAPTED_OUT}/ckpt_merged.pt \
  --base_data_dir ${S2_BASE_DATA} \
  --adapted_data_dir ${S2_ADAPTED_DATA} \
  --n_forward_batches 200 \
  --n_gen_samples 100

echo "All benchmarks complete."
EOF
}

if [[ -n "${RESUME_FROM_ADAPT}" ]]; then
    JOB_S2_ADAPT="${RESUME_FROM_ADAPT}"
    JOB_BENCH=$(submit_bench_job "--dependency=afterok:${JOB_S2_ADAPT}")
    echo "  [2] S2 Analytic LoRA (EXISTING): ${JOB_S2_ADAPT}"
    echo "  [3] Benchmark: ${JOB_BENCH} (afterok:${JOB_S2_ADAPT})"
elif [[ -n "${RESUME_FROM_DATA}" ]]; then
    JOB_S2_DATA="${RESUME_FROM_DATA}"
    JOB_S2_ADAPT=$(submit_adapt_job "--dependency=afterok:${JOB_S2_DATA}")
    JOB_BENCH=$(submit_bench_job "--dependency=afterok:${JOB_S2_ADAPT}")
    echo "  [1] S2 Data Gen (EXISTING): ${JOB_S2_DATA}"
    echo "  [2] S2 Analytic LoRA: ${JOB_S2_ADAPT} (afterok:${JOB_S2_DATA})"
    echo "  [3] Benchmark: ${JOB_BENCH} (afterok:${JOB_S2_ADAPT})"
else
    # --- Step 1: S2 Data Gen (CPU) ---
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
  --n_test 3000 \
  --analytic_adapted

if [ ! -f ${ARITH_BENCH} ]; then
  ${APPTAINER} python3 ${SCRIPT_DIR}/generate_arithmetic_data.py \
    --out_path ${ARITH_BENCH} \
    --n_problems 1000 \
    --seed 42
fi

echo "Stage 2 data generation complete."
EOF
)

    JOB_S2_ADAPT=$(submit_adapt_job "--dependency=afterok:${JOB_S2_DATA}")
    JOB_BENCH=$(submit_bench_job "--dependency=afterok:${JOB_S2_ADAPT}")
    echo "  [1] S2 Data Gen: ${JOB_S2_DATA}"
    echo "  [2] S2 Analytic LoRA: ${JOB_S2_ADAPT} (afterok:${JOB_S2_DATA})"
    echo "  [3] Benchmark: ${JOB_BENCH} (afterok:${JOB_S2_ADAPT})"
fi

echo ""
echo "=========================================================="
echo "Jobs submitted:"
if [[ -n "${JOB_S2_DATA:-}" ]]; then
    echo "  [1] S2 Data Gen:      ${JOB_S2_DATA}"
fi
echo "  [2] S2 Analytic LoRA: ${JOB_S2_ADAPT}"
echo "  [3] Benchmark:        ${JOB_BENCH}"
echo ""
echo "Dependency graph:"
if [[ -n "${JOB_S2_DATA:-}" ]]; then
    echo "  [1] ──→ [2] ──→ [3]"
else
    echo "  [2] ──→ [3]"
fi
echo ""
echo "Using existing base LoRA: ${BASE_CKPT}"
echo "Monitor: squeue -u \$USER"
echo "=========================================================="
