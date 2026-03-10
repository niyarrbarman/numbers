#!/bin/bash
#
# Run arithmetic benchmark on synth-trained base vs adapted LoRA models.
#
# Pipeline:
#   1. Generate comprehensive arithmetic benchmark data (CPU)
#   2. Evaluate both models (GPU)
#
# Usage:
#   bash run_arithmetic_benchmark.sh
#
set -euo pipefail
mkdir -p slurm

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/calmip/nemo_25.04.03_arm.sif"

# Model checkpoints (from synth experiment)
BASE_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/synth_base_lora/ckpt_merged.pt"
ADAPTED_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/synth_adapted_lora/ckpt_merged.pt"

# Benchmark data
BENCH_DATA="/tmpdir/m24047brmn/numbers/data/arithmetic_bench_v2.json"

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

echo "=========================================="
echo "Arithmetic Benchmark (v2: ID + OOD splits)"
echo "  Base ckpt:    ${BASE_CKPT}"
echo "  Adapted ckpt: ${ADAPTED_CKPT}"
echo "  Bench data:   ${BENCH_DATA}"
echo "=========================================="

# --- Step 1: Generate benchmark data ---
JOB_GEN=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J arith_gen_v2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=00:15:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_arithmetic_data.py \
  --out_path ${BENCH_DATA} \
  --n_problems 1500 \
  --seed 42

echo "Benchmark data generated."
EOF
)
echo "Submitted data gen job: ${JOB_GEN}"

# --- Step 2: Benchmark both models ---
JOB_BENCH=$(sbatch --parsable --dependency=afterok:${JOB_GEN} <<EOF
#!/bin/bash
#SBATCH -J arith_bench_v2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_arithmetic.py \
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${ADAPTED_CKPT} \
  --data_path ${BENCH_DATA} \
  --max_samples 1500 \
  --max_new_tokens 128

echo "Benchmark complete."
EOF
)
echo "Submitted benchmark job: ${JOB_BENCH} (depends on ${JOB_GEN})"

echo ""
echo "=========================================="
echo "Jobs submitted:"
echo "  1. Data gen:   ${JOB_GEN}"
echo "  2. Benchmark:  ${JOB_BENCH}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "=========================================="
