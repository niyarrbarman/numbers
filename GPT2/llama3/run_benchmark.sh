#!/bin/bash
#
# Benchmark augmented vs baseline after S2 training completes.
#
# Usage:
#   bash run_benchmark.sh
#   bash run_benchmark.sh --s2_aug_job 79842 --s2_base_job 79843
#
set -euo pipefail
mkdir -p slurm

# --- Parse args ---
S2_AUG_JOB=""
S2_BASE_JOB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --s2_aug_job)  S2_AUG_JOB="$2";  shift 2 ;;
        --s2_base_job) S2_BASE_JOB="$2"; shift 2 ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/llama3"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
QWEN_PATH="/work/m24047/m24047brmn/Qwen2.5-0.5B-Instruct"

S1_DATA="/tmpdir/m24047brmn/numbers/data/qwen_s1"
S2_DATA="/tmpdir/m24047brmn/numbers/data/qwen_s2"
S2_AUG_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/qwen_s2_augmented"
S2_BASE_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/qwen_s2_baseline"
BENCH_OUT="/tmpdir/m24047brmn/numbers/benchmark"

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=\${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

# build dependency flag
DEP_FLAG=""
if [ -n "$S2_AUG_JOB" ] && [ -n "$S2_BASE_JOB" ]; then
    DEP_FLAG="--dependency=afterok:${S2_AUG_JOB}:${S2_BASE_JOB}"
fi

echo "=========================================="
echo "Benchmark: Augmented vs Baseline"
echo "  Augmented ckpt: ${S2_AUG_CKPT}/stage2_epoch3.pt"
echo "  Baseline ckpt:  ${S2_BASE_CKPT}/baseline_epoch3.pt"
echo "  S1 test:        ${S1_DATA}/test.jsonl"
echo "  GSM8K test:     ${S2_DATA}/augmented/test.jsonl"
if [ -n "$DEP_FLAG" ]; then
    echo "  Dependency:     afterok:${S2_AUG_JOB}:${S2_BASE_JOB}"
fi
echo "=========================================="

JOB_BENCH=$(sbatch --parsable ${DEP_FLAG} <<EOF
#!/bin/bash
#SBATCH -J benchmark
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark.py \
  --augmented_ckpt ${S2_AUG_CKPT}/stage2_epoch3.pt \
  --baseline_ckpt ${S2_BASE_CKPT}/baseline_epoch3.pt \
  --model_path ${QWEN_PATH} \
  --gsm8k_test ${S2_DATA}/augmented/test.jsonl \
  --gsm8k_test_baseline ${S2_DATA}/baseline/test.jsonl \
  --s1_test ${S1_DATA}/test.jsonl \
  --max_synth 3000 \
  --max_new_tokens 256 \
  --out_path ${BENCH_OUT}/benchmark_results.json \
  --device cuda

echo "Benchmark complete."
EOF
)
echo "  Benchmark job: ${JOB_BENCH}"
echo "  Monitor: squeue -u \$USER"
echo "=========================================="
