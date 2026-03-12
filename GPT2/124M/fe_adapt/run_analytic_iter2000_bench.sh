#!/bin/bash
#
# Merge analytic LoRA checkpoint iter2000 and benchmark it against base.
#
# Usage:
#   bash run_analytic_iter2000_bench.sh
#   bash run_analytic_iter2000_bench.sh --adapt-ckpt /path/to/ckpt_iter2000.pt
#
set -euo pipefail
mkdir -p slurm

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

BASE_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_base_lora/ckpt_merged.pt"
ADAPT_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_adapted_lora/ckpt_iter2000.pt"
S2_DATA_DIR="/tmpdir/m24047brmn/numbers/data/synth_arith_analytic"
ARITH_BENCH="/tmpdir/m24047brmn/numbers/data/arithmetic_bench.json"
OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_iter2000_bench"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapt-ckpt)
            ADAPT_CKPT="$2"
            shift 2
            ;;
        --base-ckpt)
            BASE_CKPT="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing base checkpoint: ${BASE_CKPT}" >&2
    exit 1
fi

if [[ ! -f "${ADAPT_CKPT}" ]]; then
    echo "Missing adapted checkpoint: ${ADAPT_CKPT}" >&2
    exit 1
fi

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

MERGED_CKPT="${OUT_DIR}/$(basename "${ADAPT_CKPT%.pt}")_merged.pt"
ARITH_JSON="${OUT_DIR}/arithmetic_iter2000_detailed.json"
TULU_JSON="${OUT_DIR}/tulu_iter2000_detailed.json"

echo "=========================================================="
echo "Analytic Iter2000 Benchmark"
echo "  Base ckpt:      ${BASE_CKPT}"
echo "  Adapt ckpt:     ${ADAPT_CKPT}"
echo "  Merged output:  ${MERGED_CKPT}"
echo "  Result dir:     ${OUT_DIR}"
echo "=========================================================="

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J anl_iter2k_bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0
mkdir -p ${OUT_DIR}

${APPTAINER} python3 ${SCRIPT_DIR}/merge_tulu_lora_analytic_checkpoint.py \
  --input_ckpt ${ADAPT_CKPT} \
  --output_ckpt ${MERGED_CKPT}

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_arithmetic.py \
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${MERGED_CKPT} \
  --data_path ${ARITH_BENCH} \
  --max_samples 1000 \
  --max_new_tokens 128 \
  --out_path ${ARITH_JSON}

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_tulu.py \
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${MERGED_CKPT} \
  --base_data_dir ${S2_DATA_DIR}/base \
  --adapted_data_dir ${S2_DATA_DIR}/adapted \
  --n_forward_batches 200 \
  --n_gen_samples 100 \
  --out_path ${TULU_JSON}

echo "Detailed outputs:"
echo "  ${ARITH_JSON}"
echo "  ${TULU_JSON}"
EOF
)

echo "Submitted benchmark job: ${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
