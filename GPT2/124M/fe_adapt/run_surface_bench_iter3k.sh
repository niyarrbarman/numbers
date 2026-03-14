#!/bin/bash
#
# Benchmark a specific surface S2 checkpoint against base.
# Runs independently — safe to submit while training job is still running.
#
# Usage:
#   ./run_surface_bench_iter3k.sh                        # defaults to iter3000
#   ./run_surface_bench_iter3k.sh --ckpt ckpt_iter4000.pt
#
set -euo pipefail
mkdir -p slurm

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

S2_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/surface_s2_adapted_lora"
BASE_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_base_lora/ckpt_merged.pt"

S2_BASE_DATA="/tmpdir/m24047brmn/numbers/data/synth_arith_surface/base"
S2_ADAPTED_DATA="/tmpdir/m24047brmn/numbers/data/synth_arith_surface/adapted"

CKPT_NAME="ckpt_iter3000.pt"
N_GEN_SAMPLES=3000
N_FORWARD_BATCHES=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT_NAME="$2"
            shift 2
            ;;
        --base-ckpt)
            BASE_CKPT="$2"
            shift 2
            ;;
        --n-gen-samples)
            N_GEN_SAMPLES="$2"
            shift 2
            ;;
        --n-forward-batches)
            N_FORWARD_BATCHES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Derive a clean label from the checkpoint name (e.g. ckpt_iter3000.pt -> iter3000)
CKPT_LABEL=$(basename "${CKPT_NAME}" .pt | sed 's/^ckpt_//')

INPUT_CKPT="${S2_OUT_DIR}/${CKPT_NAME}"
BENCH_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/surface_bench_${CKPT_LABEL}"
MERGED_CKPT="${BENCH_OUT_DIR}/surface_${CKPT_LABEL}_merged.pt"
BENCH_JSON="${BENCH_OUT_DIR}/surface_${CKPT_LABEL}_benchmark.json"

# Sanity checks
if [[ ! -f "${INPUT_CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${INPUT_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "ERROR: base checkpoint not found: ${BASE_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${S2_BASE_DATA}/test_examples.json" ]]; then
    echo "ERROR: base test examples not found: ${S2_BASE_DATA}/test_examples.json" >&2
    exit 1
fi
if [[ ! -f "${S2_ADAPTED_DATA}/test_examples.json" ]]; then
    echo "ERROR: adapted test examples not found: ${S2_ADAPTED_DATA}/test_examples.json" >&2
    exit 1
fi

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

echo "=========================================================="
echo "Surface Benchmark: ${CKPT_LABEL}"
echo "  Input ckpt:  ${INPUT_CKPT}"
echo "  Merged ckpt: ${MERGED_CKPT}"
echo "  Base ckpt:   ${BASE_CKPT}"
echo "  Base data:   ${S2_BASE_DATA}"
echo "  Adapted data:${S2_ADAPTED_DATA}"
echo "  Output:      ${BENCH_JSON}"
echo "  gen samples: ${N_GEN_SAMPLES}  forward batches: ${N_FORWARD_BATCHES}"
echo "=========================================================="

JOB_BENCH=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J surf_bench_${CKPT_LABEL}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0
mkdir -p ${BENCH_OUT_DIR}

echo "=== Merging LoRA weights: ${CKPT_LABEL} ==="
${APPTAINER} python3 ${SCRIPT_DIR}/merge_tulu_lora_surface_checkpoint.py \
  --input_ckpt ${INPUT_CKPT} \
  --output_ckpt ${MERGED_CKPT}

echo "=== Running benchmark ==="
${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_synth_surface.py \
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${MERGED_CKPT} \
  --base_data_dir ${S2_BASE_DATA} \
  --adapted_data_dir ${S2_ADAPTED_DATA} \
  --n_forward_batches ${N_FORWARD_BATCHES} \
  --n_gen_samples ${N_GEN_SAMPLES} \
  --out_path ${BENCH_JSON}

echo "=== Done: ${BENCH_JSON} ==="
EOF
)

echo "Submitted: job ${JOB_BENCH}"
echo "  Output:  slurm/surf_bench_${CKPT_LABEL}_${JOB_BENCH}.out"
echo "  Results: ${BENCH_JSON}"
echo
echo "Monitor: squeue -u \$USER"
echo "Log:     tail -f slurm/surf_bench_${CKPT_LABEL}_${JOB_BENCH}.out"
