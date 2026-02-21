#!/bin/bash
#SBATCH -J gen-numtasks
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/generate_data_%j.log
#SBATCH -e slurm_logs/generate_data_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPT2_DIR="${REPO_DIR}"
# IMAGE="/work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

N_TRAIN="${N_TRAIN:-1000000}"
N_VAL="${N_VAL:-5000}"
MAX_LEN="${MAX_LEN:-100}"
BASE_DIR="/tmpdir/m24047brmn/numbers/data/numtasks"

module load gnu/11.2.0

# Generate datasets for curriculum: 100, 1k, 10k
for RANGE in 100 1000 10000; do
  OUT_DIR="${BASE_DIR}/${RANGE}"
  mkdir -p "${OUT_DIR}"
  echo "=========================================="
  echo "Generating NUMBER_RANGE=${RANGE} -> ${OUT_DIR}"
  echo "=========================================="

  apptainer exec \
    --env "PYTHONUSERBASE=${MYENVS}/numbers" \
    --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
    --bind /tmpdir,/work \
    "${IMAGE}" \
    env \
      GPT2_DIR="${GPT2_DIR}" \
      N_TRAIN="${N_TRAIN}" \
      N_VAL="${N_VAL}" \
      MAX_LEN="${MAX_LEN}" \
      NUMBER_RANGE="${RANGE}" \
      OUT_DIR="${OUT_DIR}" \
    bash -c '
      set -euo pipefail
      cd "${GPT2_DIR}"
      python3 -c "import tiktoken; print(\"tiktoken OK\")"
      python3 generate_data.py \
        --n-train "${N_TRAIN}" \
        --n-val "${N_VAL}" \
        --max-len "${MAX_LEN}" \
        --number-range "${NUMBER_RANGE}" \
        --block-size 256 \
        --allow-negative \
        --out-dir "${OUT_DIR}" \
        --allow-float
    '
done
