#!/bin/bash
#SBATCH -J qwen_smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

mkdir -p slurm
set -euo pipefail

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/llama3"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
HF_CACHE="/tmpdir/m24047brmn/hf_cache"

echo "=========================================="
echo "Smoke Test: Qwen2.5-0.5B + NumLM"
echo "  Script dir: $SCRIPT_DIR"
echo "  HF cache:   $HF_CACHE"
echo "=========================================="

module load gnu/11.2.0

apptainer exec \
  --nv \
  --env "PYTHONUSERBASE=${MYENVS}/numbers" \
  --env "TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache" \
  --env "HF_HOME=${HF_CACHE}" \
  --env "PYTHONUNBUFFERED=1" \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  python3 "${SCRIPT_DIR}/main_qwen.py" \
    --model_path "/work/m24047/m24047brmn/Qwen2.5-0.5B-Instruct" \
    --device cuda

status=$?
echo "=========================================="
echo "Smoke test finished with status $status"
echo "=========================================="
exit $status
