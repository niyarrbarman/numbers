#!/bin/bash
#SBATCH -J np-emb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o np_emb_%j.log
#SBATCH -e np_emb_%j.log

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="${REPO_DIR}/np_emb.py"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

module load gnu/11.2.0

apptainer exec \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    REPO_DIR="${REPO_DIR}" \
    SCRIPT_PATH="${SCRIPT_PATH}" \
  bash -lc '
    set -euo pipefail
    cd "${REPO_DIR}"
    /usr/bin/python3 -c "import numpy as np; print(f\"NumPy {np.__version__}\")"
    /usr/bin/python3 "${SCRIPT_PATH}" --max-steps 100000
  '
