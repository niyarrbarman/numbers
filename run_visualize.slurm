#!/bin/bash
#SBATCH -J np-emb-viz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -o slurm_logs/np_emb_viz_%j.log
#SBATCH -e slurm_logs/np_emb_viz_%j.log

mkdir -p slurm_logs
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="${REPO_DIR}/visualize_embeddings.py"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

CHECKPOINT_DIR="/tmpdir/m24047brmn/numbers/checkpoints"
PLOT_DIR="/tmpdir/m24047brmn/numbers/plots"

module load gnu/11.2.0

apptainer exec \
  --cleanenv \
  --bind /tmpdir,/work \
  "${IMAGE}" \
  env \
    REPO_DIR="${REPO_DIR}" \
    SCRIPT_PATH="${SCRIPT_PATH}" \
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    PLOT_DIR="${PLOT_DIR}" \
  bash -c '
    set -euo pipefail
    cd "${REPO_DIR}"

    echo "Checking for embeddings in ${CHECKPOINT_DIR}..."
    ls -la "${CHECKPOINT_DIR}"/*.npz 2>/dev/null || { echo "ERROR: No embeddings found. Run training first."; exit 1; }

    echo "Available packages:"
    python3 -c "import sklearn; print(f\"  sklearn {sklearn.__version__}\")"
    python3 -c "import umap; print(f\"  umap {umap.__version__}\")" 2>/dev/null || echo "  umap: not available (will use t-SNE)"
    python3 -c "import matplotlib; print(f\"  matplotlib {matplotlib.__version__}\")"
    echo ""

    python3 "${SCRIPT_PATH}" --plot-dir "${PLOT_DIR}"

    echo ""
    echo "Plots:"
    ls -la "${PLOT_DIR}"/*.png 2>/dev/null
  '
