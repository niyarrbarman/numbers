#!/bin/bash
# Submit multi-position projection experiment pipeline.
# Generates new data (token sequences change with k=5 positions per number),
# then trains and validates.
#
# Pipeline:
#   gen_multipos ──→ train_multipos ──→ validate_multipos
#
# Usage:
#   bash submit_multipos_5dig.sh

set -euo pipefail

echo "=========================================="
echo "Multi-Position Projection Experiment (5-digit, 5M data, k=5)"
echo "=========================================="

# --- Step 1: Generate data (new tokenization with k=5 positions) ---

GEN=$(sbatch --parsable run_multipos_gen.slurm)
echo "Data generation:       ${GEN}"

# --- Step 2: Train (after data is ready) ---

TRAIN=$(sbatch --parsable \
  --dependency=afterok:${GEN} \
  run_fe_train_multipos.slurm)
echo "Multipos training:     ${TRAIN}  (after gen ${GEN})"

# --- Step 3: Validate (after training is done) ---

VAL=$(sbatch --parsable \
  --dependency=afterok:${TRAIN} \
  run_validate_multipos.slurm)
echo "Multipos validation:   ${VAL}  (after train ${TRAIN})"

echo ""
echo "=========================================="
echo "Pipeline submitted!"
echo "  gen ${GEN} -> train ${TRAIN} -> val ${VAL}"
echo "=========================================="
