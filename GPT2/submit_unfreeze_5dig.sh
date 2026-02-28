#!/bin/bash
# Submit both unfrozen encoder variants in parallel.
# Uses existing 5M SME data (numtasks_sme_5dig_5m).
#
# Pipeline:
#   train_unfreeze ──→ validate_unfreeze
#   train_unfreeze_mlp ──→ validate_unfreeze_mlp
#
# Usage:
#   bash submit_unfreeze_5dig.sh

set -euo pipefail

echo "=========================================="
echo "Unfrozen Encoder Experiments (5-digit, 5M data)"
echo "=========================================="

# --- Variant 1: Unfrozen encoder, same adapter ---

TRAIN_UF=$(sbatch --parsable run_fe_train_unfreeze.slurm)
echo "Unfreeze training:     ${TRAIN_UF}"

VAL_UF=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_UF} \
  run_validate_unfreeze.slurm)
echo "Unfreeze validation:   ${VAL_UF}  (after train ${TRAIN_UF})"

# --- Variant 2: Unfrozen encoder + wider MLP adapter ---

TRAIN_MLP=$(sbatch --parsable run_fe_train_unfreeze_mlp.slurm)
echo "Unfreeze+MLP training: ${TRAIN_MLP}"

VAL_MLP=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_MLP} \
  run_validate_unfreeze_mlp.slurm)
echo "Unfreeze+MLP valid:    ${VAL_MLP}  (after train ${TRAIN_MLP})"

echo ""
echo "=========================================="
echo "Pipeline submitted!"
echo "  Unfreeze:     train ${TRAIN_UF} -> val ${VAL_UF}"
echo "  Unfreeze+MLP: train ${TRAIN_MLP} -> val ${VAL_MLP}"
echo "=========================================="
