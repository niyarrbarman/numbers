#!/bin/bash
# Submit the full text-decode pipeline: generate → train → validate
# with automatic job dependencies.
#
# Usage:
#   bash submit_textdec_5dig.sh

set -euo pipefail

echo "=== Submitting text-decode 5-digit pipeline ==="

# 1. Data generation
GEN_JOB=$(sbatch --parsable run_textdec_gen.slurm)
echo "Submitted data gen job: ${GEN_JOB}"

# 2. Training (depends on data gen)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} run_fe_train_textdec.slurm)
echo "Submitted training job: ${TRAIN_JOB} (after ${GEN_JOB})"

# 3. Validation (depends on training)
VAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} run_validate_textdec.slurm)
echo "Submitted validation job: ${VAL_JOB} (after ${TRAIN_JOB})"

echo ""
echo "Pipeline submitted:"
echo "  gen:   ${GEN_JOB}"
echo "  train: ${TRAIN_JOB}"
echo "  val:   ${VAL_JOB}"
echo ""
echo "Monitor: squeue -u \$USER"
