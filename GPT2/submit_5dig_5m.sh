#!/bin/bash
# Full 5-digit pipeline with 5M training data for both FE-SME and base GPT-2.
#
# Pipeline:
#   gen_sme ──→ train_sme ──→ val_sme
#   gen_base ──→ train_base ──→ val_base
#
# Data gen jobs run in parallel; training starts after respective gen;
# validation starts after respective training.
#
# Usage:
#   bash submit_5dig_5m.sh

set -euo pipefail

DATA_ROOT="/tmpdir/m24047brmn/numbers/data"
MODEL_ROOT="/tmpdir/m24047brmn/numbers/model_checkpoints"

N_TRAIN=5000000
N_VAL=10000

echo "=========================================="
echo "5-digit 5M-data pipeline"
echo "  N_TRAIN: ${N_TRAIN}"
echo "  N_VAL:   ${N_VAL}"
echo "=========================================="

# --- 1. Generate data (both in parallel) ---

GEN_SME=$(sbatch --parsable \
  --job-name=gen-sme-5m \
  --time=12:00:00 \
  --output=slurm_logs/gen_sme_5dig_5m_%j.log \
  --error=slurm_logs/gen_sme_5dig_5m_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},OUT_DIR=${DATA_ROOT}/numtasks_sme_5dig_5m \
  run_generate_data_sme_5dig.slurm)
echo "SME data gen:    ${GEN_SME}"

GEN_BASE=$(sbatch --parsable \
  --job-name=gen-base-5m \
  --time=12:00:00 \
  --output=slurm_logs/gen_base_5dig_5m_%j.log \
  --error=slurm_logs/gen_base_5dig_5m_%j.log \
  --export=ALL,N_TRAIN=${N_TRAIN},N_VAL=${N_VAL},OUT_DIR=${DATA_ROOT}/numtasks_base_5dig_5m \
  run_generate_data_base_5dig.slurm)
echo "Base data gen:   ${GEN_BASE}"

# --- 2. Train (after respective data gen) ---

TRAIN_SME=$(sbatch --parsable \
  --dependency=afterok:${GEN_SME} \
  --job-name=sme-5m-train \
  --time=36:00:00 \
  --output=slurm_logs/gpt2_sme_5dig_5m_%j.log \
  --error=slurm_logs/gpt2_sme_5dig_5m_%j.log \
  --export=ALL,MAX_ITERS=35000,DATASET=numtasks_sme_5dig_5m,DATA_DIR=${DATA_ROOT}/numtasks_sme_5dig_5m,OUT_DIR=${MODEL_ROOT}/sme_5dig_5m \
  run_fe_train_sme_5dig.slurm)
echo "SME training:    ${TRAIN_SME}  (after gen ${GEN_SME})"

TRAIN_BASE=$(sbatch --parsable \
  --dependency=afterok:${GEN_BASE} \
  --job-name=base-5m-train \
  --time=36:00:00 \
  --output=slurm_logs/gpt2_base_5dig_5m_%j.log \
  --error=slurm_logs/gpt2_base_5dig_5m_%j.log \
  --export=ALL,MAX_ITERS=35000,DATASET=numtasks_base_5dig_5m,DATA_DIR=${DATA_ROOT}/numtasks_base_5dig_5m,OUT_DIR=${MODEL_ROOT}/base_5dig_5m \
  run_base_train_5dig.slurm)
echo "Base training:   ${TRAIN_BASE}  (after gen ${GEN_BASE})"

# --- 3. Validate ckpt_best.pt (after respective training) ---

VAL_SME=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_SME} \
  --job-name=val-sme-5m \
  --output=slurm_logs/validate_sme_5dig_5m_%j.log \
  --error=slurm_logs/validate_sme_5dig_5m_%j.log \
  --export=ALL,CKPT_PATH=${MODEL_ROOT}/sme_5dig_5m/ckpt_best.pt,DATA_DIR=${DATA_ROOT}/numtasks_sme_5dig_5m,OUTPUT_JSON=${MODEL_ROOT}/sme_5dig_5m/val_report_best.json \
  run_sme_validate_5dig.slurm)
echo "SME validation:  ${VAL_SME}  (after train ${TRAIN_SME})"

VAL_BASE=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_BASE} \
  --job-name=val-base-5m \
  --output=slurm_logs/validate_base_5dig_5m_%j.log \
  --error=slurm_logs/validate_base_5dig_5m_%j.log \
  --export=ALL,CKPT_PATH=${MODEL_ROOT}/base_5dig_5m/ckpt_best.pt,DATA_DIR=${DATA_ROOT}/numtasks_base_5dig_5m,OUTPUT_JSON=${MODEL_ROOT}/base_5dig_5m/val_report_best.json \
  run_base_validate_5dig.slurm)
echo "Base validation: ${VAL_BASE}  (after train ${TRAIN_BASE})"

echo ""
echo "=========================================="
echo "Pipeline submitted!"
echo "  SME:  gen ${GEN_SME} -> train ${TRAIN_SME} -> val ${VAL_SME}"
echo "  Base: gen ${GEN_BASE} -> train ${TRAIN_BASE} -> val ${VAL_BASE}"
echo "=========================================="
