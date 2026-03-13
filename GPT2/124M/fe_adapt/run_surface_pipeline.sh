#!/bin/bash
#
# Full surface-decoder pipeline:
#   1. Stage 1 surface data generation
#   2. Stage 1 surface training
#   3. Stage 2 synth data generation
#   4. Stage 2 surface LoRA training
#   5. Merge best adapted checkpoint
#   6. Zero-shot synth benchmark vs existing base checkpoint
#
set -euo pipefail
mkdir -p slurm

SCRIPT_DIR="/work/m24047/m24047brmn/numbers/GPT2/124M/fe_adapt"
IMAGE="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"

PRETRAINED_CKPT="/tmpdir/m24047brmn/numbers/checkpoints/baby_luciole_converted.pt"
BASE_CKPT="/tmpdir/m24047brmn/numbers/model_checkpoints/analytic_s2_base_lora/ckpt_merged.pt"

S1_DATA_DIR="/tmpdir/m24047brmn/numbers/data/surface_stage1"
S1_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_surface_s1"

S2_DATA_DIR="/tmpdir/m24047brmn/numbers/data/synth_arith_surface"
S2_BASE_DATA="${S2_DATA_DIR}/base"
S2_ADAPTED_DATA="${S2_DATA_DIR}/adapted"
S2_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/surface_s2_adapted_lora"
BENCH_OUT_DIR="/tmpdir/m24047brmn/numbers/model_checkpoints/surface_synth_bench"
BENCH_JSON="${BENCH_OUT_DIR}/surface_synth_benchmark.json"
MERGED_BEST_CKPT="${BENCH_OUT_DIR}/surface_best_merged.pt"

N_GEN_SAMPLES=3000
N_FORWARD_BATCHES=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-ckpt)
            BASE_CKPT="$2"
            shift 2
            ;;
        --pretrained-ckpt)
            PRETRAINED_CKPT="$2"
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

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "Missing pretrained checkpoint: ${PRETRAINED_CKPT}" >&2
    exit 1
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "Missing base benchmark checkpoint: ${BASE_CKPT}" >&2
    exit 1
fi

APPTAINER="apptainer exec --nv \
  --env PYTHONUSERBASE=${MYENVS}/numbers \
  --env TIKTOKEN_CACHE_DIR=/tmpdir/m24047brmn/tiktoken_cache \
  --env PYTHONUNBUFFERED=1 \
  --bind /tmpdir,/work \
  ${IMAGE}"

echo "=========================================================="
echo "Surface Decoder Pipeline"
echo "  Pretrained ckpt: ${PRETRAINED_CKPT}"
echo "  Base ckpt:       ${BASE_CKPT}"
echo "  S1 data:         ${S1_DATA_DIR}"
echo "  S1 out:          ${S1_OUT_DIR}"
echo "  S2 data:         ${S2_DATA_DIR}"
echo "  S2 out:          ${S2_OUT_DIR}"
echo "  Bench out:       ${BENCH_OUT_DIR}"
echo "=========================================================="

JOB_S1_DATA=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J surf_s1_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_data_analytic_surface.py \
  --out_dir ${S1_DATA_DIR} \
  --n_train 100000 \
  --n_val 5000 \
  --n_test 3000 \
  --seed 42 \
  --surface_max_digits 32 \
  --surface_scale_min 0 \
  --surface_scale_max 32
EOF
)
echo "  [1] Stage 1 data: ${JOB_S1_DATA}"

JOB_S1_TRAIN=$(sbatch --parsable --dependency=afterok:${JOB_S1_DATA} <<EOF
#!/bin/bash
#SBATCH -J surf_s1_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_analytic_surface.py \
  init_from=pretrained \
  pretrained_ckpt=${PRETRAINED_CKPT} \
  data_dir=${S1_DATA_DIR} \
  out_dir=${S1_OUT_DIR} \
  numeric_output_mode=surface \
  surface_max_digits=32 \
  surface_scale_min=0 \
  surface_scale_max=32 \
  numeric_trunk_hidden=1024 \
  same_number_consistency_lambda=0.1 \
  same_number_digit_consistency_weight=2.0 \
  block_size=256 \
  batch_size=4 \
  gradient_accumulation_steps=40 \
  max_iters=15000 \
  learning_rate=1e-3 \
  adapter_lr_scale=1.0 \
  warmup_iters=1000 \
  lr_decay_iters=15000 \
  min_lr=1e-4 \
  num_loss_lambda=1.0 \
  eval_interval=2000 \
  diag_interval=100 \
  sample_interval=1000 \
  log_interval=10
EOF
)
echo "  [2] Stage 1 train: ${JOB_S1_TRAIN}"

JOB_S2_DATA=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J surf_s2_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/generate_synth_math_surface.py \
  --out_dir ${S2_DATA_DIR} \
  --n_train 50000 \
  --n_val 3000 \
  --n_test 3000 \
  --seed 42 \
  --analytic_adapted \
  --surface_max_digits 32 \
  --surface_scale_min 0 \
  --surface_scale_max 32
EOF
)
echo "  [3] Stage 2 data: ${JOB_S2_DATA}"

JOB_S2_TRAIN=$(sbatch --parsable --dependency=afterok:${JOB_S1_TRAIN}:${JOB_S2_DATA} <<EOF
#!/bin/bash
#SBATCH -J surf_s2_train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=18:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0

${APPTAINER} python3 ${SCRIPT_DIR}/train_tulu_lora_surface.py \
  stage1_ckpt=${S1_OUT_DIR}/ckpt_best.pt \
  data_dir=${S2_ADAPTED_DATA} \
  out_dir=${S2_OUT_DIR} \
  numeric_output_mode=surface \
  surface_max_digits=32 \
  surface_scale_min=0 \
  surface_scale_max=32 \
  numeric_trunk_hidden=1024 \
  same_number_consistency_lambda=0.1 \
  same_number_digit_consistency_weight=2.0 \
  block_size=256 \
  batch_size=8 \
  gradient_accumulation_steps=20 \
  lora_rank=16 \
  lora_alpha=32 \
  lora_dropout=0.05 \
  lora_targets=q_proj,v_proj,k_proj,o_proj \
  max_iters=10000 \
  learning_rate=3e-4 \
  lora_lr_scale=1.0 \
  adapter_lr_scale=0.3 \
  decoder_lr_scale=0.3 \
  warmup_iters=500 \
  lr_decay_iters=10000 \
  min_lr=3e-5 \
  decoder_warmup_iters=1000 \
  rollout_start_iter=1000 \
  rollout_every=20 \
  rollout_steps=4 \
  rollout_future_steps=8 \
  rollout_prefix_tokens=128 \
  rollout_loss_lambda=0.25 \
  rollout_consistency_lambda=0.25 \
  eval_interval=1000 \
  diag_interval=100 \
  sample_interval=500 \
  log_interval=10
EOF
)
echo "  [4] Stage 2 train: ${JOB_S2_TRAIN}"

JOB_BENCH=$(sbatch --parsable --dependency=afterok:${JOB_S2_TRAIN} <<EOF
#!/bin/bash
#SBATCH -J surf_bench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail
module load gnu/11.2.0
mkdir -p ${BENCH_OUT_DIR}

${APPTAINER} python3 ${SCRIPT_DIR}/merge_tulu_lora_surface_checkpoint.py \
  --input_ckpt ${S2_OUT_DIR}/ckpt_best.pt \
  --output_ckpt ${MERGED_BEST_CKPT}

${APPTAINER} python3 ${SCRIPT_DIR}/benchmark_synth_surface.py \
  --base_ckpt ${BASE_CKPT} \
  --adapted_ckpt ${MERGED_BEST_CKPT} \
  --base_data_dir ${S2_BASE_DATA} \
  --adapted_data_dir ${S2_ADAPTED_DATA} \
  --n_forward_batches ${N_FORWARD_BATCHES} \
  --n_gen_samples ${N_GEN_SAMPLES} \
  --out_path ${BENCH_JSON}
EOF
)
echo "  [5] Benchmark:    ${JOB_BENCH}"

echo
echo "Submitted pipeline:"
echo "  [1] Stage 1 data  -> ${JOB_S1_DATA}"
echo "  [2] Stage 1 train -> ${JOB_S1_TRAIN}"
echo "  [3] Stage 2 data  -> ${JOB_S2_DATA}"
echo "  [4] Stage 2 train -> ${JOB_S2_TRAIN}"
echo "  [5] Benchmark     -> ${JOB_BENCH}"
echo
echo "Dependencies:"
echo "  [1] -> [2]"
echo "  [2] + [3] -> [4] -> [5]"
echo
echo "Monitor with: squeue -u \$USER"
