# fe_adapt: NumberEncoder v10 → Baby Luciole (114M Nemotron3)

## Overview

Integrate pretrained NumberEncoder v10 into pretrained Baby Luciole (114M Nemotron3)
via LLaVA-style adapter injection. The approach is identical to the GPT-2 PLAN.md
but targets a different base model architecture.

## Why Baby Luciole?

- Already pretrained on FineWeb-Edu (21K steps, 6 nodes × 2 A100s)
- Same hidden_size=768 as GPT-2 124M → adapter MLP is identical (128→768→768)
- Same vocab (GPT-2 tokenizer, 50256 tokens) → data pipeline reusable as-is
- More modern architecture (GQA, RoPE, RMSNorm) — better baseline

## Architecture

Baby Luciole (Nemotron3 114M):
- 12 layers, hidden=768, 24 Q heads, 8 KV heads (GQA)
- FFN hidden=3072, squared ReLU activation
- RMSNorm (no bias), RoPE (no learned position embeddings)
- vocab_size=50256, shared embedding/output weights

NumberEncoder v10 (frozen, 6.2K params):
- 128d output: Scale(16) + Residue(22) + Semantic(90)
- Float64 residue lane for 1B-range precision

Adapter MLP (trainable, ~590K params):
- 128 → 768 (GELU) → 768
- Blend schedule: β ramps 0→1 (base <NUM> → adapter output)

## Pipeline

```
1. Convert NeMo checkpoint → simple PyTorch state dict
   (convert_nemo_ckpt.py, runs inside NeMo container)

2. Build standalone Nemotron3 in pure PyTorch (model.py)
   - No NeMo/Megatron dependency for training
   - Load converted weights + NumberEncoder checkpoint

3. Generate numerical task data (reuse fe/generate_data.py)
   - Same format: {split}.bin (uint16) + {split}_nums.bin (float32)

4. Train adapter (train.py)
   - Stage 1: Freeze base model, train adapter MLP only
   - Stage 2: Add LoRA to attention, train adapter + LoRA
```

## Key Arch Differences from GPT-2

| Component          | GPT-2 124M        | Baby Luciole 114M |
|-------------------|--------------------|-------------------|
| Attention          | MHA (12 heads)     | GQA (24Q / 8KV)  |
| Normalization      | LayerNorm + bias   | RMSNorm, no bias  |
| Position encoding  | Learned (wpe)      | RoPE              |
| FFN activation     | GELU               | Squared ReLU      |
| FFN structure      | 768→3072→768       | 768→3072→768      |
| Weight tying       | Yes                | Yes               |
| Bias               | Yes                | No                |

## NeMo Checkpoint Conversion

The pretrained checkpoint uses NeMo/Megatron-Core distributed checkpointing.
Key mapping (Megatron → standalone):

```
model.embedding.word_embeddings.weight      → transformer.wte.weight
model.decoder.layers.{i}.self_attention.linear_qkv.weight → split into q/k/v_proj
model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight → attn_norm.weight
model.decoder.layers.{i}.self_attention.linear_proj.weight → o_proj.weight
model.decoder.layers.{i}.mlp.linear_fc1.weight → up_proj.weight
model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight → ffn_norm.weight
model.decoder.layers.{i}.mlp.linear_fc2.weight → down_proj.weight
model.decoder.final_layernorm.weight → ln_f.weight
```

QKV split for GQA (8 groups, 3 Q heads/group):
- Per group: [Q0,Q1,Q2 (96d), K (32d), V (32d)] = 160d
- Total: 8 × 160 = 1280d → split into Q(768), K(256), V(256)

## Vocab Size

Baby Luciole: vocab_size=50256 (no EOT/padding during NeMo pretraining)
Our training: vocab_size=50304 (50256 base + EOT(50256) + NUM(50257) + padding)
→ Expand embedding by 48 rows (random init for new tokens)

## Training Stages (same as GPT-2 plan)

### Stage 1: Adapter Alignment
- Frozen: Base Nemotron (114M), NumberEncoder (6.2K)
- Trainable: Adapter MLP (~590K)
- β ramp: 0→1 over 20K iters
- LR: 1e-3 for adapter
- Duration: ~20K iters

### Stage 2: LoRA Fine-Tune
- Trainable: Adapter MLP + LoRA on Q/V projections (rank 8-16)
- β: fixed 1.0
- LR: 2e-4
- Duration: ~30K iters

## SLURM Environment

Uses Apptainer with NeMo container (for conversion step):
- Container: /work/conteneurs/calmip/nemo_25.04.03_arm.sif
- PYTHONUSERBASE: ${MYENVS}/nemo

Training can run with standard PyTorch (no NeMo dependency):
- 1-2 A100 GPUs
- Same SLURM partition as encoder training

## Files

- `PLAN_LUCIOLE.md` — This plan
- `model.py` — Standalone Nemotron3 + NumberEncoder adapter
- `convert_nemo_ckpt.py` — NeMo checkpoint → PyTorch state dict
- `train.py` — Training loop (adapted from fe/train.py)
- `prepare.py` — Tokenization (symlink to fe/prepare.py)
- `generate_data.py` — Task generation (symlink to fe/generate_data.py)
- `run_convert.sh` — SLURM script for checkpoint conversion
- `run_adapt.sh` — SLURM script for adapter training

## Checkpoints

- Base model: /tmpdir/m24047brmn/nemo_1b/output/baby_luciole-softmax-test/checkpoints/baby_luciole-softmax-test-step=0020998-last
- NumberEncoder v10: checkpoints/np_emb_v10_2000k_model.pt (after training completes)
