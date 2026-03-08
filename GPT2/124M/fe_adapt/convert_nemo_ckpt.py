#!/usr/bin/env python3
"""
Convert NeMo Nemotron3 checkpoint to simple PyTorch state dict.

Runs INSIDE the NeMo Apptainer container. Extracts model weights from the
distributed checkpoint and saves a flat PyTorch state dict with our key names.

Usage (inside container):
  python convert_nemo_ckpt.py \
    --nemo-ckpt /path/to/baby_luciole-softmax-test-step=0020998-last \
    --output /path/to/baby_luciole_converted.pt

Key mapping:
  Megatron-Core (NeMo)                              → Standalone
  model.embedding.word_embeddings.weight             → transformer.wte.weight
  model.decoder.layers.{i}.self_attention.linear_qkv → split q/k/v_proj
  model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight → attn_norm
  model.decoder.layers.{i}.self_attention.linear_proj → o_proj
  model.decoder.layers.{i}.mlp.linear_fc1           → up_proj
  model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight → ffn_norm
  model.decoder.layers.{i}.mlp.linear_fc2           → down_proj
  model.decoder.final_layernorm.weight               → ln_f.weight
"""

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

import torch


# Baby Luciole architecture constants
N_LAYERS = 12
N_HEAD = 24        # query heads
N_KV_HEAD = 8      # KV heads
HEAD_DIM = 32      # 768 / 24
N_REP = N_HEAD // N_KV_HEAD  # 3 queries per KV group


def split_qkv_weight(fused_weight):
    """Split Megatron-Core fused QKV weight into separate Q, K, V.

    Megatron-Core GQA layout (per KV group):
      [Q0, Q1, Q2 (3 heads × 32d = 96d), K (32d), V (32d)] = 160d
    Total: 8 groups × 160d = 1280d

    Returns:
        q_weight: (768, hidden)
        k_weight: (256, hidden)
        v_weight: (256, hidden)
    """
    group_size = (N_REP + 2) * HEAD_DIM  # (3 + 2) * 32 = 160

    q_chunks, k_chunks, v_chunks = [], [], []
    for g in range(N_KV_HEAD):
        offset = g * group_size
        q_end = offset + N_REP * HEAD_DIM     # +96
        k_end = q_end + HEAD_DIM               # +32
        v_end = k_end + HEAD_DIM               # +32

        q_chunks.append(fused_weight[offset:q_end])
        k_chunks.append(fused_weight[q_end:k_end])
        v_chunks.append(fused_weight[k_end:v_end])

    q = torch.cat(q_chunks, dim=0)  # (768, hidden)
    k = torch.cat(k_chunks, dim=0)  # (256, hidden)
    v = torch.cat(v_chunks, dim=0)  # (256, hidden)
    return q, k, v


def convert_state_dict(megatron_state):
    """Convert Megatron-Core state dict to our standalone format."""
    converted = {}

    for key, value in megatron_state.items():
        # Strip common prefixes
        k = key
        for prefix in ['model.module.', 'module.', 'model.']:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break

        # Embedding
        if k == 'embedding.word_embeddings.weight':
            converted['transformer.wte.weight'] = value
            # Weight tying: lm_head.weight = wte.weight (handled by model)
            continue

        # Output projection (if not tied)
        if k == 'output_layer.weight':
            # Usually tied with wte — skip if same shape
            converted['lm_head.weight'] = value
            continue

        # Final layer norm
        if k == 'decoder.final_layernorm.weight':
            converted['transformer.ln_f.weight'] = value
            continue

        # Transformer layers
        layer_match = re.match(r'decoder\.layers\.(\d+)\.(.*)', k)
        if not layer_match:
            print(f"  unmapped: {key}")
            continue

        layer_idx = int(layer_match.group(1))
        rest = layer_match.group(2)
        prefix = f'transformer.h.{layer_idx}'

        # Self-attention QKV (fused) → split into q/k/v_proj
        if rest == 'self_attention.linear_qkv.weight':
            q, k_w, v = split_qkv_weight(value)
            converted[f'{prefix}.attn.q_proj.weight'] = q
            converted[f'{prefix}.attn.k_proj.weight'] = k_w
            converted[f'{prefix}.attn.v_proj.weight'] = v
            continue

        if rest == 'self_attention.linear_qkv.bias':
            q_b, k_b, v_b = split_qkv_weight(value.unsqueeze(-1))
            converted[f'{prefix}.attn.q_proj.bias'] = q_b.squeeze(-1)
            converted[f'{prefix}.attn.k_proj.bias'] = k_b.squeeze(-1)
            converted[f'{prefix}.attn.v_proj.bias'] = v_b.squeeze(-1)
            continue

        # Pre-attention norm (fused into linear_qkv in Megatron)
        if rest == 'self_attention.linear_qkv.layer_norm_weight':
            converted[f'{prefix}.attn_norm.weight'] = value
            continue

        # Output projection
        if rest == 'self_attention.linear_proj.weight':
            converted[f'{prefix}.attn.o_proj.weight'] = value
            continue
        if rest == 'self_attention.linear_proj.bias':
            converted[f'{prefix}.attn.o_proj.bias'] = value
            continue

        # FFN up projection
        if rest == 'mlp.linear_fc1.weight':
            converted[f'{prefix}.mlp.up_proj.weight'] = value
            continue
        if rest == 'mlp.linear_fc1.bias':
            converted[f'{prefix}.mlp.up_proj.bias'] = value
            continue

        # Pre-FFN norm (fused into linear_fc1 in Megatron)
        if rest == 'mlp.linear_fc1.layer_norm_weight':
            converted[f'{prefix}.ffn_norm.weight'] = value
            continue

        # FFN down projection
        if rest == 'mlp.linear_fc2.weight':
            converted[f'{prefix}.mlp.down_proj.weight'] = value
            continue
        if rest == 'mlp.linear_fc2.bias':
            converted[f'{prefix}.mlp.down_proj.bias'] = value
            continue

        print(f"  unmapped: {key} (rest={rest})")

    return converted


def load_nemo_checkpoint(ckpt_path):
    """Load NeMo distributed checkpoint and extract model state dict.

    Uses PyTorch DCP (distributed checkpoint) with a single-process group.
    NeMo stores weights in DCP format under <ckpt>/weights/, which cannot
    be loaded with plain torch.load — it needs torch.distributed.checkpoint.
    """
    ckpt_path = Path(ckpt_path)

    # If it's a plain .pt file, load directly
    if ckpt_path.is_file():
        print(f"Loading plain checkpoint file: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            return ckpt['state_dict']
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        return ckpt

    # --- DCP loading for NeMo distributed checkpoints ---
    weights_dir = ckpt_path / "weights"
    if not weights_dir.exists():
        weights_dir = ckpt_path / "model"
    if not weights_dir.exists():
        print(f"ERROR: No weights/ or model/ directory in {ckpt_path}")
        print(f"  Contents: {list(ckpt_path.iterdir())}")
        sys.exit(1)

    print(f"Loading via PyTorch DCP from {weights_dir}...")

    # List shard files for diagnostics
    shard_files = list(weights_dir.glob("*.distcp")) or list(weights_dir.glob("*.pt"))
    print(f"  Found {len(shard_files)} shard file(s)")

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    # Initialize single-process distributed group (required by DCP)
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    if not dist.is_initialized():
        dist.init_process_group('gloo')
        print("  Initialized single-process distributed group (gloo)")

    # Read metadata to discover tensor names and shapes
    reader = FileSystemReader(str(weights_dir))
    metadata = reader.read_metadata()
    print(f"  Metadata: {len(metadata.state_dict_metadata)} entries")

    # Build empty state dict template from metadata
    state_dict = {}
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if hasattr(tensor_meta, 'size'):
            state_dict[key] = torch.empty(tensor_meta.size)

    # Load weights into the template
    dcp.load(state_dict, storage_reader=reader)

    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"  Loaded {len(state_dict)} tensors via DCP")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoint to PyTorch")
    parser.add_argument("--nemo-ckpt", type=str, required=True,
                        help="Path to NeMo checkpoint directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    args = parser.parse_args()

    print("=" * 60)
    print("NeMo → PyTorch Checkpoint Converter")
    print("=" * 60)
    print(f"  Input:  {args.nemo_ckpt}")
    print(f"  Output: {args.output}")
    print()

    # Load
    megatron_state = load_nemo_checkpoint(args.nemo_ckpt)
    print(f"\nMegatron state dict: {len(megatron_state)} keys")
    for k in sorted(megatron_state.keys())[:20]:
        print(f"  {k}: {megatron_state[k].shape}")
    if len(megatron_state) > 20:
        print(f"  ... ({len(megatron_state) - 20} more)")
    print()

    # Convert
    converted = convert_state_dict(megatron_state)
    print(f"\nConverted state dict: {len(converted)} keys")
    for k in sorted(converted.keys()):
        print(f"  {k}: {converted[k].shape}")

    # Verify expected shapes
    print("\nVerification:")
    expected = {
        'transformer.wte.weight': (50256, 768),
        'transformer.ln_f.weight': (768,),
    }
    for i in range(N_LAYERS):
        expected[f'transformer.h.{i}.attn.q_proj.weight'] = (768, 768)
        expected[f'transformer.h.{i}.attn.k_proj.weight'] = (256, 768)
        expected[f'transformer.h.{i}.attn.v_proj.weight'] = (256, 768)
        expected[f'transformer.h.{i}.attn.o_proj.weight'] = (768, 768)
        expected[f'transformer.h.{i}.attn_norm.weight'] = (768,)
        expected[f'transformer.h.{i}.mlp.up_proj.weight'] = (3072, 768)
        expected[f'transformer.h.{i}.mlp.down_proj.weight'] = (768, 3072)
        expected[f'transformer.h.{i}.ffn_norm.weight'] = (768,)

    ok = True
    for name, shape in expected.items():
        if name not in converted:
            print(f"  MISSING: {name}")
            ok = False
        elif converted[name].shape != torch.Size(shape):
            print(f"  WRONG SHAPE: {name}: {converted[name].shape} (expected {shape})")
            ok = False

    if ok:
        print(f"  All {len(expected)} expected keys present with correct shapes")
    else:
        print("  WARNING: Some keys missing or wrong — check conversion logic")

    # Save
    torch.save({'model_state_dict': converted}, args.output)
    print(f"\nSaved: {args.output}")

    # Total params
    total = sum(v.numel() for v in converted.values())
    print(f"Total parameters: {total:,} ({total / 1e6:.2f}M)")


if __name__ == '__main__':
    main()
