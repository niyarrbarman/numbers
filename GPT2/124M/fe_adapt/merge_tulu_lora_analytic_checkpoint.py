"""Merge an intermediate analytic LoRA checkpoint into a standalone checkpoint.

Usage:
  python merge_tulu_lora_analytic_checkpoint.py \
    --input_ckpt /path/to/ckpt_iter2000.pt \
    --output_ckpt /path/to/ckpt_iter2000_merged.pt
"""

import argparse
import os
import sys
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_analytic import NemotronAnalyticConfig, NemotronAnalytic


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper matching train_tulu_lora_analytic.py."""

    def __init__(self, original: nn.Linear, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in original.parameters():
            p.requires_grad = False

    def forward(self, x):
        result = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return result + lora_out * self.scaling


def apply_lora(model, rank=16, alpha=32.0, dropout=0.0, target_modules=None):
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']
    count = 0
    for _, module in model.named_modules():
        for target_name in target_modules:
            if hasattr(module, target_name):
                original = getattr(module, target_name)
                if isinstance(original, nn.Linear):
                    setattr(module, target_name,
                            LoRALinear(original, rank, alpha, dropout))
                    count += 1
    return count


def merge_lora_weights(model):
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.original.weight.data += (
                        child.lora_B.T @ child.lora_A.T * child.scaling
                    ).to(child.original.weight.dtype)
                setattr(module, child_name, child.original)


def strip_orig_mod_prefix(state_dict):
    for key in list(state_dict.keys()):
        if key.startswith('_orig_mod.'):
            state_dict[key[len('_orig_mod.'):]] = state_dict.pop(key)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='Merge analytic LoRA checkpoint')
    parser.add_argument('--input_ckpt', required=True)
    parser.add_argument('--output_ckpt', required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input_ckpt, map_location='cpu', weights_only=False)
    state_dict = strip_orig_mod_prefix(dict(ckpt['model']))
    model_args = dict(ckpt['model_args'])
    nemconf = NemotronAnalyticConfig(**model_args)
    model = NemotronAnalytic(nemconf)

    lora_cfg = ckpt.get('lora_config')
    if lora_cfg is not None:
        apply_lora(
            model,
            rank=lora_cfg['rank'],
            alpha=lora_cfg['alpha'],
            dropout=lora_cfg['dropout'],
            target_modules=lora_cfg['target_modules'],
        )

    model.load_state_dict(state_dict)
    if lora_cfg is not None:
        merge_lora_weights(model)

    out_dir = os.path.dirname(args.output_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save({
        'model': model.state_dict(),
        'model_args': model_args,
        'config': ckpt.get('config', {}),
        'source_checkpoint': args.input_ckpt,
    }, args.output_ckpt)
    print(f"Merged checkpoint written to {args.output_ckpt}")


if __name__ == '__main__':
    main()
