"""
Stage 2: LoRA + Adapter Fine-Tuning for Baby Luciole + NumberEncoder v10.

Loads a Stage 1 checkpoint (with trained adapter and frozen LLM) and applies
LoRA (Low-Rank Adaptation) to transformer attention layers.  Trains both LoRA
and adapter parameters while keeping the encoder and base LLM weights frozen.
Beta is fixed at 1.0 (adapter already aligned from Stage 1).

LLaVA Stage 2 analogy:
  - Encoder (NumberEncoder v10):  FROZEN
  - Adapter (128d -> 768d MLP):  TRAINABLE (warm-started from Stage 1)
  - LLM base weights:            FROZEN
  - LoRA on attention layers:    TRAINABLE (new, zero-init so identity at start)

Usage:
  python train_stage2.py stage1_ckpt=/path/to/ckpt.pt data_dir=/path/to/data
  python train_stage2.py init_from=resume resume_ckpt=/path/to/s2_ckpt.pt data_dir=...
"""

import os
import sys
import time
import math
import re
import inspect
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

from model import NemotronConfig, Nemotron, NUM_TOKEN_ID
from prepare import EOT_TOKEN_ID


# =============================================================================
# LoRA Implementation
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Computes: y = W(x) + (dropout(x) @ A @ B) * (alpha / rank)
    W (original) is frozen.  Only A and B are trainable.
    B is zero-initialized so LoRA starts as identity (no perturbation).
    """

    def __init__(self, original: nn.Linear, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # A: down-project (in -> rank), Kaiming init
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: up-project (rank -> out), zero init -> starts as identity
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for p in original.parameters():
            p.requires_grad = False

    def forward(self, x):
        result = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return result + lora_out * self.scaling


def apply_lora(model, rank=16, alpha=32.0, dropout=0.0, target_modules=None):
    """Wrap specified nn.Linear attributes with LoRA across all sub-modules.

    Returns the number of layers wrapped.
    """
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
    """Merge LoRA A*B into base weights for efficient inference.  Irreversible."""
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    # W_merged = W + (B^T @ A^T) * scaling   [shapes: (out,in)]
                    child.original.weight.data += (
                        child.lora_B.T @ child.lora_A.T * child.scaling
                    ).to(child.original.weight.dtype)
                setattr(module, child_name, child.original)


# =============================================================================
# Config defaults -- Stage 2
# =============================================================================

# I/O
stage1_ckpt = ''   # REQUIRED for init_from='stage1'
out_dir = '/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_fe_adapt_s2'
eval_interval = 2000
log_interval = 10
diag_interval = 100
sample_interval = 1000
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'stage1'  # 'stage1' or 'resume'
resume_ckpt = ''       # explicit path for resume; if empty, uses out_dir/ckpt.pt

# data (same format as Stage 1)
data_dir = ''
batch_size = 4
block_size = 256
gradient_accumulation_steps = 80

# LoRA
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
lora_targets = 'q_proj,v_proj,k_proj,o_proj'  # comma-separated layer names

# optimizer
learning_rate = 3e-4
lora_lr_scale = 1.0
adapter_lr_scale = 0.3   # lower: adapter is already warm from Stage 1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
max_iters = 30000

# lr schedule
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 30000
min_lr = 3e-5

# number embedding
num_norm_match = True

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# wandb
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'luciole-114M-fe-s2-lora'

# DDP
backend = 'nccl'

# -----------------------------------------------------------------------------
# Parse config from command line (same as train.py)
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
for arg in sys.argv[1:]:
    if '=' in arg:
        key, val = arg.split('=', 1)
        key = key.lstrip('-')
        if key in config_keys:
            try:
                val = eval(val)
            except Exception:
                pass
            globals()[key] = val
config = {k: globals()[k] for k in config_keys}


# =============================================================================
# Setup
# =============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
ctx = (nullcontext() if device_type == 'cpu'
       else torch.amp.autocast(device_type=device_type, dtype=ptdtype))


# =============================================================================
# Data loader (same format as Stage 1)
# =============================================================================

assert data_dir, "data_dir is required"
print(f"data directory: {data_dir}")


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'),
                     dtype=np.uint16, mode='r')
    nums = np.memmap(os.path.join(data_dir, f'{split}_nums.bin'),
                     dtype=np.float32, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
                     for i in ix])
    nv = torch.stack([torch.from_numpy(nums[i:i + block_size].copy())
                      for i in ix])
    nm = (x == NUM_TOKEN_ID)
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        nv = nv.pin_memory().to(device, non_blocking=True)
        nm = nm.pin_memory().to(device, non_blocking=True)
    else:
        x, y, nv, nm = x.to(device), y.to(device), nv.to(device), nm.to(device)
    return x, y, nv, nm


# =============================================================================
# Model init
# =============================================================================

iter_num = 0
best_val_loss = 1e9
target_modules = [t.strip() for t in lora_targets.split(',')]

if init_from == 'stage1':
    assert stage1_ckpt, "stage1_ckpt is required when init_from='stage1'"
    print(f"Loading Stage 1 checkpoint: {stage1_ckpt}")
    checkpoint = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']

    # Don't re-load encoder from disk -- it is in the checkpoint state dict
    model_args['num_emb_checkpoint'] = ''
    # Allow overriding block_size for Stage 2
    model_args['block_size'] = block_size

    nemconf = NemotronConfig(**model_args)
    model = Nemotron(nemconf)

    # Load Stage 1 weights (strip _orig_mod. prefix from compiled model)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print(f"  Loaded Stage 1 weights ({len(state_dict)} keys)")

    # Apply LoRA AFTER loading (so base weights are set correctly first)
    n_lora = apply_lora(model, lora_rank, lora_alpha, lora_dropout, target_modules)
    print(f"  Applied LoRA (rank={lora_rank}, alpha={lora_alpha}, "
          f"dropout={lora_dropout}) to {n_lora} layers")
    print(f"  Target modules: {target_modules}")

    # Freeze: base transformer + encoder frozen, LoRA + adapter trainable
    for name, p in model.named_parameters():
        if 'lora_' in name:
            p.requires_grad = True
        elif name.startswith('num_adapter.'):
            p.requires_grad = True
        else:
            p.requires_grad = False

    checkpoint = None  # free memory

elif init_from == 'resume':
    ckpt_path = resume_ckpt if resume_ckpt else os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming Stage 2 from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    model_args['num_emb_checkpoint'] = ''
    model_args['block_size'] = block_size

    lora_cfg = checkpoint['lora_config']

    nemconf = NemotronConfig(**model_args)
    model = Nemotron(nemconf)

    # Apply LoRA BEFORE loading (state dict keys include lora_A/lora_B)
    n_lora = apply_lora(model, lora_cfg['rank'], lora_cfg['alpha'],
                        lora_cfg['dropout'], lora_cfg['target_modules'])

    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

    # Restore freeze pattern
    for name, p in model.named_parameters():
        if 'lora_' in name:
            p.requires_grad = True
        elif name.startswith('num_adapter.'):
            p.requires_grad = True
        else:
            p.requires_grad = False

    print(f"  Resumed from iter {iter_num}, best_val_loss {best_val_loss:.4f}")
    # checkpoint kept alive for optimizer loading below

else:
    raise ValueError(f"Unknown init_from: {init_from}. Use 'stage1' or 'resume'.")

# Print parameter budget
lora_p = adapter_p = frozen_p = 0
for name, p in model.named_parameters():
    n = p.numel()
    if 'lora_' in name:
        lora_p += n
    elif 'num_adapter.' in name:
        adapter_p += n
    else:
        frozen_p += n
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f"Stage 2 parameter budget:")
print(f"  LoRA:      {lora_p:>10,} params ({lora_p / total_p * 100:.2f}%)")
print(f"  Adapter:   {adapter_p:>10,} params ({adapter_p / total_p * 100:.2f}%)")
print(f"  Frozen:    {frozen_p:>10,} params")
print(f"  Trainable: {trainable_p:>10,} / {total_p:,} ({trainable_p / total_p * 100:.2f}%)")

model.to(device)

# GradScaler (only needed for float16; bfloat16 doesn't need it)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# =============================================================================
# Optimizer: LoRA + Adapter groups (everything else frozen)
# =============================================================================

lora_decay_params, lora_nodecay_params = [], []
adapter_decay_params, adapter_nodecay_params = [], []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    is_lora = 'lora_' in name
    is_adapter = name.startswith('num_adapter.')
    use_decay = p.dim() >= 2

    if is_lora:
        (lora_decay_params if use_decay else lora_nodecay_params).append(p)
    elif is_adapter:
        (adapter_decay_params if use_decay else adapter_nodecay_params).append(p)

optim_groups = []
group_specs = [
    ('lora_decay', lora_decay_params, weight_decay, lora_lr_scale),
    ('lora_nodecay', lora_nodecay_params, 0.0, lora_lr_scale),
    ('adapter_decay', adapter_decay_params, weight_decay, adapter_lr_scale),
    ('adapter_nodecay', adapter_nodecay_params, 0.0, adapter_lr_scale),
]
print("optimizer parameter groups:")
for gname, params, wd, lrs in group_specs:
    if not params:
        continue
    n_p = sum(p.numel() for p in params)
    print(f"  {gname}: {len(params)} tensors, {n_p:,} params, "
          f"weight_decay={wd}, lr_scale={lrs}")
    optim_groups.append({
        'params': params, 'weight_decay': wd,
        'lr': learning_rate * lrs, 'lr_scale': lrs,
        'group_name': gname,
    })

fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                              betas=(beta1, beta2), **extra_args)
print(f"using fused AdamW: {use_fused}")

if init_from == 'resume' and checkpoint is not None:
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("  Restored optimizer state from checkpoint")
    except (ValueError, KeyError) as e:
        print(f"WARNING: could not load optimizer state: {e}")
        print("WARNING: continuing with freshly initialized optimizer.")
    checkpoint = None  # free memory

# Compile
base_model = model  # keep reference to uncompiled model for final merge
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


# =============================================================================
# Helpers
# =============================================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    base_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, NV, NM = get_batch(split)
            with ctx:
                _, loss = base_model(X, Y, num_values=NV, num_mask=NM,
                                     num_blend_beta=1.0,
                                     num_norm_match=num_norm_match)
            losses[k] = loss.item()
        out[split] = losses.mean()
    base_model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def collect_grad_norms(named_params):
    """L2 grad norms split into lora / adapter / transformer."""
    group_sq = {}
    total_sq = 0.0
    for name, p in named_params:
        if p.grad is None:
            continue
        sq = p.grad.data.norm(2).item() ** 2
        total_sq += sq
        if 'lora_' in name:
            key = 'lora'
        elif 'num_adapter' in name or 'num_encoder' in name:
            key = 'adapter'
        else:
            key = 'transformer'
        group_sq[key] = group_sq.get(key, 0.0) + sq
    out = {k: v ** 0.5 for k, v in group_sq.items()}
    out['total'] = total_sq ** 0.5
    return out


@torch.no_grad()
def compute_output_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    valid = (targets >= 0) & (targets != EOT_TOKEN_ID)
    n = valid.sum().item()
    if n == 0:
        return None
    correct = (preds[valid] == targets[valid]).sum().item()
    return {'overall': correct / n, 'n_correct': correct, 'n_tokens': n}


def decode_context(token_ids, num_values, enc):
    parts = []
    for i, tok in enumerate(token_ids):
        if tok == NUM_TOKEN_ID:
            parts.append(f"<{num_values[i]:g}>")
        elif tok == EOT_TOKEN_ID:
            break
        else:
            try:
                parts.append(enc.decode([tok]))
            except Exception:
                parts.append(f"[{tok}]")
    return ''.join(parts)


_number_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


@torch.no_grad()
def eval_samples(max_samples=5):
    try:
        from prepare import _get_tokenizer
        enc = _get_tokenizer()
    except Exception:
        return

    base_model.eval()
    X, Y, NV, NM = get_batch('val')
    with ctx:
        logits, _ = base_model(X, Y, num_values=NV, num_mask=NM,
                               num_blend_beta=1.0,
                               num_norm_match=num_norm_match)
    base_model.train()

    preds = logits.argmax(dim=-1)
    print("  --- Sample eval (val) ---")
    n_shown = 0
    for b in range(X.size(0)):
        if n_shown >= max_samples:
            break
        x_row = X[b].tolist()
        y_row = Y[b].tolist()
        p_row = preds[b].tolist()
        nv_row = NV[b].tolist()

        eot_positions = [i for i, t in enumerate(x_row) if t == EOT_TOKEN_ID]
        if not eot_positions:
            eot_positions = [0]

        for seg_start_idx in range(len(eot_positions)):
            if n_shown >= max_samples:
                break
            seg_start = (eot_positions[seg_start_idx] + 1
                         if seg_start_idx > 0 else 0)
            T = len(x_row)
            seg_end = T
            for t in range(seg_start, T):
                if y_row[t] == EOT_TOKEN_ID:
                    seg_end = t
                    break
            if seg_end <= seg_start + 2:
                continue

            context = decode_context(x_row[seg_start:seg_end],
                                     nv_row[seg_start:seg_end], enc)
            if "\u2192" not in context:
                continue

            arrow_pos = None
            for t in range(seg_start, seg_end):
                tok_text = (enc.decode([x_row[t]])
                            if 0 <= x_row[t] < 50256 else "")
                if "\u2192" in tok_text:
                    arrow_pos = t
                    break
            if arrow_pos is None:
                continue

            out_start = arrow_pos + 1
            if out_start >= seg_end:
                continue

            tgt_ids = y_row[out_start:seg_end]
            pred_ids = p_row[out_start:seg_end]
            tgt_text = enc.decode([t for t in tgt_ids if 0 <= t < 50256])
            pred_text = enc.decode([t for t in pred_ids if 0 <= t < 50256])

            input_ctx = decode_context(x_row[seg_start:arrow_pos + 1],
                                       nv_row[seg_start:arrow_pos + 1], enc)
            print(f"  {input_ctx}")
            print(f"    target: {tgt_text.strip()}")
            print(f"    pred:   {pred_text.strip()}")

            tgt_nums = [float(v) for v in _number_re.findall(tgt_text)]
            pred_nums = [float(v) for v in _number_re.findall(pred_text)]
            if tgt_nums and pred_nums and len(tgt_nums) == len(pred_nums):
                errs = [f"{abs(t - p):.2f}" for t, p in
                        zip(tgt_nums, pred_nums)]
                print(f"    error:  {' '.join(errs)}")
            elif tgt_nums and len(tgt_nums) != len(pred_nums):
                print(f"    (count mismatch: target {len(tgt_nums)}, "
                      f"pred {len(pred_nums)})")
            print()
            n_shown += 1
            break  # one per batch row


@torch.no_grad()
def collect_num_injection_stats(num_values, num_mask):
    if num_mask is None or num_mask.sum().item() == 0:
        return None
    flat_vals = num_values[num_mask].float()
    num_emb = base_model.num_encoder(flat_vals)
    delta_raw = base_model.num_adapter(num_emb)
    n_num = delta_raw.size(0)
    base_vec = base_model.transformer.wte.weight[NUM_TOKEN_ID]
    base_vec = base_vec.unsqueeze(0).to(delta_raw.dtype)
    base = base_vec.expand(n_num, -1)
    base_norm = base.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta_raw_norm = delta_raw.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if num_norm_match:
        scale = base_norm / delta_raw_norm
        delta_eff = delta_raw * scale.to(delta_raw.dtype)
    else:
        scale = torch.ones_like(base_norm)
        delta_eff = delta_raw
    # beta=1.0 in Stage 2, so blended = delta_eff
    return {
        'base_norm': base_norm.mean().item(),
        'delta_raw_norm': delta_raw_norm.mean().item(),
        'delta_eff_norm': delta_eff.float().norm(dim=-1).mean().item(),
        'blend_norm': delta_eff.float().norm(dim=-1).mean().item(),
        'scale_mean': scale.mean().item(),
        'scale_max': scale.max().item(),
    }


def save_checkpoint(it, val_loss, is_best=False):
    """Save Stage 2 checkpoint with LoRA config."""
    ckpt = {
        'model': base_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'lora_config': {
            'rank': lora_rank, 'alpha': lora_alpha,
            'dropout': lora_dropout,
            'target_modules': target_modules,
        },
        'iter_num': it,
        'best_val_loss': val_loss,
        'config': config,
    }
    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
    torch.save(ckpt, os.path.join(out_dir, f'ckpt_iter{it}.pt'))
    print(f"  saved checkpoint to {out_dir}/ckpt_iter{it}.pt")
    if is_best:
        torch.save(ckpt, os.path.join(out_dir, 'ckpt_best.pt'))
        print(f"  new best val loss: {val_loss:.4f}")


# =============================================================================
# Training loop
# =============================================================================

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y, NV, NM = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

print(f"\nStarting Stage 2 training from iter {iter_num} "
      f"(max_iters={max_iters}, beta=1.0 fixed)")

while True:

    # LR schedule (cosine warmup + decay)
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr * float(pg.get('lr_scale', 1.0))
    lora_lr = lr * lora_lr_scale
    adapt_lr = lr * adapter_lr_scale

    # === Eval and checkpoint ===
    if iter_num % eval_interval == 0:
        if ddp:
            barrier()
        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                if iter_num > 0:
                    is_best = losses['val'] < best_val_loss
                    save_checkpoint(iter_num, best_val_loss, is_best)
                best_val_loss = min(best_val_loss, losses['val'])
        if ddp:
            barrier()
        if iter_num == 0 and eval_only:
            break

    # === Forward / backward with gradient accumulation ===
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(
                X, Y, num_values=NV, num_mask=NM,
                num_blend_beta=1.0,
                num_norm_match=num_norm_match,
            )
            loss = loss / gradient_accumulation_steps

        # snapshot for diagnostics (last micro-step only)
        if (micro_step == gradient_accumulation_steps - 1
                and iter_num % diag_interval == 0 and master_process):
            _diag_logits = logits.detach()
            _diag_targets = Y.clone()
            _diag_nv = NV.clone()
            _diag_nm = NM.clone()

        X, Y, NV, NM = get_batch('train')
        scaler.scale(loss).backward()

    # === Gradient clipping + diagnostics ===
    _pre_norms = _post_norms = None
    _clip_pre = _clip_coef = None

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if iter_num % diag_interval == 0 and master_process:
            _pre_norms = collect_grad_norms(base_model.named_parameters())
        pre_total = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   grad_clip)
        if iter_num % diag_interval == 0 and master_process:
            _clip_pre = float(pre_total.item()
                              if hasattr(pre_total, 'item') else pre_total)
            _clip_coef = min(1.0, grad_clip / max(_clip_pre, 1e-8))
    elif iter_num % diag_interval == 0 and master_process:
        _pre_norms = collect_grad_norms(base_model.named_parameters())
        _clip_pre = _pre_norms.get('total', 0.0)
        _clip_coef = 1.0

    if iter_num % diag_interval == 0 and master_process:
        _post_norms = collect_grad_norms(base_model.named_parameters())
        _num_inj = collect_num_injection_stats(_diag_nv, _diag_nm)

    # Step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # === Timing and logging ===
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = base_model.estimate_mfu(
                batch_size * gradient_accumulation_steps, dt)
            running_mfu = (mfu if running_mfu == -1.0
                           else 0.9 * running_mfu + 0.1 * mfu)
        print(f"iter {iter_num}: loss {lossf:.4f}, "
              f"time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    # === Detailed diagnostics ===
    if iter_num % diag_interval == 0 and master_process:
        g = _post_norms or {}
        g_pre = _pre_norms or {}

        num_count = (int(_diag_nm.sum().item())
                     if _diag_nm is not None else 0)
        total_tokens = batch_size * block_size

        print(f"  === DIAG iter {iter_num} ===")
        print(f"  loss: {loss.item() * gradient_accumulation_steps:.4f}")
        print(f"  grads (preclip): total {g_pre.get('total', 0):.4f}, "
              f"lora {g_pre.get('lora', 0):.4f}, "
              f"adapter {g_pre.get('adapter', 0):.4f}")
        print(f"  grads (postclip): total {g.get('total', 0):.4f}, "
              f"lora {g.get('lora', 0):.4f}, "
              f"adapter {g.get('adapter', 0):.4f}, "
              f"transformer {g.get('transformer', 0):.4f}")
        if _clip_pre is not None:
            print(f"  grad clip: max_norm {grad_clip:.4f}, "
                  f"pre_total {_clip_pre:.4f}, "
                  f"coef {_clip_coef:.4f}")
        print(f"  <NUM> tokens: {num_count}/{total_tokens} "
              f"({num_count / total_tokens * 100:.1f}%)")
        print(f"  lr: base {lr:.2e}, lora {lora_lr:.2e}, "
              f"adapter {adapt_lr:.2e}")

        if _num_inj is not None:
            print(f"  num inject: beta 1.0000, "
                  f"norm_match {str(bool(num_norm_match)).lower()}, "
                  f"base_norm {_num_inj['base_norm']:.4f}, "
                  f"delta_raw_norm {_num_inj['delta_raw_norm']:.4f}, "
                  f"delta_eff_norm {_num_inj['delta_eff_norm']:.4f}, "
                  f"blend_norm {_num_inj['blend_norm']:.4f}")

        out_acc = compute_output_accuracy(_diag_logits, _diag_targets)
        if out_acc is not None:
            print(f"  output token accuracy: {out_acc['overall']:.3f} "
                  f"({out_acc['n_correct']}/{out_acc['n_tokens']} tokens)")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "grad/total": g.get('total', 0),
                "grad/lora": g.get('lora', 0),
                "grad/adapter": g.get('adapter', 0),
                "grad_preclip/total": g_pre.get('total', 0),
                "grad_preclip/lora": g_pre.get('lora', 0),
                "grad_preclip/adapter": g_pre.get('adapter', 0),
                "lr": lr,
                "lr/lora": lora_lr,
                "lr/adapter": adapt_lr,
            }
            if _num_inj is not None:
                log_dict["num/delta_raw_norm"] = _num_inj["delta_raw_norm"]
                log_dict["num/scale_mean"] = _num_inj["scale_mean"]
            if out_acc is not None:
                log_dict["output/token_accuracy"] = out_acc['overall']
            wandb.log(log_dict)

    # === Sample evaluation ===
    if iter_num % sample_interval == 0 and iter_num > 0:
        if ddp:
            barrier()
        if master_process:
            eval_samples()
        if ddp:
            barrier()

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break


# =============================================================================
# Final save: unmerged + merged
# =============================================================================

if master_process:
    print("\n" + "=" * 60)
    print("Training complete. Saving final checkpoints...")

    # 1. Save unmerged (for further training / resume)
    save_checkpoint(iter_num, best_val_loss, is_best=False)
    print(f"  Unmerged checkpoint: {out_dir}/ckpt_iter{iter_num}.pt")

    # 2. Merge LoRA into base weights and save (for efficient inference)
    merge_lora_weights(base_model)
    merged_state = base_model.state_dict()
    torch.save({
        'model': merged_state,
        'model_args': model_args,
        'config': config,
    }, os.path.join(out_dir, 'ckpt_merged.pt'))
    print(f"  Merged checkpoint:   {out_dir}/ckpt_merged.pt")
    print("=" * 60)

if ddp:
    destroy_process_group()
