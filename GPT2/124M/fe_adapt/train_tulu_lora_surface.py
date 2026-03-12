"""Surface-decoder LoRA fine-tuning on synthetic arithmetic data.

Uses NemotronAnalytic with the surface-oriented numeric decoder:
  sign, scale, length, digits.

Loads a Stage 1 surface checkpoint, applies LoRA to attention layers, and
trains with both teacher-forced numeric supervision and short rollout loss.

Usage:
  python train_tulu_lora_surface.py \
    stage1_ckpt=.../ckpt.pt \
    data_dir=.../adapted \
    out_dir=.../adapted_lora

  # Resume
  python train_tulu_lora_surface.py init_from=resume resume_ckpt=...
"""

import os
import sys
import time
import math
import inspect
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

from model_analytic_surface import (
    NemotronAnalyticConfig,
    NemotronAnalytic,
    NUM_TOKEN_ID,
)
from numeric_surface import render_surface_row
from prepare import EOT_TOKEN_ID


# =============================================================================
# LoRA
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""

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


# =============================================================================
# Config defaults
# =============================================================================

# I/O
stage1_ckpt = ''      # Stage 1 surface checkpoint
out_dir = '/tmpdir/m24047brmn/numbers/model_checkpoints/surface_s2_adapted_lora'
eval_interval = 2000
log_interval = 10
diag_interval = 100
sample_interval = 1000
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'fresh'  # 'fresh' or 'resume'
resume_ckpt = ''

# data
data_dir = ''
batch_size = 4
block_size = 512
gradient_accumulation_steps = 40
numeric_output_mode = 'surface'
surface_max_digits = 32
surface_scale_min = 0
surface_scale_max = 32

# LoRA
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
lora_targets = 'q_proj,v_proj,k_proj,o_proj'

# optimizer
learning_rate = 3e-4
lora_lr_scale = 1.0
adapter_lr_scale = 0.3
decoder_lr_scale = 0.3
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
max_iters = 15000

# lr schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 15000
min_lr = 3e-5
decoder_warmup_iters = 1000

# rollout training
rollout_start_iter = 1000
rollout_every = 20
rollout_steps = 4
rollout_future_steps = 8
rollout_prefix_tokens = 128
rollout_loss_lambda = 0.25

# loss
num_loss_lambda = 1.0

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# wandb
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'tulu-lora-analytic'

# DDP
backend = 'nccl'

# -----------------------------------------------------------------------------
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
print(f"Mode: SURFACE ADAPTED (Stage1 Surface + LoRA)")
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
# Data loader
# =============================================================================

assert data_dir, "data_dir is required"
print(f"data directory: {data_dir}")
target_cols = None
target_suffix = 'surface' if numeric_output_mode == 'surface' else 'components'


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

    target_path = os.path.join(data_dir, f'{split}_{target_suffix}.bin')
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Missing numeric supervision file: {target_path}\n"
            "Surface stage-2 training requires adapted data generated with "
            "generate_synth_math_surface.py --analytic_adapted."
        )
    if target_cols is None:
        raise RuntimeError("target_cols not initialized before get_batch()")
    comps = np.memmap(target_path, dtype=np.uint8, mode='r')
    n_tokens = len(data)
    comps = comps.reshape(n_tokens, target_cols)
    nc = torch.stack([
        torch.from_numpy(comps[i + 1:i + 1 + block_size].copy())
        for i in ix
    ])  # aligned with y (shifted +1)

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        nv = nv.pin_memory().to(device, non_blocking=True)
        nm = nm.pin_memory().to(device, non_blocking=True)
        nc = nc.pin_memory().to(device, non_blocking=True)
    else:
        x, y, nv, nm, nc = (x.to(device), y.to(device), nv.to(device),
                             nm.to(device), nc.to(device))
    return x, y, nv, nm, nc


def numeric_target_kwargs(targets):
    if numeric_output_mode == 'surface':
        return {'num_target_surface': targets}
    return {'num_target_components': targets}


# =============================================================================
# Model init
# =============================================================================

iter_num = 0
best_val_loss = 1e9
target_modules = [t.strip() for t in lora_targets.split(',')]

if init_from == 'fresh':
    assert stage1_ckpt, "stage1_ckpt required"
    print(f"Loading Stage 1 surface checkpoint: {stage1_ckpt}")
    checkpoint = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    model_args['block_size'] = block_size
    model_args['numeric_output_mode'] = numeric_output_mode
    model_args['surface_max_digits'] = surface_max_digits
    model_args['surface_scale_min'] = surface_scale_min
    model_args['surface_scale_max'] = surface_scale_max

    nemconf = NemotronAnalyticConfig(**model_args)
    model = NemotronAnalytic(nemconf)

    state_dict = checkpoint['model']
    # Strip _orig_mod. prefix from compiled checkpoints
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print(f"  Loaded Stage 1 weights ({len(state_dict)} keys)")
    checkpoint = None

elif init_from == 'resume':
    ckpt_path = resume_ckpt if resume_ckpt else os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    model_args['block_size'] = block_size
    model_args['numeric_output_mode'] = model_args.get('numeric_output_mode', numeric_output_mode)
    model_args['surface_max_digits'] = model_args.get('surface_max_digits', surface_max_digits)
    model_args['surface_scale_min'] = model_args.get('surface_scale_min', surface_scale_min)
    model_args['surface_scale_max'] = model_args.get('surface_scale_max', surface_scale_max)

    lora_cfg = checkpoint['lora_config']
    nemconf = NemotronAnalyticConfig(**model_args)
    model = NemotronAnalytic(nemconf)

    # Apply LoRA BEFORE loading (state dict has lora keys)
    n_lora = apply_lora(model, lora_cfg['rank'], lora_cfg['alpha'],
                        lora_cfg['dropout'], lora_cfg['target_modules'])

    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"  Resumed iter {iter_num}, best_val_loss {best_val_loss:.4f}")

else:
    raise ValueError(f"Unknown init_from: {init_from}. Use 'fresh' or 'resume'.")

# Apply LoRA (for 'fresh' mode; already done in 'resume')
if init_from == 'fresh':
    n_lora = apply_lora(model, lora_rank, lora_alpha, lora_dropout, target_modules)
    print(f"  Applied LoRA (rank={lora_rank}, alpha={lora_alpha}, "
          f"dropout={lora_dropout}) to {n_lora} layers")
    print(f"  Target modules: {target_modules}")

# Freeze pattern: only LoRA, adapter, and decoder heads are trainable
for name, p in model.named_parameters():
    if 'lora_' in name:
        p.requires_grad = True
    elif name.startswith('num_adapter.'):
        p.requires_grad = True
    elif name.startswith('num_decoder_') or name.startswith('num_surface_'):
        p.requires_grad = True
    else:
        p.requires_grad = False

# Parameter budget
lora_p = adapter_p = decoder_p = frozen_p = 0
for name, p in model.named_parameters():
    n = p.numel()
    if 'lora_' in name:
        lora_p += n
    elif 'num_adapter.' in name:
        adapter_p += n
    elif name.startswith('num_decoder_') or name.startswith('num_surface_'):
        decoder_p += n
    else:
        frozen_p += n
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f"Parameter budget (SURFACE ADAPTED):")
print(f"  LoRA:      {lora_p:>10,} params ({lora_p / total_p * 100:.2f}%)")
print(f"  Adapter:   {adapter_p:>10,} params ({adapter_p / total_p * 100:.2f}%)")
print(f"  Decoder:   {decoder_p:>10,} params ({decoder_p / total_p * 100:.2f}%)")
print(f"  Frozen:    {frozen_p:>10,} params")
print(f"  Trainable: {trainable_p:>10,} / {total_p:,} ({trainable_p / total_p * 100:.2f}%)")

model.to(device)
target_cols = (
    3 + model.config.surface_max_digits
    if numeric_output_mode == 'surface'
    else 2 + model.config.analytic_K
)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))


# =============================================================================
# Optimizer
# =============================================================================

lora_decay_params, lora_nodecay_params = [], []
adapter_decay_params, adapter_nodecay_params = [], []
decoder_decay_params, decoder_nodecay_params = [], []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    is_lora = 'lora_' in name
    is_adapter = name.startswith('num_adapter.')
    is_decoder = name.startswith('num_decoder_') or name.startswith('num_surface_')
    use_decay = p.dim() >= 2

    if is_lora:
        (lora_decay_params if use_decay else lora_nodecay_params).append(p)
    elif is_adapter:
        (adapter_decay_params if use_decay else adapter_nodecay_params).append(p)
    elif is_decoder:
        (decoder_decay_params if use_decay else decoder_nodecay_params).append(p)

optim_groups = []
group_specs = [
    ('lora_decay', lora_decay_params, weight_decay, lora_lr_scale),
    ('lora_nodecay', lora_nodecay_params, 0.0, lora_lr_scale),
    ('adapter_decay', adapter_decay_params, weight_decay, adapter_lr_scale),
    ('adapter_nodecay', adapter_nodecay_params, 0.0, adapter_lr_scale),
    ('decoder_decay', decoder_decay_params, weight_decay, decoder_lr_scale),
    ('decoder_nodecay', decoder_nodecay_params, 0.0, decoder_lr_scale),
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
        print("  Restored optimizer state")
    except (ValueError, KeyError) as e:
        print(f"WARNING: could not load optimizer state: {e}")
    checkpoint = None

# Compile
base_model = model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


# =============================================================================
# Helpers
# =============================================================================

def validate_numeric_supervision(split, max_checks=2048):
    """Fail fast if numeric labels are missing or inconsistent."""
    data_path = os.path.join(data_dir, f'{split}.bin')
    nums_path = os.path.join(data_dir, f'{split}_nums.bin')
    target_path = os.path.join(data_dir, f'{split}_{target_suffix}.bin')
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Missing numeric supervision file: {target_path}\n"
            "Surface stage-2 training requires generate_synth_math_surface.py "
            "--analytic_adapted."
        )

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    nums = np.memmap(nums_path, dtype=np.float32, mode='r')
    expected_cols = target_cols
    comps = np.memmap(target_path, dtype=np.uint8, mode='r')
    if comps.size != len(data) * expected_cols:
        raise ValueError(
            f"{target_path} has {comps.size} values but expected "
            f"{len(data) * expected_cols} ({len(data)} x {expected_cols})."
        )
    comps = comps.reshape(len(data), expected_cols)

    num_positions = np.flatnonzero(data == NUM_TOKEN_ID)
    if num_positions.size == 0:
        raise ValueError(f"No <NUM> tokens found in {data_path}.")

    sample_positions = num_positions[:max_checks]
    mismatches = []
    for pos in sample_positions:
        want_value = float(nums[pos])
        got = comps[pos]
        if numeric_output_mode == 'surface':
            try:
                got_text = render_surface_row(
                    got,
                    max_digits=surface_max_digits,
                    scale_min=surface_scale_min,
                )
                got_value = float(got_text)
            except Exception:
                got_value = float('nan')
        else:
            got_value = float('nan')
        if want_value == 0.0:
            tol = 1e-6
        else:
            tol = max(
                1e-6,
                float(abs(np.spacing(np.float32(want_value)))) * 8.0,
                abs(want_value) * 1e-6,
            )
        if (not np.isfinite(got_value)
                or abs(got_value - want_value) > tol):
            mismatches.append((
                int(pos), want_value, got_value, tol, got[:6].tolist(),
            ))
            if len(mismatches) >= 3:
                break

    if mismatches:
        details = []
        for pos, want_value, got_value, tol, got in mismatches:
            details.append(
                f"pos={pos} want={want_value:g} got={got_value:g} "
                f"tol={tol:g} label={got}..."
            )
        raise ValueError(
            f"Invalid numeric supervision in {target_path}. Sample mismatches: "
            + "; ".join(details)
        )

    if master_process:
        print(f"Validated numeric supervision for {split}: "
              f"{len(sample_positions)} sampled <NUM> labels in sync")


validate_numeric_supervision('train')
validate_numeric_supervision('val')


@torch.no_grad()
def estimate_loss():
    out = {}
    base_model.eval()
    for split in ['train', 'val']:
        text_losses = torch.zeros(eval_iters)
        num_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, NV, NM, NC = get_batch(split)
            with ctx:
                logits, text_loss, num_loss_dict = base_model(
                    X, Y, num_values=NV, num_mask=NM,
                    **numeric_target_kwargs(NC),
                )
            text_losses[k] = text_loss.item() if text_loss is not None else 0.0
            if num_loss_dict is not None:
                num_losses[k] = num_loss_dict['total'].item()
        out[split] = {
            'text_loss': text_losses.mean().item(),
            'num_loss': num_losses.mean().item(),
            'total_loss': text_losses.mean().item() + num_loss_lambda * num_losses.mean().item(),
        }
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
    group_sq = {}
    total_sq = 0.0
    for name, p in named_params:
        if p.grad is None:
            continue
        sq = p.grad.data.norm(2).item() ** 2
        total_sq += sq
        if 'lora_' in name:
            key = 'lora'
        elif 'num_adapter' in name:
            key = 'adapter'
        elif 'num_decoder' in name or 'num_surface' in name:
            key = 'decoder'
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


@torch.no_grad()
def eval_samples(max_samples=3):
    try:
        from prepare import _get_tokenizer
        enc = _get_tokenizer()
    except Exception:
        return

    base_model.eval()
    X, Y, NV, NM, NC = get_batch('val')
    with ctx:
        logits, _, num_loss_dict = base_model(
            X, Y, num_values=NV, num_mask=NM,
            **numeric_target_kwargs(NC),
        )
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

        for seg_idx in range(len(eot_positions)):
            if n_shown >= max_samples:
                break
            seg_start = eot_positions[seg_idx] + 1 if seg_idx > 0 else 0
            seg_end = len(x_row)
            for t in range(seg_start, len(x_row)):
                if y_row[t] == EOT_TOKEN_ID:
                    seg_end = t
                    break
            if seg_end <= seg_start + 5:
                continue

            context = decode_context(x_row[seg_start:seg_end],
                                     nv_row[seg_start:seg_end], enc)

            assistant_pos = None
            for t in range(seg_start, seg_end):
                tok_text = enc.decode([x_row[t]]) if 0 <= x_row[t] < 50256 else ""
                if "Assistant" in tok_text:
                    assistant_pos = t
                    break

            if assistant_pos is None:
                continue

            out_start = assistant_pos + 1
            if out_start >= seg_end:
                continue

            ctx_text = context[:200] + ('...' if len(context) > 200 else '')
            tgt_ids = y_row[out_start:min(seg_end, out_start + 60)]
            pred_ids = p_row[out_start:min(seg_end, out_start + 60)]
            tgt_text = enc.decode([t for t in tgt_ids if 0 <= t < 50256])
            pred_text = enc.decode([t for t in pred_ids if 0 <= t < 50256])

            print(f"  [{b}] {ctx_text}")
            print(f"    target: {tgt_text[:150].strip()}")
            print(f"    pred:   {pred_text[:150].strip()}")
            print()
            n_shown += 1
            break

    # Show numeric decode quality if applicable
    if num_loss_dict is not None:
        if numeric_output_mode == 'surface':
            print(f"  numeric loss: total={num_loss_dict['total'].item():.4f} "
                  f"(sign={num_loss_dict['sign_loss'].item():.4f} "
                  f"scale={num_loss_dict['scale_loss'].item():.4f} "
                  f"len={num_loss_dict['len_loss'].item():.4f} "
                  f"digit={num_loss_dict['digit_loss'].item():.4f})")
        else:
            print(f"  numeric loss: total={num_loss_dict['total'].item():.4f} "
                  f"(sign={num_loss_dict['sign_loss'].item():.4f} "
                  f"exp={num_loss_dict['exp_loss'].item():.4f} "
                  f"digit={num_loss_dict['digit_loss'].item():.4f}"
                  f"{' mant={:.4f}'.format(num_loss_dict['mantissa_mse'].item()) if 'mantissa_mse' in num_loss_dict else ''})")


def should_run_rollout(it):
    return (
        rollout_loss_lambda > 0.0
        and rollout_every > 0
        and it >= rollout_start_iter
        and (it % rollout_every == 0)
    )


@torch.no_grad()
def build_rollout_batch(model_ref, x, y, nv, nm, nc):
    """Create a short self-conditioned suffix-training batch from one sample."""
    if x.size(1) < (rollout_prefix_tokens + rollout_steps + rollout_future_steps + 2):
        return None

    prefix_len = min(rollout_prefix_tokens, x.size(1) - rollout_steps - rollout_future_steps - 1)
    if prefix_len < 8:
        return None

    prefix_x = x[:1, :prefix_len]
    prefix_nv = nv[:1, :prefix_len]
    prefix_nm = nm[:1, :prefix_len]

    generated = model_ref.generate(
        prefix_x,
        max_new_tokens=rollout_steps,
        temperature=0.0,
        top_k=1,
        num_values=prefix_nv,
        num_mask=prefix_nm,
        return_components=True,
    )
    gen_x, _, gen_components = generated
    gen_tokens = gen_x[:, prefix_len:]
    gen_len = gen_tokens.size(1)
    if gen_len == 0:
        return None

    future_start = prefix_len + gen_len
    future_len = min(rollout_future_steps, x.size(1) - future_start)
    if future_len <= 0:
        return None

    surface_cols = 3 + surface_max_digits
    gen_nv = torch.zeros((1, gen_len), dtype=nv.dtype, device=x.device)
    gen_nm = (gen_tokens == NUM_TOKEN_ID)
    gen_surface_rows = torch.zeros((1, gen_len, surface_cols), dtype=torch.long, device=x.device)
    gen_surface_mask = torch.zeros((1, gen_len), dtype=torch.bool, device=x.device)
    if gen_components:
        comp_rows = model_ref.components_to_feedback_rows(
            [comp for _, comp in gen_components],
            device=x.device,
        )
        for row_idx, (abs_pos, _) in enumerate(gen_components):
            local_idx = abs_pos - prefix_len
            if 0 <= local_idx < gen_len:
                gen_surface_rows[0, local_idx] = comp_rows[row_idx]
                gen_surface_mask[0, local_idx] = True

    suffix_x = x[:1, future_start:future_start + future_len]
    suffix_nv = nv[:1, future_start:future_start + future_len]
    suffix_nm = nm[:1, future_start:future_start + future_len]
    suffix_surface_rows = torch.zeros((1, future_len, surface_cols), dtype=torch.long, device=x.device)
    suffix_surface_mask = torch.zeros((1, future_len), dtype=torch.bool, device=x.device)

    roll_x = torch.cat([prefix_x, gen_tokens, suffix_x], dim=1)
    roll_nv = torch.cat([prefix_nv, gen_nv, suffix_nv], dim=1)
    roll_nm = torch.cat([prefix_nm, gen_nm, suffix_nm], dim=1)
    roll_surface_rows = torch.cat([
        torch.zeros((1, prefix_len, surface_cols), dtype=torch.long, device=x.device),
        gen_surface_rows,
        suffix_surface_rows,
    ], dim=1)
    roll_surface_mask = torch.cat([
        torch.zeros((1, prefix_len), dtype=torch.bool, device=x.device),
        gen_surface_mask,
        suffix_surface_mask,
    ], dim=1)

    roll_targets = torch.full_like(roll_x, -1)
    roll_nc = torch.zeros((1, roll_x.size(1), nc.size(-1)), dtype=nc.dtype, device=x.device)
    target_start = prefix_len + gen_len - 1
    source_start = future_start - 1
    roll_targets[:, target_start:target_start + future_len] = y[:1, source_start:source_start + future_len]
    roll_nc[:, target_start:target_start + future_len] = nc[:1, source_start:source_start + future_len]

    return {
        'x': roll_x,
        'targets': roll_targets,
        'num_values': roll_nv,
        'num_mask': roll_nm,
        'num_surface_rows': roll_surface_rows,
        'num_surface_mask': roll_surface_mask,
        'num_targets': roll_nc,
        'generated_steps': gen_len,
        'future_steps': future_len,
        'generated_num_count': int(gen_surface_mask.sum().item()),
    }


def save_checkpoint(it, val_loss, is_best=False):
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

X, Y, NV, NM, NC = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

print(f"\nStarting training from iter {iter_num} (max_iters={max_iters})")

while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    lora_phase_scale = 0.0 if iter_num < decoder_warmup_iters else 1.0
    for pg in optimizer.param_groups:
        group_scale = float(pg.get('lr_scale', 1.0))
        if str(pg.get('group_name', '')).startswith('lora_'):
            group_scale *= lora_phase_scale
        pg['lr'] = lr * group_scale
    lora_lr = lr * lora_lr_scale * lora_phase_scale
    adapt_lr = lr * adapter_lr_scale
    decoder_lr = lr * decoder_lr_scale

    # === Eval and checkpoint ===
    if iter_num % eval_interval == 0:
        if ddp:
            barrier()
        if master_process:
            losses = estimate_loss()
            t_train = losses['train']
            t_val = losses['val']
            print(f"step {iter_num}: "
                  f"train text={t_train['text_loss']:.4f} num={t_train['num_loss']:.4f} "
                  f"total={t_train['total_loss']:.4f} | "
                  f"val text={t_val['text_loss']:.4f} num={t_val['num_loss']:.4f} "
                  f"total={t_val['total_loss']:.4f}")
            val_total = t_val['total_loss']
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/text_loss": t_train['text_loss'],
                    "train/num_loss": t_train['num_loss'],
                    "val/text_loss": t_val['text_loss'],
                    "val/num_loss": t_val['num_loss'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })
            if val_total < best_val_loss or always_save_checkpoint:
                if iter_num > 0:
                    is_best = val_total < best_val_loss
                    best_val_loss = min(best_val_loss, val_total)
                    save_checkpoint(iter_num, best_val_loss, is_best)
                else:
                    best_val_loss = min(best_val_loss, val_total)
        if ddp:
            barrier()
        if iter_num == 0 and eval_only:
            break

    # === Forward / backward ===
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, text_loss, num_loss_dict = model(
                X, Y, num_values=NV, num_mask=NM,
                **numeric_target_kwargs(NC),
            )
            loss = text_loss
            if num_loss_dict is not None:
                loss = loss + num_loss_lambda * num_loss_dict['total']

            rollout_value = 0.0
            rollout_meta = None
            if should_run_rollout(iter_num):
                rollout_batch = build_rollout_batch(raw_model, X, Y, NV, NM, NC)
                if rollout_batch is not None:
                    _, rollout_text_loss, rollout_num_loss_dict = model(
                        rollout_batch['x'],
                        rollout_batch['targets'],
                        num_values=rollout_batch['num_values'],
                        num_mask=rollout_batch['num_mask'],
                        num_surface_rows=rollout_batch['num_surface_rows'],
                        num_surface_mask=rollout_batch['num_surface_mask'],
                        **numeric_target_kwargs(rollout_batch['num_targets']),
                    )
                    rollout_loss = rollout_text_loss
                    if rollout_num_loss_dict is not None:
                        rollout_loss = rollout_loss + num_loss_lambda * rollout_num_loss_dict['total']
                    rollout_value = float(rollout_loss.detach().item())
                    rollout_meta = rollout_batch
                    loss = loss + rollout_loss_lambda * rollout_loss
            loss = loss / gradient_accumulation_steps

        if (micro_step == gradient_accumulation_steps - 1
                and iter_num % diag_interval == 0 and master_process):
            _diag_logits = logits.detach()
            _diag_targets = Y.clone()
            _diag_nv = NV.clone()
            _diag_nm = NM.clone()
            _diag_text_loss = text_loss.item() if text_loss is not None else 0.0
            _diag_num_loss = num_loss_dict['total'].item() if num_loss_dict else 0.0
            _diag_sign_loss = num_loss_dict['sign_loss'].item() if num_loss_dict else 0.0
            _diag_scale_loss = num_loss_dict.get('scale_loss', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_len_loss = num_loss_dict.get('len_loss', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_exp_loss = num_loss_dict.get('exp_loss', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_digit_loss = num_loss_dict['digit_loss'].item() if num_loss_dict else 0.0
            _diag_mantissa_mse = num_loss_dict.get('mantissa_mse', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_rollout_loss = rollout_value
            _diag_rollout_gen = rollout_meta['generated_steps'] if rollout_meta is not None else 0
            _diag_rollout_future = rollout_meta['future_steps'] if rollout_meta is not None else 0
            _diag_rollout_nums = rollout_meta['generated_num_count'] if rollout_meta is not None else 0

        X, Y, NV, NM, NC = get_batch('train')
        scaler.scale(loss).backward()

    # === Gradient clipping ===
    _grad_norms = None

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if iter_num % diag_interval == 0 and master_process:
            _grad_norms = collect_grad_norms(base_model.named_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif iter_num % diag_interval == 0 and master_process:
        _grad_norms = collect_grad_norms(base_model.named_parameters())

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # === Timing ===
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

    # === Diagnostics ===
    if iter_num % diag_interval == 0 and master_process:
        g = _grad_norms or {}
        num_count = int(_diag_nm.sum().item()) if _diag_nm is not None else 0
        total_tokens = batch_size * block_size
        num_target_mask = (_diag_targets == NUM_TOKEN_ID)
        n_num_targets = int(num_target_mask.sum().item())

        print(f"  === DIAG iter {iter_num} ===")
        if numeric_output_mode == 'surface':
            print(f"  text_loss: {_diag_text_loss:.4f}, "
                  f"num_loss: {_diag_num_loss:.4f} "
                  f"(sign={_diag_sign_loss:.4f} scale={_diag_scale_loss:.4f} "
                  f"len={_diag_len_loss:.4f} digit={_diag_digit_loss:.4f})")
        else:
            print(f"  text_loss: {_diag_text_loss:.4f}, "
                  f"num_loss: {_diag_num_loss:.4f} "
                  f"(sign={_diag_sign_loss:.4f} exp={_diag_exp_loss:.4f} "
                  f"digit={_diag_digit_loss:.4f} "
                  f"mantissa_mse={_diag_mantissa_mse:.4f})")
        print(f"  grads: total {g.get('total', 0):.4f}, "
              f"lora {g.get('lora', 0):.4f}, "
              f"adapter {g.get('adapter', 0):.4f}, "
              f"decoder {g.get('decoder', 0):.4f}")
        print(f"  <NUM> tokens: {num_count}/{total_tokens} "
              f"({num_count / total_tokens * 100:.1f}%)")
        print(f"  lr: lora {lora_lr:.2e}, adapter {adapt_lr:.2e}, "
              f"decoder {decoder_lr:.2e}")
        if _diag_rollout_gen > 0:
            print(f"  rollout: loss={_diag_rollout_loss:.4f}, "
                  f"gen_steps={_diag_rollout_gen}, future_steps={_diag_rollout_future}, "
                  f"gen_nums={_diag_rollout_nums}")
        else:
            print(f"  rollout: disabled or skipped")

        out_acc = compute_output_accuracy(_diag_logits, _diag_targets)
        if out_acc is not None:
            print(f"  token accuracy: {out_acc['overall']:.3f} "
                  f"({out_acc['n_correct']}/{out_acc['n_tokens']})")
        if n_num_targets > 0:
            preds = _diag_logits.argmax(dim=-1)
            num_pred_correct = (preds[num_target_mask] == NUM_TOKEN_ID).sum().item()
            print(f"  <NUM> token prediction: {num_pred_correct}/{n_num_targets} "
                  f"({num_pred_correct / n_num_targets:.3f})")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "diag/text_loss": _diag_text_loss,
                "diag/num_loss": _diag_num_loss,
                "diag/sign_loss": _diag_sign_loss,
                "diag/scale_loss": _diag_scale_loss,
                "diag/len_loss": _diag_len_loss,
                "diag/digit_loss": _diag_digit_loss,
                "diag/rollout_loss": _diag_rollout_loss,
                "diag/rollout_gen_steps": _diag_rollout_gen,
                "diag/rollout_future_steps": _diag_rollout_future,
                "grad/total": g.get('total', 0.0),
                "grad/lora": g.get('lora', 0.0),
                "grad/adapter": g.get('adapter', 0.0),
                "grad/decoder": g.get('decoder', 0.0),
                "lr/lora": lora_lr,
                "lr/adapter": adapt_lr,
                "lr/decoder": decoder_lr,
            })

    # === Sample eval ===
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

    save_checkpoint(iter_num, best_val_loss, is_best=False)
    print(f"  Unmerged: {out_dir}/ckpt_iter{iter_num}.pt")

    merge_lora_weights(base_model)
    merged_state = base_model.state_dict()
    torch.save({
        'model': merged_state,
        'model_args': model_args,
        'config': config,
    }, os.path.join(out_dir, 'ckpt_merged.pt'))
    print(f"  Merged:   {out_dir}/ckpt_merged.pt")
    print("=" * 60)

if ddp:
    destroy_process_group()
