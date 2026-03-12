"""
Training script for Stage 1: surface-oriented analytic number integration.

Loads a pretrained Baby Luciole checkpoint and attaches:
  - AnalyticNumberCodec (frozen, deterministic)
  - Trainable numeric adapter MLP (80→768)
  - Trainable numeric decoder heads (surface or legacy exponent mode)

Base model is completely frozen. Only adapter + decoder heads are trained.

Dual loss:  L = L_text + λ_num * (L_sign + L_exp + λ_d * K * L_digit_mean)

Data format (from generate_data_analytic_surface.py):
  - {split}.bin            : uint16 token IDs with <NUM> placeholders
  - {split}_nums.bin       : float32 values at <NUM> positions
  - {split}_components.bin : uint8 (N, 34) pre-computed [sign, exp, d0..d31]
  - {split}_surface.bin    : uint8 (N, 3 + K) pre-computed [sign, scale, len, digits...]

Usage:
  python train_analytic.py
  python train_analytic.py init_from=pretrained pretrained_ckpt=path/to/converted.pt
  python train_analytic.py init_from=resume
"""

import os
import sys
import time
import math
import re
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier

from model_analytic_surface import NemotronAnalyticConfig, NemotronAnalytic, NUM_TOKEN_ID
from prepare import NUM_TOKEN_ID as _NUM_CHECK, EOT_TOKEN_ID


# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = '/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_surface_s1'
eval_interval = 5000
log_interval = 1
diag_interval = 100
sample_interval = 1000
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'pretrained'
pretrained_ckpt = ''
resume_ckpt = ''
# wandb logging
wandb_log = False
wandb_project = 'numbers'
wandb_run_name = 'luciole-analytic-s1'
# data
dataset = 'analytic_stage1'
data_dir = ''
gradient_accumulation_steps = 5 * 8
batch_size = 4
block_size = 256
# model
n_layer = 12
n_head = 24
n_kv_head = 8
n_embd = 768
ffn_hidden = 3072
dropout = 0.0
bias = False
# analytic codec
analytic_K = 32
analytic_exp_min = -32
analytic_exp_max = 32
numeric_output_mode = 'surface'
surface_max_digits = 32
surface_scale_min = 0
surface_scale_max = 32
# loss config
num_loss_lambda = 1.0
digit_loss_lambda = 1.0 / 32.0
# adamw optimizer
learning_rate = 6e-4
adapter_lr_scale = 1.0
max_iters = 15000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 15000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
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
# -----------------------------------------------------------------------------

TARGET_COLS = (3 + surface_max_digits) if numeric_output_mode == 'surface' else (2 + analytic_K)

# various inits, derived attributes, I/O setup
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
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data
if not data_dir:
    data_dir = os.path.join('data', dataset)
print(f"data directory: {data_dir}")


def get_batch(split):
    """Load a batch of tokens, number values, and aligned numeric supervision.

    Returns:
        x:  (B, block_size) int64   input token IDs
        y:  (B, block_size) int64   target token IDs (shifted +1)
        nv: (B, block_size) float32 number values aligned with x
        nm: (B, block_size) bool    True where x == NUM_TOKEN_ID
        nt: (B, block_size, C) uint8 numeric targets aligned with y
    """
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    nums = np.memmap(os.path.join(data_dir, f'{split}_nums.bin'), dtype=np.float32, mode='r')
    target_suffix = 'surface' if numeric_output_mode == 'surface' else 'components'
    comps = np.memmap(os.path.join(data_dir, f'{split}_{target_suffix}.bin'),
                      dtype=np.uint8, mode='r').reshape(-1, TARGET_COLS)

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    nv = torch.stack([torch.from_numpy(nums[i:i + block_size].copy()) for i in ix])
    # Components aligned with y (targets are shifted +1)
    nc = torch.stack([torch.from_numpy(comps[i + 1:i + 1 + block_size].copy()) for i in ix])

    nm = (x == NUM_TOKEN_ID)

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        nv = nv.pin_memory().to(device, non_blocking=True)
        nm = nm.pin_memory().to(device, non_blocking=True)
        nc = nc.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        nv, nm = nv.to(device), nm.to(device)
        nc = nc.to(device)

    return x, y, nv, nm, nc


def numeric_target_kwargs(targets):
    if numeric_output_mode == 'surface':
        return {'num_target_surface': targets}
    return {'num_target_components': targets}


# init
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head,
    n_embd=n_embd, ffn_hidden=ffn_hidden, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    analytic_K=analytic_K, analytic_exp_min=analytic_exp_min,
    analytic_exp_max=analytic_exp_max,
    numeric_output_mode=numeric_output_mode,
    surface_max_digits=surface_max_digits,
    surface_scale_min=surface_scale_min,
    surface_scale_max=surface_scale_max,
    num_loss_lambda=num_loss_lambda, digit_loss_lambda=digit_loss_lambda,
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size 50304")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    nemconf = NemotronAnalyticConfig(**model_args)
    model = NemotronAnalytic(nemconf)
elif init_from == 'pretrained':
    print(f"Initializing from pretrained Baby Luciole checkpoint")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    nemconf = NemotronAnalyticConfig(**model_args)
    model = NemotronAnalytic(nemconf)
    if pretrained_ckpt:
        model.load_pretrained_nemotron(pretrained_ckpt)
    else:
        print("WARNING: init_from=pretrained but no pretrained_ckpt provided")
    # Stage 1: freeze everything except adapter + decoder
    print("Stage 1: Freezing base model, training adapter + decoder only")
    for name, p in model.named_parameters():
        if not (name.startswith('num_adapter.') or
                name.startswith('num_decoder_sign.') or
                name.startswith('num_decoder_exp.') or
                name.startswith('num_decoder_digits.') or
                name.startswith('num_decoder_scale.') or
                name.startswith('num_decoder_len.') or
                name.startswith('num_surface_')):
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.1f}%)")
elif init_from == 'resume':
    ckpt_path = resume_ckpt if resume_ckpt else os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'ffn_hidden',
              'block_size', 'bias', 'vocab_size',
              'analytic_K', 'analytic_exp_min', 'analytic_exp_max',
              'num_loss_lambda', 'digit_loss_lambda']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    nemconf = NemotronAnalyticConfig(**model_args)
    model = NemotronAnalytic(nemconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Re-freeze base
    for name, p in model.named_parameters():
        if not (name.startswith('num_adapter.') or
                name.startswith('num_decoder_sign.') or
                name.startswith('num_decoder_exp.') or
                name.startswith('num_decoder_digits.') or
                name.startswith('num_decoder_scale.') or
                name.startswith('num_decoder_len.') or
                name.startswith('num_surface_')):
            p.requires_grad = False

model.to(device)

# GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type,
    adapter_lr_scale=adapter_lr_scale,
)
if init_from == 'resume':
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except ValueError as e:
        print(f"WARNING: could not load optimizer state: {e}")
checkpoint = None

# compile
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# ── helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(eval_model):
    out = {}
    eval_model.eval()
    for split in ['train', 'val']:
        text_losses = torch.zeros(eval_iters)
        num_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, NV, NM, NC = get_batch(split)
            with ctx:
                logits, text_loss, num_loss_dict = eval_model(
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
    eval_model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def decode_context(token_ids, num_values, enc):
    parts = []
    for i, tok in enumerate(token_ids):
        if tok == NUM_TOKEN_ID:
            val = num_values[i]
            parts.append(f"<{val:g}>")
        elif tok == EOT_TOKEN_ID:
            break
        else:
            try:
                parts.append(enc.decode([tok]))
            except Exception:
                parts.append(f"[{tok}]")
    return ''.join(parts)


def collect_group_grad_norms(named_parameters):
    group_sq = {}
    total_sq = 0.0
    for name, p in named_parameters:
        if p.grad is None:
            continue
        sq = p.grad.data.norm(2).item() ** 2
        total_sq += sq
        if ('num_adapter' in name or 'num_decoder' in name):
            key = 'adapter'
        else:
            key = 'transformer'
        group_sq[key] = group_sq.get(key, 0.0) + sq
    out = {k: v ** 0.5 for k, v in group_sq.items()}
    out['total'] = total_sq ** 0.5
    return out


@torch.no_grad()
def compute_numeric_accuracy(model_ref, x_hidden, targets, nc):
    """Compute sign, exponent, and digit accuracy on <NUM> output positions."""
    target_num_mask = (targets == NUM_TOKEN_ID)
    if not target_num_mask.any():
        return None

    hidden_num = x_hidden[target_num_mask]
    components = nc[target_num_mask]

    sign_targets = components[:, 0].long()
    exp_targets = components[:, 1].long()
    digit_targets = components[:, 2:].long()

    sign_logits = model_ref.num_decoder_sign(hidden_num)
    exp_logits = model_ref.num_decoder_exp(hidden_num)
    digit_logits = model_ref.num_decoder_digits(hidden_num)
    digit_logits = digit_logits.view(-1, model_ref.config.analytic_K, 10)

    M = hidden_num.size(0)

    sign_acc = (sign_logits.argmax(-1) == sign_targets).float().mean().item()
    exp_acc = (exp_logits.argmax(-1) == exp_targets).float().mean().item()
    digit_preds = digit_logits.argmax(-1)
    digit_acc = (digit_preds == digit_targets).float().mean().item()

    # Full number exact match: all 34 components correct
    sign_ok = (sign_logits.argmax(-1) == sign_targets)
    exp_ok = (exp_logits.argmax(-1) == exp_targets)
    digits_ok = (digit_preds == digit_targets).all(dim=-1)
    exact_match = (sign_ok & exp_ok & digits_ok).float().mean().item()

    return {
        'n_num': M,
        'sign_acc': sign_acc,
        'exp_acc': exp_acc,
        'digit_acc': digit_acc,
        'exact_match': exact_match,
    }


@torch.no_grad()
def eval_samples(eval_model, raw_model_ref, max_samples=5):
    """Show sample predictions with decoded numbers."""
    try:
        from prepare import _get_tokenizer
        enc = _get_tokenizer()
    except Exception:
        enc = None

    eval_model.eval()
    X, Y, NV, NM, NC = get_batch('val')
    with ctx:
        logits, text_loss, num_loss_dict = eval_model(
            X, Y, num_values=NV, num_mask=NM,
            **numeric_target_kwargs(NC),
        )
    eval_model.train()

    if enc is None:
        return

    B, T, _ = logits.shape
    preds = logits.argmax(dim=-1)

    # Get hidden states for numeric decoding by running a clean forward
    with torch.no_grad(), ctx:
        x = raw_model_ref.compute_hidden_states(
            X,
            num_values=NV,
            num_mask=NM,
        )

    print(f"  --- Sample eval (val) ---")
    n_shown = 0
    for b in range(B):
        if n_shown >= max_samples:
            break

        x_row = X[b].tolist()
        y_row = Y[b].tolist()
        nv_row = NV[b].tolist()

        eot_positions = [i for i, t in enumerate(x_row) if t == EOT_TOKEN_ID]
        if not eot_positions:
            eot_positions = [0]

        for seg_start_idx in range(len(eot_positions)):
            if n_shown >= max_samples:
                break

            seg_start = eot_positions[seg_start_idx] + 1 if seg_start_idx > 0 else 0
            seg_end = T
            for t in range(seg_start, T):
                if y_row[t] == EOT_TOKEN_ID:
                    seg_end = t
                    break

            if seg_end <= seg_start + 2:
                continue

            seg_x_ids = x_row[seg_start:seg_end]
            seg_nv = nv_row[seg_start:seg_end]
            context = decode_context(seg_x_ids, seg_nv, enc)

            if "Assistant" not in context:
                continue

            # Find <NUM> positions in target that fall in this segment
            num_positions = []
            for t in range(seg_start, seg_end):
                if y_row[t] == NUM_TOKEN_ID:
                    num_positions.append(t)

            print(f"  {context}")

            if num_positions:
                for pos in num_positions[:3]:  # show up to 3
                    # Decode number from hidden state
                    h = x[b:b+1, pos:pos+1, :]  # (1, 1, n_embd)
                    decoded = raw_model_ref.decode_numeric_output(h.squeeze(1))
                    target_val = NV[b, pos].item() if pos < T else 0.0
                    nc_row = NC[b, pos].tolist()
                    target_sign = "+" if nc_row[0] == 0 else "-"
                    if numeric_output_mode == 'surface':
                        target_scale = nc_row[1] + surface_scale_min
                        target_len = nc_row[2]
                        target_digits = "".join(str(d) for d in nc_row[3:3 + min(target_len, 6)]) + "..."
                        print(f"    NUM@{pos}: predicted={decoded[0]}, "
                              f"target_val={target_val:g} "
                              f"(s={target_sign} scale={target_scale} len={target_len} d={target_digits})")
                    else:
                        target_exp = nc_row[1] + analytic_exp_min
                        target_digits = "".join(str(d) for d in nc_row[2:8]) + "..."
                        print(f"    NUM@{pos}: predicted={decoded[0]}, "
                              f"target_val={target_val:g} "
                              f"(s={target_sign} e={target_exp} d={target_digits})")
            print()
            n_shown += 1
            break


# ── logging ──────────────────────────────────────────────────────────────────

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# ── training loop ────────────────────────────────────────────────────────────

X, Y, NV, NM, NC = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:

    # LR schedule
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        lr_scale = float(param_group.get('lr_scale', 1.0))
        param_group['lr'] = lr * lr_scale
    adapter_lr = lr * adapter_lr_scale

    # evaluate
    if iter_num % eval_interval == 0:
        if ddp:
            barrier()
        if master_process:
            losses = estimate_loss(raw_model)
            print(f"step {iter_num}: "
                  f"train text={losses['train']['text_loss']:.4f} "
                  f"num={losses['train']['num_loss']:.4f} "
                  f"total={losses['train']['total_loss']:.4f} | "
                  f"val text={losses['val']['text_loss']:.4f} "
                  f"num={losses['val']['num_loss']:.4f} "
                  f"total={losses['val']['total_loss']:.4f}")

            val_total = losses['val']['total_loss']
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/text_loss": losses['train']['text_loss'],
                    "train/num_loss": losses['train']['num_loss'],
                    "train/total_loss": losses['train']['total_loss'],
                    "val/text_loss": losses['val']['text_loss'],
                    "val/num_loss": losses['val']['num_loss'],
                    "val/total_loss": losses['val']['total_loss'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })

            if val_total < best_val_loss or always_save_checkpoint:
                if iter_num > 0:
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
                    torch.save(ckpt, os.path.join(out_dir, f'ckpt_iter{iter_num}.pt'))
                    print(f"saving checkpoint to {out_dir}/ckpt_iter{iter_num}.pt")
                    if val_total < best_val_loss:
                        torch.save(ckpt, os.path.join(out_dir, 'ckpt_best.pt'))
                        print(f"  new best val loss: {val_total:.4f}")
                best_val_loss = val_total
        if ddp:
            barrier()
    if iter_num == 0 and eval_only:
        break

    # forward backward
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, text_loss, num_loss_dict = model(
                X, Y,
                num_values=NV,
                num_mask=NM,
                **numeric_target_kwargs(NC),
            )
            # Combined loss
            loss = text_loss
            if num_loss_dict is not None:
                loss = loss + num_loss_lambda * num_loss_dict['total']
            loss = loss / gradient_accumulation_steps

        # diagnostics on last micro-step
        if (micro_step == gradient_accumulation_steps - 1
                and iter_num % diag_interval == 0 and master_process):
            _diag_logits = logits.detach()
            _diag_targets = Y.clone()
            _diag_NV = NV.clone()
            _diag_NM = NM.clone()
            _diag_NC = NC.clone()
            _diag_text_loss = text_loss.item() if text_loss is not None else 0.0
            _diag_num_loss = num_loss_dict['total'].item() if num_loss_dict else 0.0
            _diag_sign_loss = num_loss_dict['sign_loss'].item() if num_loss_dict else 0.0
            _diag_scale_loss = num_loss_dict.get('scale_loss', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_len_loss = num_loss_dict.get('len_loss', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_exp_loss = num_loss_dict.get('exp_loss', torch.tensor(0.0)).item() if num_loss_dict else 0.0
            _diag_digit_loss = num_loss_dict['digit_loss'].item() if num_loss_dict else 0.0
            _diag_mantissa_mse = num_loss_dict.get('mantissa_mse', torch.tensor(0.0)).item() if num_loss_dict else 0.0

        X, Y, NV, NM, NC = get_batch('train')
        scaler.scale(loss).backward()

    # clip
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if iter_num % diag_interval == 0 and master_process:
            _diag_grad_norms = collect_group_grad_norms(raw_model.named_parameters())
        pre_total = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif iter_num % diag_interval == 0 and master_process:
        _diag_grad_norms = collect_group_grad_norms(raw_model.named_parameters())
        pre_total = _diag_grad_norms.get('total', 0.0)

    # step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
              f"mfu {running_mfu * 100:.2f}%")

    # diagnostics
    if iter_num % diag_interval == 0 and master_process:
        grad_norm_total = _diag_grad_norms.get('total', 0.0)
        grad_norm_adapter = _diag_grad_norms.get('adapter', 0.0)
        grad_norm_transformer = _diag_grad_norms.get('transformer', 0.0)

        num_count = int(_diag_NM.sum().item())
        total_tokens = batch_size * block_size
        target_num_count = int((_diag_targets == NUM_TOKEN_ID).sum().item())

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
        print(f"  grads: total {grad_norm_total:.4f}, "
              f"transformer {grad_norm_transformer:.4f}, "
              f"adapter {grad_norm_adapter:.4f}")
        print(f"  <NUM> input: {num_count}/{total_tokens}, "
              f"<NUM> target: {target_num_count}/{total_tokens}")
        print(f"  lr: {lr:.2e}, adapter_lr: {adapter_lr:.2e}")

        # Text and <NUM> token accuracy from cached logits
        with torch.no_grad():
            preds = _diag_logits.argmax(dim=-1)
            valid_mask = (_diag_targets >= 0) & (_diag_targets != EOT_TOKEN_ID)
            # Exclude <NUM> targets from text accuracy
            text_mask = valid_mask & (_diag_targets != NUM_TOKEN_ID)
            n_text = text_mask.sum().item()
            if n_text > 0:
                text_correct = (preds[text_mask] == _diag_targets[text_mask]).sum().item()
                print(f"  text token accuracy: {text_correct/n_text:.3f} ({text_correct}/{n_text})")

            # <NUM> prediction accuracy (does model predict <NUM> when target is <NUM>?)
            num_target_mask = (_diag_targets == NUM_TOKEN_ID)
            n_num_targets = num_target_mask.sum().item()
            if n_num_targets > 0:
                num_pred_correct = (preds[num_target_mask] == NUM_TOKEN_ID).sum().item()
                print(f"  <NUM> token prediction: {num_pred_correct}/{n_num_targets} "
                      f"({num_pred_correct/n_num_targets:.3f})")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "diag/text_loss": _diag_text_loss,
                "diag/num_loss": _diag_num_loss,
                "diag/sign_loss": _diag_sign_loss,
                "diag/scale_loss": _diag_scale_loss,
                "diag/len_loss": _diag_len_loss,
                "diag/exp_loss": _diag_exp_loss,
                "diag/digit_loss": _diag_digit_loss,
                "grad/total": grad_norm_total,
                "grad/transformer": grad_norm_transformer,
                "grad/adapter": grad_norm_adapter,
                "lr": lr,
            }
            wandb.log(log_dict)

    # sample eval
    if iter_num % sample_interval == 0 and master_process and iter_num > 0:
        eval_samples(raw_model, raw_model, max_samples=3)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
