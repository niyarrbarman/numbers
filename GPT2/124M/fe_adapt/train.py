"""
Training script for number-aware Baby Luciole (114M Nemotron3) — adapter fine-tune.

Loads a pretrained Baby Luciole checkpoint (converted from NeMo) and attaches
a NumberEncoder v10 adapter. Supports staged training:
  - Stage 1: Freeze base model, train adapter MLP only
  - Stage 2: Unfreeze base + adapter (full fine-tune or LoRA)

Data format is identical to GPT-2 FE:
  - Token IDs ({split}.bin, uint16) with <NUM> placeholders
  - Number values ({split}_nums.bin, float32) at <NUM> positions

Usage:
  python train.py                          # defaults (scratch)
  python train.py init_from=pretrained pretrained_ckpt=path/to/converted.pt
  python train.py init_from=resume
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

from model import NemotronConfig, Nemotron, NUM_TOKEN_ID
from prepare import NUM_TOKEN_ID as _NUM_CHECK  # verify prepare.py is importable


# -----------------------------------------------------------------------------
# default config values for Baby Luciole adapter fine-tuning
# I/O
out_dir = '/tmpdir/m24047brmn/numbers/model_checkpoints/luciole_fe_adapt'
eval_interval = 5000
log_interval = 1
diag_interval = 100
sample_interval = 1000
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'pretrained'  # 'scratch' or 'resume' or 'pretrained'
pretrained_ckpt = ''  # path to converted Nemotron checkpoint (from convert_nemo_ckpt.py)
resume_ckpt = ''  # explicit checkpoint path for resume (overrides out_dir/ckpt.pt)
freeze_base = True  # Stage 1: freeze base model, train adapter only
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'luciole-114M-fe-adapt'
# data
dataset = 'numtasks_124M_fe'
data_dir = ''  # override to set absolute path; if empty, uses data/{dataset}
gradient_accumulation_steps = 10 * 8
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
# number embedding
num_emb_checkpoint = ''  # path to NumberEncoder .pt checkpoint
num_emb_dim = 128
num_emb_scale_dims = 16
num_emb_residue_periods = '2,5,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000'
num_norm_match = True
num_blend_beta_start = 0.0
num_blend_beta_end = 1.0
num_blend_warmup_iters = 2000
num_blend_ramp_iters = 18000
# adamw optimizer
learning_rate = 6e-4
adapter_lr_scale = 0.2
max_iters = 40000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 40000  # should match max_iters for full cosine schedule
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
# Simple configurator: parse key=value args from command line
for arg in sys.argv[1:]:
    if '=' in arg:
        key, val = arg.split('=', 1)
        key = key.lstrip('-')
        if key in config_keys:
            # Try to evaluate the value (handles int, float, bool)
            try:
                val = eval(val)
            except Exception:
                pass
            globals()[key] = val
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

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

# poor man's data loader (dual-stream: tokens + numbers for input embedding)
if not data_dir:
    data_dir = os.path.join('data', dataset)
print(f"data directory: {data_dir}")


def get_batch(split):
    """Load a batch of tokens and parallel number values.

    Returns:
        x:  (B, block_size) int64   input token IDs
        y:  (B, block_size) int64   target token IDs (shifted +1)
        nv: (B, block_size) float32 number values aligned with x
        nm: (B, block_size) bool    True where x == NUM_TOKEN_ID
    """
    # Recreate memmap each time to avoid memory leak
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    nums = np.memmap(os.path.join(data_dir, f'{split}_nums.bin'), dtype=np.float32, mode='r')

    # -1 extra because targets are shifted by 1
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    nv = torch.stack([torch.from_numpy(nums[i:i + block_size].copy()) for i in ix])

    nm = (x == NUM_TOKEN_ID)

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        nv = nv.pin_memory().to(device, non_blocking=True)
        nm = nm.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        nv = nv.to(device)
        nm = nm.to(device)

    return x, y, nv, nm


# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
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
    num_emb_dim=num_emb_dim, num_emb_checkpoint=num_emb_checkpoint,
    num_emb_scale_dims=num_emb_scale_dims,
    num_emb_residue_periods=num_emb_residue_periods,
    num_norm_match=num_norm_match,
    num_blend_beta_infer=num_blend_beta_end,
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size 50304 (GPT-2 50257 rounded up)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    nemconf = NemotronConfig(**model_args)
    model = Nemotron(nemconf)
elif init_from == 'pretrained':
    print(f"Initializing from pretrained Baby Luciole checkpoint")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    nemconf = NemotronConfig(**model_args)
    model = Nemotron(nemconf)
    if pretrained_ckpt:
        model.load_pretrained_nemotron(pretrained_ckpt)
    else:
        print("WARNING: init_from=pretrained but no pretrained_ckpt provided")
    # Stage 1: freeze base model, train only adapter
    if freeze_base:
        print("Stage 1: Freezing base model, training adapter only")
        for name, p in model.named_parameters():
            if not (name.startswith('num_adapter.') or name.startswith('num_encoder.')):
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
              'num_emb_dim', 'num_emb_checkpoint',
              'num_emb_scale_dims', 'num_emb_residue_periods',
              'num_norm_match', 'num_blend_beta_infer']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    nemconf = NemotronConfig(**model_args)
    model = Nemotron(nemconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Restore freeze state if specified
    if freeze_base:
        for name, p in model.named_parameters():
            if not (name.startswith('num_adapter.') or name.startswith('num_encoder.')):
                p.requires_grad = False

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
    adapter_lr_scale=adapter_lr_scale,
)
if init_from == 'resume':
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except ValueError as e:
        print(f"WARNING: could not load optimizer state from checkpoint: {e}")
        print("WARNING: continuing with freshly initialized optimizer state.")
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(eval_model, num_blend_beta):
    out = {}
    eval_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, NV, NM = get_batch(split)
            with ctx:
                logits, loss = eval_model(
                    X, Y,
                    num_values=NV,
                    num_mask=NM,
                    num_blend_beta=num_blend_beta,
                    num_norm_match=num_norm_match,
                )
            losses[k] = loss.item()
        out[split] = losses.mean()
    eval_model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_num_blend_beta(it):
    """Cosine ramp for blending base <NUM> embedding with FE adapter output."""
    start = float(num_blend_beta_start)
    end = float(num_blend_beta_end)
    if num_blend_ramp_iters <= 0:
        return max(0.0, min(1.0, end))
    if it < num_blend_warmup_iters:
        return max(0.0, min(1.0, start))
    ramp_t = it - num_blend_warmup_iters
    if ramp_t >= num_blend_ramp_iters:
        return max(0.0, min(1.0, end))
    frac = ramp_t / max(1, num_blend_ramp_iters)
    frac = 0.5 * (1.0 - math.cos(math.pi * frac))
    beta = start + (end - start) * frac
    return max(0.0, min(1.0, beta))


def decode_context(token_ids, num_values, enc):
    """Decode a sequence of token IDs to readable text, showing <NUM:val> for number tokens."""
    parts = []
    for i, tok in enumerate(token_ids):
        if tok == NUM_TOKEN_ID:
            val = num_values[i]
            parts.append(f"<{val:g}>")
        elif tok == 50256:  # EOT
            break
        else:
            try:
                parts.append(enc.decode([tok]))
            except Exception:
                parts.append(f"[{tok}]")
    return ''.join(parts)


# Text output diagnostic helpers
_number_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


@torch.no_grad()
def compute_output_accuracy(logits, targets):
    """Compute token-level accuracy on non-padding, non-EOT output tokens.

    Returns dict with overall accuracy and number of tokens evaluated.
    """
    preds = logits.argmax(dim=-1)
    # Exclude padding (-1) and EOT (50256)
    valid_mask = (targets >= 0) & (targets != 50256)
    n_valid = valid_mask.sum().item()
    if n_valid == 0:
        return None
    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
    # Also check accuracy on <NUM> token predictions (should be rare in targets
    # since NUM only appears in input, but check anyway)
    num_mask = (targets == NUM_TOKEN_ID)
    num_total = num_mask.sum().item()
    num_correct = (preds[num_mask] == targets[num_mask]).sum().item() if num_total > 0 else 0

    return {
        'overall': correct / n_valid,
        'n_tokens': n_valid,
        'n_correct': correct,
        'num_tok_total': num_total,
        'num_tok_correct': num_correct,
    }


def collect_group_grad_norms(named_parameters):
    """L2 grad norms split into transformer vs adapter/encoder groups."""
    group_sq = {}
    total_sq = 0.0
    for name, p in named_parameters:
        if p.grad is None:
            continue
        sq = p.grad.data.norm(2).item() ** 2
        total_sq += sq
        if 'num_adapter' in name or 'num_encoder' in name:
            key = 'adapter'
        else:
            key = 'transformer'
        group_sq[key] = group_sq.get(key, 0.0) + sq

    out = {k: v ** 0.5 for k, v in group_sq.items()}
    out['total'] = total_sq ** 0.5
    return out


@torch.no_grad()
def collect_num_injection_stats(eval_model, num_values, num_mask, num_blend_beta):
    """Summarize scale behavior of base/delta/blended number embeddings."""
    if num_mask is None or int(num_mask.sum().item()) == 0:
        return None

    flat_vals = num_values[num_mask].float()
    num_emb = eval_model.num_encoder(flat_vals)
    delta_raw = eval_model.num_adapter(num_emb)

    n_num = delta_raw.size(0)
    base_vec = eval_model.transformer.wte.weight[NUM_TOKEN_ID].unsqueeze(0).to(delta_raw.dtype)
    base = base_vec.expand(n_num, -1)

    base_norm = base.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta_raw_norm = delta_raw.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)

    if num_norm_match:
        scale = base_norm / delta_raw_norm
        delta_eff = delta_raw * scale.to(delta_raw.dtype)
    else:
        scale = torch.ones_like(base_norm)
        delta_eff = delta_raw

    beta = float(max(0.0, min(1.0, num_blend_beta)))
    blended = (1.0 - beta) * base + beta * delta_eff
    blended_norm = blended.float().norm(dim=-1)
    delta_eff_norm = delta_eff.float().norm(dim=-1)

    return {
        'beta': beta,
        'n_num': n_num,
        'base_norm_mean': base_norm.mean().item(),
        'delta_raw_norm_mean': delta_raw_norm.mean().item(),
        'delta_eff_norm_mean': delta_eff_norm.mean().item(),
        'blended_norm_mean': blended_norm.mean().item(),
        'norm_scale_mean': scale.mean().item(),
        'norm_scale_max': scale.max().item(),
        'norm_scale_min': scale.min().item(),
    }


@torch.no_grad()
def eval_samples(eval_model, num_blend_beta, max_samples=5):
    """Run model on val batch, show full task context with predicted vs target text output."""
    # Lazy load tiktoken (needs TIKTOKEN_CACHE_DIR set)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    eval_model.eval()
    X, Y, NV, NM = get_batch('val')
    with ctx:
        logits, loss = eval_model(
            X, Y,
            num_values=NV,
            num_mask=NM,
            num_blend_beta=num_blend_beta,
            num_norm_match=num_norm_match,
        )
    eval_model.train()

    B, T, _ = logits.shape
    preds = logits.argmax(dim=-1)  # (B, T)

    print(f"  --- Sample eval (val) ---")
    n_shown = 0
    for b in range(B):
        if n_shown >= max_samples:
            break

        x_row = X[b].tolist()
        y_row = Y[b].tolist()
        p_row = preds[b].tolist()
        nv_row = NV[b].tolist()

        # Find the arrow token(s) in x to split input/output
        if enc is None:
            continue

        # Find tasks bounded by EOT tokens
        eot_positions = [i for i, t in enumerate(x_row) if t == 50256]
        if not eot_positions:
            eot_positions = [0]

        # Try each task segment
        for seg_start_idx in range(len(eot_positions)):
            if n_shown >= max_samples:
                break

            seg_start = eot_positions[seg_start_idx] + 1 if seg_start_idx > 0 else 0
            # Find next EOT in target to bound the output
            seg_end = T
            for t in range(seg_start, T):
                if y_row[t] == 50256:
                    seg_end = t
                    break

            if seg_end <= seg_start + 2:
                continue

            # Decode segment context from x
            seg_x_ids = x_row[seg_start:seg_end]
            seg_nv = nv_row[seg_start:seg_end]
            context = decode_context(seg_x_ids, seg_nv, enc)

            if "\u2192" not in context:
                continue

            # Find arrow position within segment to split input/output
            arrow_pos = None
            for t in range(seg_start, seg_end):
                tok_text = enc.decode([x_row[t]]) if 0 <= x_row[t] < 50257 else ""
                if "\u2192" in tok_text:
                    arrow_pos = t
                    break

            if arrow_pos is None:
                continue

            # Target and predicted output (tokens after arrow)
            out_start = arrow_pos + 1
            if out_start >= seg_end:
                continue

            tgt_ids = y_row[out_start:seg_end]
            pred_ids = p_row[out_start:seg_end]

            tgt_text = enc.decode([t for t in tgt_ids if 0 <= t < 50257])
            pred_text = enc.decode([t for t in pred_ids if 0 <= t < 50257])

            # Parse numbers from both
            tgt_nums = [float(v) for v in _number_re.findall(tgt_text)]
            pred_nums = [float(v) for v in _number_re.findall(pred_text)]

            # Input context (up to arrow)
            input_ctx = decode_context(x_row[seg_start:arrow_pos + 1],
                                       nv_row[seg_start:arrow_pos + 1], enc)

            print(f"  {input_ctx}")
            print(f"    target: {tgt_text.strip()}")
            print(f"    pred:   {pred_text.strip()}")

            if tgt_nums and pred_nums and len(tgt_nums) == len(pred_nums):
                errs = [abs(t - p) for t, p in zip(tgt_nums, pred_nums)]
                err_strs = [f"{e:.2f}" for e in errs]
                print(f"    error:  {' '.join(err_strs)}")
            elif tgt_nums and (not pred_nums or len(tgt_nums) != len(pred_nums)):
                print(f"    (number count mismatch: target {len(tgt_nums)}, pred {len(pred_nums)})")
            print()

            n_shown += 1
            break  # one sample per batch row, move to next b


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, NV, NM = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
current_num_blend_beta = get_num_blend_beta(iter_num)

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    current_num_blend_beta = get_num_blend_beta(iter_num)
    for param_group in optimizer.param_groups:
        lr_scale = float(param_group.get('lr_scale', 1.0))
        param_group['lr'] = lr * lr_scale
    adapter_lr = lr * adapter_lr_scale

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        # Keep all ranks in lockstep around rank-0-only evaluation/checkpointing.
        if ddp:
            barrier()
        if master_process:
            losses = estimate_loss(raw_model, current_num_blend_beta)
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
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    # Always save latest (for resume)
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    # Save periodic snapshot
                    torch.save(checkpoint, os.path.join(out_dir, f'ckpt_iter{iter_num}.pt'))
                    print(f"saving checkpoint to {out_dir}/ckpt_iter{iter_num}.pt")
                    # Save best separately
                    if losses['val'] < best_val_loss:
                        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_best.pt'))
                        print(f"  new best val loss: {losses['val']:.4f}")
                best_val_loss = losses['val']
        if ddp:
            barrier()
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(
                X, Y,
                num_values=NV,
                num_mask=NM,
                num_blend_beta=current_num_blend_beta,
                num_norm_match=num_norm_match,
            )
            loss = loss / gradient_accumulation_steps
        # snapshot logits for diagnostics (last micro_step only)
        if micro_step == gradient_accumulation_steps - 1 and iter_num % diag_interval == 0 and master_process:
            _diag_logits = logits.detach()
            _diag_targets = Y.clone()
            _diag_num_values = NV.clone()
            _diag_num_mask = NM.clone()
        # async prefetch next batch
        X, Y, NV, NM = get_batch('train')
        # backward pass
        scaler.scale(loss).backward()
    _diag_grad_norms_pre = None
    _diag_grad_norms = None
    _diag_clip_pre_total = None
    _diag_clip_coef = None
    _diag_num_inj = None

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if iter_num % diag_interval == 0 and master_process:
            _diag_grad_norms_pre = collect_group_grad_norms(raw_model.named_parameters())

        pre_total = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if iter_num % diag_interval == 0 and master_process:
            _diag_clip_pre_total = float(pre_total.item() if hasattr(pre_total, "item") else pre_total)
            if _diag_clip_pre_total > 0.0:
                _diag_clip_coef = min(1.0, float(grad_clip) / _diag_clip_pre_total)
            else:
                _diag_clip_coef = 1.0
    elif iter_num % diag_interval == 0 and master_process:
        # No clipping case: pre/post norms are identical.
        _diag_grad_norms_pre = collect_group_grad_norms(raw_model.named_parameters())
        _diag_clip_pre_total = _diag_grad_norms_pre.get('total', 0.0)
        _diag_clip_coef = 1.0

    # snapshot post-clip grad norms BEFORE zero_grad (for diagnostics)
    if iter_num % diag_interval == 0 and master_process:
        _diag_grad_norms = collect_group_grad_norms(raw_model.named_parameters())
        _diag_num_inj = collect_num_injection_stats(
            raw_model, _diag_num_values, _diag_num_mask, current_num_blend_beta
        )
    # step the optimizer
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
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

    # --- Detailed diagnostics every diag_interval steps ---
    if iter_num % diag_interval == 0 and master_process:
        grad_norm_total = _diag_grad_norms.get('total', 0.0)
        grad_norm_adapter = _diag_grad_norms.get('adapter', 0.0)
        grad_norm_transformer = _diag_grad_norms.get('transformer', 0.0)
        grad_norm_total_pre = _diag_grad_norms_pre.get('total', 0.0) if _diag_grad_norms_pre else 0.0
        grad_norm_adapter_pre = _diag_grad_norms_pre.get('adapter', 0.0) if _diag_grad_norms_pre else 0.0
        grad_norm_transformer_pre = _diag_grad_norms_pre.get('transformer', 0.0) if _diag_grad_norms_pre else 0.0
        grad_ratio_transformer_pre = (
            grad_norm_transformer_pre / grad_norm_total_pre if grad_norm_total_pre > 0 else 0.0
        )
        grad_ratio_adapter_pre = (
            grad_norm_adapter_pre / grad_norm_total_pre if grad_norm_total_pre > 0 else 0.0
        )
        grad_ratio_transformer = (
            grad_norm_transformer / grad_norm_total if grad_norm_total > 0 else 0.0
        )
        grad_ratio_adapter = (
            grad_norm_adapter / grad_norm_total if grad_norm_total > 0 else 0.0
        )

        # Count NUM tokens in the batch
        num_count = int(_diag_num_mask.sum().item()) if _diag_num_mask is not None else 0
        total_tokens = batch_size * block_size

        print(f"  === DIAG iter {iter_num} ===")
        print(f"  loss: {loss.item() * gradient_accumulation_steps:.4f}")
        print(f"  grads (preclip): total {grad_norm_total_pre:.4f}, "
              f"transformer {grad_norm_transformer_pre:.4f}, "
              f"adapter {grad_norm_adapter_pre:.4f}")
        print(f"  grad ratio (preclip): transformer {grad_ratio_transformer_pre:.4f}, "
              f"adapter {grad_ratio_adapter_pre:.4f}")
        print(f"  grads: total {grad_norm_total:.4f}, "
              f"transformer {grad_norm_transformer:.4f}, "
              f"adapter {grad_norm_adapter:.4f}")
        print(f"  grad ratio: transformer {grad_ratio_transformer:.4f}, "
              f"adapter {grad_ratio_adapter:.4f}")
        if _diag_clip_pre_total is not None:
            print(f"  grad clip: max_norm {grad_clip:.4f}, "
                  f"pre_total {_diag_clip_pre_total:.4f}, "
                  f"coef {(_diag_clip_coef if _diag_clip_coef is not None else 1.0):.4f}")
        print(f"  <NUM> input tokens: {num_count}/{total_tokens} ({num_count / total_tokens * 100:.1f}%)")
        print(f"  lr: base {lr:.2e}, adapter {adapter_lr:.2e}")
        if _diag_num_inj is not None:
            print(f"  num inject: beta {_diag_num_inj['beta']:.4f}, "
                  f"norm_match {str(bool(num_norm_match)).lower()}, "
                  f"base_norm {_diag_num_inj['base_norm_mean']:.4f}, "
                  f"delta_raw_norm {_diag_num_inj['delta_raw_norm_mean']:.4f}, "
                  f"delta_eff_norm {_diag_num_inj['delta_eff_norm_mean']:.4f}, "
                  f"blend_norm {_diag_num_inj['blended_norm_mean']:.4f}, "
                  f"scale mean/max {_diag_num_inj['norm_scale_mean']:.3f}/"
                  f"{_diag_num_inj['norm_scale_max']:.3f}")
        else:
            print(f"  num inject: beta {current_num_blend_beta:.4f}, "
                  f"norm_match {str(bool(num_norm_match)).lower()}, no <NUM> tokens")

        # Token-level output accuracy
        out_acc = compute_output_accuracy(_diag_logits, _diag_targets)
        if out_acc is not None:
            print(f"  output token accuracy: {out_acc['overall']:.3f} "
                  f"({out_acc['n_correct']}/{out_acc['n_tokens']} tokens)")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "grad/total": grad_norm_total,
                "grad/transformer": grad_norm_transformer,
                "grad/adapter": grad_norm_adapter,
                "grad_ratio/transformer": grad_ratio_transformer,
                "grad_ratio/adapter": grad_ratio_adapter,
                "grad_preclip/total": grad_norm_total_pre,
                "grad_preclip/transformer": grad_norm_transformer_pre,
                "grad_preclip/adapter": grad_norm_adapter_pre,
                "grad_ratio_preclip/transformer": grad_ratio_transformer_pre,
                "grad_ratio_preclip/adapter": grad_ratio_adapter_pre,
                "grad/clip_pre_total": (_diag_clip_pre_total if _diag_clip_pre_total is not None else 0.0),
                "grad/clip_coef": (_diag_clip_coef if _diag_clip_coef is not None else 1.0),
                "lr": lr,
                "lr/base": lr,
                "lr/adapter": adapter_lr,
                "num/beta": current_num_blend_beta,
                "num/norm_match": 1.0 if num_norm_match else 0.0,
            }
            if _diag_num_inj is not None:
                log_dict.update({
                    "num/n_count": float(_diag_num_inj["n_num"]),
                    "num/base_norm_mean": _diag_num_inj["base_norm_mean"],
                    "num/delta_raw_norm_mean": _diag_num_inj["delta_raw_norm_mean"],
                    "num/delta_eff_norm_mean": _diag_num_inj["delta_eff_norm_mean"],
                    "num/blended_norm_mean": _diag_num_inj["blended_norm_mean"],
                    "num/norm_scale_mean": _diag_num_inj["norm_scale_mean"],
                    "num/norm_scale_max": _diag_num_inj["norm_scale_max"],
                    "num/norm_scale_min": _diag_num_inj["norm_scale_min"],
                })
            if out_acc is not None:
                log_dict["output/token_accuracy"] = out_acc['overall']
            wandb.log(log_dict)

    # --- Sample evaluation every sample_interval steps ---
    if iter_num % sample_interval == 0 and iter_num > 0:
        if ddp:
            barrier()
        if master_process:
            eval_samples(raw_model, current_num_blend_beta)
        if ddp:
            barrier()

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
