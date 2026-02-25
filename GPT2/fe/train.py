"""
Training script for number-aware GPT-2 (SME output).

Extends nanoGPT's training loop to support dual-stream data:
  - Token IDs ({split}.bin, uint16) with <NUM> placeholders for input numbers
  - Number values ({split}_nums.bin, float32) at <NUM> positions

Output numbers are encoded as SME text tokens in the token stream.
All loss is standard cross-entropy — no separate number loss.

Supports single GPU, DDP, gradient accumulation, mixed precision,
wandb logging, and torch.compile — same as base nanoGPT.

Usage:
  python train.py                          # defaults (scratch)
  python train.py num_emb_checkpoint=path/to/model.pt  # with pretrained encoder
  python train.py init_from=gpt2           # finetune from pretrained GPT-2
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, NUM_TOKEN_ID

# SME token ranges for diagnostics
from prepare import (
    SME_SIGN_POS, SME_SIGN_NEG, SME_EXP_BASE, SME_EXP_OFFSET, SME_N_EXP,
    SME_DIGIT_BASE, SME_END, SME_ALL_TOKENS, sme_tokens_to_number,
    parse_sme_number_tokens,
)


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on numerical tasks
# I/O
out_dir = '/tmpdir/m24047brmn/numbers/model_checkpoints'
eval_interval = 5000
log_interval = 1
diag_interval = 100
sample_interval = 1000
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
resume_ckpt = ''  # explicit checkpoint path for resume (overrides out_dir/ckpt.pt)
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2-sme'
# data
dataset = 'openwebtext'
data_dir = ''  # override to set absolute path; if empty, uses data/{dataset}
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 256
# model
n_layer = 12
n_head = 8
n_embd = 256
dropout = 0.0
bias = False
# number embedding
num_emb_checkpoint = ''  # path to NumberEncoder .pt checkpoint
num_emb_dim = 128
# adamw optimizer
learning_rate = 6e-4
max_iters = 15000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 15000  # should match max_iters for full cosine schedule
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
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    num_emb_dim=num_emb_dim, num_emb_checkpoint=num_emb_checkpoint,
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    ckpt_path = resume_ckpt if resume_ckpt else os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'num_emb_dim', 'num_emb_checkpoint']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout, num_emb_checkpoint=num_emb_checkpoint,
                         num_emb_dim=num_emb_dim)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
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
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, NV, NM = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, num_values=NV, num_mask=NM)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
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


# SME diagnostic helper
_sme_token_set = torch.tensor(sorted(SME_ALL_TOKENS), dtype=torch.long)


@torch.no_grad()
def compute_sme_accuracy(logits, targets):
    """Compute accuracy of SME token predictions.

    Returns overall/sign/exp/digit/end plus d0/d1/d2 digit-position accuracies.
    """
    sme_set = _sme_token_set.to(targets.device)
    sme_mask = torch.isin(targets, sme_set)
    n_sme = sme_mask.sum().item()
    if n_sme == 0:
        return None

    preds = logits.argmax(dim=-1)
    sme_targets = targets[sme_mask]
    sme_preds = preds[sme_mask]

    overall_acc = (sme_preds == sme_targets).float().mean().item()
    sign_mask = (sme_targets == SME_SIGN_POS) | (sme_targets == SME_SIGN_NEG)
    exp_mask = (sme_targets >= SME_EXP_BASE) & (sme_targets < SME_EXP_BASE + SME_N_EXP)
    digit_mask = (sme_targets >= SME_DIGIT_BASE) & (sme_targets <= SME_DIGIT_BASE + 9)
    end_mask = (sme_targets == SME_END)

    def acc(mask):
        if int(mask.sum().item()) == 0:
            return 0.0
        return (sme_preds[mask] == sme_targets[mask]).float().mean().item()

    B, T = targets.shape
    t_cpu = targets.detach().cpu().tolist()
    p_cpu = preds.detach().cpu().tolist()
    d_pos = {"d0": [0, 0], "d1": [0, 0], "d2": [0, 0]}  # correct, total

    for b in range(B):
        row_t = t_cpu[b]
        row_p = p_cpu[b]
        pos = 0
        while pos < T:
            tok = row_t[pos]
            if tok not in (SME_SIGN_POS, SME_SIGN_NEG):
                pos += 1
                continue
            parsed, next_pos = parse_sme_number_tokens(row_t, start_idx=pos)
            if parsed is None:
                pos += 1
                continue

            n_digits = len(parsed) - 3  # remove sign, exp, END
            for di in range(min(3, n_digits)):
                d_key = f"d{di}"
                d_pos[d_key][1] += 1
                d_idx = pos + 2 + di
                if d_idx < T and row_p[d_idx] == row_t[d_idx]:
                    d_pos[d_key][0] += 1

            pos = max(next_pos, pos + 1)

    def d_acc(key):
        correct, total = d_pos[key]
        return (correct / total) if total else 0.0

    return {
        'overall': overall_acc,
        'sign': acc(sign_mask),
        'exp': acc(exp_mask),
        'digit': acc(digit_mask),
        'end': acc(end_mask),
        'd0': d_acc('d0'),
        'd1': d_acc('d1'),
        'd2': d_acc('d2'),
        'n_sme': n_sme,
    }


def sme_token_label(tok_id):
    """Convert SME token ID to human-readable label."""
    if tok_id == SME_SIGN_POS: return 'S+'
    if tok_id == SME_SIGN_NEG: return 'S-'
    if SME_EXP_BASE <= tok_id < SME_EXP_BASE + SME_N_EXP:
        exp = (tok_id - SME_EXP_BASE) - SME_EXP_OFFSET
        return f'E{exp}'
    if SME_DIGIT_BASE <= tok_id <= SME_DIGIT_BASE + 9:
        return f'D{tok_id - SME_DIGIT_BASE}'
    if tok_id == SME_END:
        return 'END'
    return f'?{tok_id}'


def decode_context(token_ids, num_values, enc):
    """Decode a sequence of token IDs to readable text, showing <NUM:val> for number tokens."""
    parts = []
    for i, tok in enumerate(token_ids):
        if tok == NUM_TOKEN_ID:
            val = num_values[i]
            parts.append(f"<{val:g}>")
        elif tok in SME_ALL_TOKENS:
            parts.append(sme_token_label(tok))
        elif tok == 50256:  # EOT
            break
        else:
            try:
                parts.append(enc.decode([tok]))
            except Exception:
                parts.append(f"[{tok}]")
    return ''.join(parts)


@torch.no_grad()
def eval_samples(max_samples=5):
    """Run model on val batch, show full task context with predicted vs target numbers."""
    # Lazy load tiktoken (needs TIKTOKEN_CACHE_DIR set)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    model.eval()
    X, Y, NV, NM = get_batch('val')
    with ctx:
        logits, loss = model(X, Y, num_values=NV, num_mask=NM)
    model.train()

    B, T, _ = logits.shape
    preds = logits.argmax(dim=-1)  # (B, T)

    # Find complete task examples: scan for EOT boundaries in X
    # then find SME numbers within each task
    print(f"  --- Sample eval (val) ---")
    n_shown = 0
    for b in range(B):
        if n_shown >= max_samples:
            break

        y_row = Y[b].tolist()
        p_row = preds[b].tolist()

        # Find first parseable SME number in the target row
        for t in range(T):
            if n_shown >= max_samples:
                break
            target_tok = y_row[t]
            if target_tok not in (SME_SIGN_POS, SME_SIGN_NEG):
                continue
            first_num, _ = parse_sme_number_tokens(y_row, start_idx=t)
            if first_num is None:
                continue

            # Found an SME number — get context before it
            # Scan backwards to find start of this task (EOT or beginning)
            task_start = 0
            for s in range(t - 1, -1, -1):
                if X[b, s].item() == 50256:  # EOT
                    task_start = s + 1
                    break

            # Collect all SME numbers in this task's output
            target_nums = []
            pred_nums = []
            pos = t
            while pos < T:
                tok = y_row[pos]
                if tok in (SME_SIGN_POS, SME_SIGN_NEG):
                    t_sme, next_pos = parse_sme_number_tokens(y_row, start_idx=pos)
                    p_sme, _ = parse_sme_number_tokens(p_row, start_idx=pos)
                    if t_sme is None:
                        pos += 1
                        continue
                    t_val = sme_tokens_to_number(t_sme)
                    p_val = sme_tokens_to_number(p_sme) if p_sme is not None else None
                    if t_val is not None:
                        target_nums.append((t_sme, t_val))
                        pred_nums.append((p_sme, p_val))
                    pos = max(next_pos, pos + 1)
                elif tok == 50256:  # EOT — end of task
                    break
                else:
                    pos += 1

            if not target_nums:
                continue

            # Decode context (input portion up to the SME output)
            if enc is not None:
                ctx_ids = X[b, task_start:t].tolist()
                ctx_nv = NV[b, task_start:t].tolist()
                context = decode_context(ctx_ids, ctx_nv, enc)
            else:
                context = f"[tokens {task_start}:{t}]"

            # Print
            t_vals = [f"{v:.4g}" for _, v in target_nums]
            p_strs = []
            errs = []
            for (p_sme, p_val), (_, t_val) in zip(pred_nums, target_nums):
                if p_val is not None:
                    p_strs.append(f"{p_val:.4g}")
                    errs.append(abs(t_val - p_val))
                else:
                    if p_sme is None:
                        p_labels = "unparseable"
                    else:
                        p_labels = ' '.join(sme_token_label(t) for t in p_sme)
                    p_strs.append(f"INVALID({p_labels})")
                    errs.append(float('inf'))

            print(f"  {context}")
            print(f"    target: {' '.join(t_vals)}")
            print(f"    pred:   {' '.join(p_strs)}")
            if any(e != float('inf') for e in errs):
                err_strs = [f"{e:.2f}" if e != float('inf') else "N/A" for e in errs]
                print(f"    error:  {' '.join(err_strs)}")
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
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
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
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, num_values=NV, num_mask=NM)
            loss = loss / gradient_accumulation_steps
        # snapshot logits for diagnostics (last micro_step only)
        if micro_step == gradient_accumulation_steps - 1 and iter_num % diag_interval == 0 and master_process:
            _diag_logits = logits.detach()
            _diag_targets = Y.clone()
        # async prefetch next batch
        X, Y, NV, NM = get_batch('train')
        # backward pass
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # snapshot grad norms BEFORE zero_grad (for diagnostics)
    if iter_num % diag_interval == 0 and master_process:
        _diag_grad_norms = {}
        _diag_grad_total = 0.0
        for name, p in raw_model.named_parameters():
            if p.grad is not None:
                pnorm = p.grad.data.norm(2).item() ** 2
                _diag_grad_total += pnorm
                if 'num_adapter' in name:
                    _diag_grad_norms['adapter'] = _diag_grad_norms.get('adapter', 0.0) + pnorm
                else:
                    _diag_grad_norms['transformer'] = _diag_grad_norms.get('transformer', 0.0) + pnorm
        _diag_grad_norms = {k: v ** 0.5 for k, v in _diag_grad_norms.items()}
        _diag_grad_norms['total'] = _diag_grad_total ** 0.5
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

        # Count NUM tokens in the batch
        num_count = int(NM.sum().item()) if NM is not None else 0
        total_tokens = batch_size * block_size

        print(f"  === DIAG iter {iter_num} ===")
        print(f"  loss: {loss.item() * gradient_accumulation_steps:.4f}")
        print(f"  grads: total {grad_norm_total:.4f}, "
              f"transformer {grad_norm_transformer:.4f}, "
              f"adapter {grad_norm_adapter:.4f}")
        print(f"  <NUM> input tokens: {num_count}/{total_tokens} ({num_count / total_tokens * 100:.1f}%)")
        print(f"  lr: {lr:.2e}")

        # SME token accuracy
        sme_acc = compute_sme_accuracy(_diag_logits, _diag_targets)
        if sme_acc is not None:
            print(f"  SME accuracy: overall {sme_acc['overall']:.3f}, "
                  f"sign {sme_acc['sign']:.3f}, "
                  f"exp {sme_acc['exp']:.3f}, "
                  f"digit {sme_acc['digit']:.3f}, "
                  f"end {sme_acc['end']:.3f} "
                  f"[d0 {sme_acc['d0']:.3f} d1 {sme_acc['d1']:.3f} d2 {sme_acc['d2']:.3f}] "
                  f"({sme_acc['n_sme']} SME tokens)")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "grad/total": grad_norm_total,
                "grad/transformer": grad_norm_transformer,
                "grad/adapter": grad_norm_adapter,
                "lr": lr,
            }
            if sme_acc is not None:
                log_dict.update({
                    "sme/overall": sme_acc['overall'],
                    "sme/sign": sme_acc['sign'],
                    "sme/exp": sme_acc['exp'],
                    "sme/digit": sme_acc['digit'],
                    "sme/end": sme_acc['end'],
                    "sme/d0": sme_acc['d0'],
                    "sme/d1": sme_acc['d1'],
                    "sme/d2": sme_acc['d2'],
                })
            wandb.log(log_dict)

    # --- Sample evaluation every sample_interval steps ---
    if iter_num % sample_interval == 0 and master_process and iter_num > 0:
        eval_samples()

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
