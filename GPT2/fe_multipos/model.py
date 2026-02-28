"""
GPT-2 with Multi-Position Number Embedding Projection (SME output).

Extends the nanoGPT architecture with:
  1. Trainable NumberEncoder — maps scalar values to 128d embeddings (unfrozen)
  2. K separate projection heads — each projects 128d to n_embd for one position

Each input number occupies k consecutive <NUM> token positions in the sequence.
The encoder produces a single 128d embedding per number, and k different learned
projection heads create k distinct n_embd vectors — one per position. This gives
the transformer k attention targets per number, analogous to how base GPT-2 gets
multiple digit tokens per number.

References:
  - nanoGPT: https://github.com/karpathy/nanoGPT
  - Number Embedding System: ../../np_emb_torch.py
"""

import sys
import os
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add project root so we can import NumberEncoder from np_emb_torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from np_emb_torch import NumberEncoder

NUM_TOKEN_ID = 50257  # GPT-2 vocab is 0..50256; 50257 = <NUM>

# SME token ids used for constrained generation.
SME_SIGN_POS = 50258
SME_SIGN_NEG = 50259
SME_EXP_BASE = 50260
SME_N_EXP = 19
SME_DIGIT_BASE = 50279
SME_END = 50289
SME_MAX_DIGITS = 15


# =============================================================================
# Transformer building blocks (from nanoGPT base.py, unchanged)
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# GPT Config and Model
# =============================================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 50257, padded for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # Number embedding fields
    num_emb_dim: int = 128          # NumberEncoder output dimension
    num_emb_checkpoint: str = ''    # Path to .pt checkpoint for NumberEncoder
    num_positions: int = 5          # Number of token positions per input number


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.num_positions = config.num_positions

        # === Standard transformer ===
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # === Number embedding components ===
        # 1. Trainable NumberEncoder (unfrozen — fine-tuned end-to-end)
        self.num_encoder = NumberEncoder(embedding_dim=config.num_emb_dim)

        # 2. K separate projection heads: each maps 128d → n_embd
        #    Position i uses num_projections[i] to create a position-specific view
        self.num_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.num_emb_dim, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
            )
            for _ in range(config.num_positions)
        ])

        # === Standard weight init ===
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Load encoder checkpoint AFTER apply so pretrained weights aren't overwritten
        if config.num_emb_checkpoint:
            ckpt = torch.load(config.num_emb_checkpoint, map_location='cpu',
                              weights_only=False)
            enc_state = {}
            for k, v in ckpt['state_dict'].items():
                if k.startswith('encoder.'):
                    enc_state[k[len('encoder.'):]] = v
            self.num_encoder.load_state_dict(enc_state)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, num_values=None, num_mask=None,
                pos_indices=None):
        """
        Args:
            idx:         (B, T) int64 token IDs
            targets:     (B, T) int64 target token IDs, or None
            num_values:  (B, T) float32 number values aligned with idx
            num_mask:    (B, T) bool, True where idx == NUM_TOKEN_ID
            pos_indices: (B, T) int8, position index 0..k-1 at NUM positions, -1 elsewhere

        Returns:
            logits: (B, T, vocab_size) or (B, 1, vocab_size) at inference
            loss:   scalar cross-entropy loss, or None at inference
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # --- Token + position embeddings ---
        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)

        # --- Inject number embeddings at <NUM> positions ---
        if num_mask is not None and num_mask.any():
            num_vals_flat = num_values[num_mask]       # (M,) where M = total NUM tokens
            pos_flat = pos_indices[num_mask]            # (M,) values 0..k-1

            # Encode all number values through the shared encoder
            num_emb = self.num_encoder(num_vals_flat.float())  # (M, 128)

            # Apply position-specific projection heads
            num_proj = torch.empty(num_emb.size(0), self.config.n_embd,
                                   device=device, dtype=num_emb.dtype)
            for k in range(self.num_positions):
                mask_k = (pos_flat == k)
                if mask_k.any():
                    # AMP/autocast can return bf16/fp16 here even when num_proj is fp32.
                    proj_k = self.num_projections[k](num_emb[mask_k]).to(num_proj.dtype)
                    num_proj[mask_k] = proj_k

            tok_emb = tok_emb.clone()
            tok_emb[num_mask] = num_proj.to(tok_emb.dtype)

        x = self.transformer.drop(tok_emb + pos_emb)

        # --- Transformer blocks ---
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # --- Training: cross-entropy loss ---
            logits = self.lm_head(x)               # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # --- Inference ---
            logits = self.lm_head(x[:, [-1], :])   # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50304, block_size=1024, bias=True")
        config_args['vocab_size'] = 50304
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        for k in ['num_emb_dim', 'num_emb_checkpoint', 'num_positions']:
            if k in override_args:
                config_args[k] = override_args[k]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if k not in sd:
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             adapter_lr_scale=1.0):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        adapter_lr_scale = float(adapter_lr_scale)
        if adapter_lr_scale <= 0:
            raise ValueError("adapter_lr_scale must be > 0")

        transformer_decay, transformer_nodecay = [], []
        adapter_decay, adapter_nodecay = [], []
        for name, p in param_dict.items():
            is_adapter = (name.startswith('num_projections.')
                          or name.startswith('num_encoder.'))
            use_decay = p.dim() >= 2
            if is_adapter and use_decay:
                adapter_decay.append(p)
            elif is_adapter:
                adapter_nodecay.append(p)
            elif use_decay:
                transformer_decay.append(p)
            else:
                transformer_nodecay.append(p)

        optim_groups = []
        group_specs = [
            ("transformer_decay", transformer_decay, weight_decay, 1.0),
            ("transformer_nodecay", transformer_nodecay, 0.0, 1.0),
            ("adapter_decay", adapter_decay, weight_decay, adapter_lr_scale),
            ("adapter_nodecay", adapter_nodecay, 0.0, adapter_lr_scale),
        ]
        for group_name, params, decay, lr_scale in group_specs:
            if not params:
                continue
            optim_groups.append({
                'params': params,
                'weight_decay': decay,
                'lr': learning_rate * lr_scale,
                'lr_scale': lr_scale,
                'group_name': group_name,
            })

        print("optimizer parameter groups:")
        for group_name, params, decay, lr_scale in group_specs:
            if not params:
                continue
            n_params = sum(p.numel() for p in params)
            print(f"  {group_name}: {len(params)} tensors, {n_params:,} params, "
                  f"weight_decay={decay}, lr_scale={lr_scale}")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                      betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 num_values=None, num_mask=None, pos_indices=None,
                 constrain_sme=True, constrain_sme_max_digits=SME_MAX_DIGITS):
        """Generate tokens autoregressively with multi-position number embeddings."""
        if num_values is None:
            num_values = torch.zeros_like(idx, dtype=torch.float32)
        if num_mask is None:
            num_mask = (idx == NUM_TOKEN_ID)
        if pos_indices is None:
            # Default: assign position 0 to all NUM tokens (single-pos fallback)
            pos_indices = torch.where(
                num_mask,
                torch.zeros_like(idx, dtype=torch.int8),
                torch.full_like(idx, -1, dtype=torch.int8),
            )

        constrain_sme_max_digits = max(
            1,
            min(int(constrain_sme_max_digits), SME_MAX_DIGITS),
        )

        def _advance_sme_state(tok, state, digit_count):
            if state == 0:
                if tok in (SME_SIGN_POS, SME_SIGN_NEG):
                    return 1, 0
                return 0, 0
            if state == 1:
                if SME_EXP_BASE <= tok < SME_EXP_BASE + SME_N_EXP:
                    return 2, 0
                if tok in (SME_SIGN_POS, SME_SIGN_NEG):
                    return 1, 0
                return 0, 0
            # state == 2 (inside mantissa)
            if SME_DIGIT_BASE <= tok <= SME_DIGIT_BASE + 9:
                return 2, min(digit_count + 1, constrain_sme_max_digits)
            if tok == SME_END:
                return 0, 0
            if tok in (SME_SIGN_POS, SME_SIGN_NEG):
                return 1, 0
            return 0, 0

        if constrain_sme:
            B = idx.size(0)
            sme_state = [0] * B
            sme_digit_count = [0] * B
            for b, row in enumerate(idx.detach().cpu().tolist()):
                state = 0
                digit_count = 0
                for tok in row:
                    state, digit_count = _advance_sme_state(int(tok), state, digit_count)
                sme_state[b] = state
                sme_digit_count[b] = digit_count

        for step in range(max_new_tokens):
            T = idx.size(1)
            if T > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
                nv_cond = num_values[:, -self.config.block_size:]
                nm_cond = num_mask[:, -self.config.block_size:]
                pi_cond = pos_indices[:, -self.config.block_size:]
            else:
                idx_cond = idx
                nv_cond = num_values
                nm_cond = num_mask
                pi_cond = pos_indices

            logits, _ = self(idx_cond, num_values=nv_cond, num_mask=nm_cond,
                             pos_indices=pi_cond)
            logits = logits[:, -1, :] / temperature

            if constrain_sme:
                for b in range(logits.size(0)):
                    state = sme_state[b]
                    if state == 0:
                        continue
                    row = logits[b]
                    constrained = torch.full_like(row, -float('inf'))
                    if state == 1:
                        constrained[SME_EXP_BASE:SME_EXP_BASE + SME_N_EXP] = \
                            row[SME_EXP_BASE:SME_EXP_BASE + SME_N_EXP]
                    else:
                        digit_count = sme_digit_count[b]
                        if digit_count <= 0:
                            constrained[SME_DIGIT_BASE:SME_DIGIT_BASE + 10] = \
                                row[SME_DIGIT_BASE:SME_DIGIT_BASE + 10]
                        elif digit_count >= constrain_sme_max_digits:
                            constrained[SME_END] = row[SME_END]
                        else:
                            constrained[SME_DIGIT_BASE:SME_DIGIT_BASE + 10] = \
                                row[SME_DIGIT_BASE:SME_DIGIT_BASE + 10]
                            constrained[SME_END] = row[SME_END]
                    logits[b] = constrained

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if constrain_sme:
                for b, tok in enumerate(idx_next.squeeze(1).detach().cpu().tolist()):
                    state, digit_count = _advance_sme_state(
                        int(tok), sme_state[b], sme_digit_count[b]
                    )
                    sme_state[b] = state
                    sme_digit_count[b] = digit_count

            idx = torch.cat([idx, idx_next], dim=1)
            # Generated tokens are text (including SME), not <NUM>
            new_nv = torch.zeros_like(idx_next, dtype=torch.float32)
            new_nm = torch.zeros_like(idx_next, dtype=torch.bool)
            new_pi = torch.full_like(idx_next, -1, dtype=torch.int8)
            num_values = torch.cat([num_values, new_nv], dim=1)
            num_mask = torch.cat([num_mask, new_nm], dim=1)
            pos_indices = torch.cat([pos_indices, new_pi], dim=1)

        return idx
