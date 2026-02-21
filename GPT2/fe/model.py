"""
GPT-2 with Number Embedding Integration (SME output).

Extends the nanoGPT architecture with:
  1. Frozen NumberEncoder — maps scalar values to 128d embeddings
  2. Learned adapter — projects 128d number embeddings to n_embd

At positions where the input contains <NUM> (token 50257), the standard
token embedding is replaced with adapter(encoder(value)). Output numbers
are encoded as SME (Sign-Mantissa-Exponent) text tokens and predicted
through the standard lm_head — no separate number output head needed.

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


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

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
        # 1. Frozen NumberEncoder
        self.num_encoder = NumberEncoder(embedding_dim=config.num_emb_dim)
        if config.num_emb_checkpoint:
            ckpt = torch.load(config.num_emb_checkpoint, map_location='cpu',
                              weights_only=False)
            enc_state = {}
            for k, v in ckpt['state_dict'].items():
                if k.startswith('encoder.'):
                    enc_state[k[len('encoder.'):]] = v
            self.num_encoder.load_state_dict(enc_state)
        for p in self.num_encoder.parameters():
            p.requires_grad = False
        self.num_encoder.eval()

        # 2. Learned adapter: 128d number embedding -> n_embd (2-layer with GELU)
        self.num_adapter = nn.Sequential(
            nn.Linear(config.num_emb_dim, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # === Standard weight init ===
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

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

    def forward(self, idx, targets=None, num_values=None, num_mask=None):
        """
        Args:
            idx:        (B, T) int64 token IDs
            targets:    (B, T) int64 target token IDs, or None
            num_values: (B, T) float32 number values aligned with idx
            num_mask:   (B, T) bool, True where idx == NUM_TOKEN_ID

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
            num_vals_flat = num_values[num_mask]   # (K,)
            with torch.no_grad():
                num_emb = self.num_encoder(num_vals_flat.float())  # (K, 128)
            num_proj = self.num_adapter(num_emb)   # (K, n_embd)
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
        # Pass number embedding config through override_args
        for k in ['num_emb_dim', 'num_emb_checkpoint']:
            if k in override_args:
                config_args[k] = override_args[k]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # Load HuggingFace GPT-2 weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter to transformer keys only
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            # Map HF key to our key (skip if not present in our model)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 2D params get weight decay, 1D params (biases, LN) don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, "
              f"with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
              f"with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                      betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) vs A100 bfloat16 peak."""
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
                 num_values=None, num_mask=None):
        """Generate tokens autoregressively.

        Input numbers are embedded via adapter. Output numbers come as
        SME token sequences — the caller decodes them with sme_tokens_to_number().

        Args:
            idx:        (B, T) initial token IDs
            num_values: (B, T) float values at NUM positions (None = no numbers)
            num_mask:   (B, T) bool mask (None = inferred from idx)

        Returns:
            idx: (B, T + max_new_tokens) generated token IDs
        """
        if num_values is None:
            num_values = torch.zeros_like(idx, dtype=torch.float32)
        if num_mask is None:
            num_mask = (idx == NUM_TOKEN_ID)

        for step in range(max_new_tokens):
            T = idx.size(1)
            if T > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
                nv_cond = num_values[:, -self.config.block_size:]
                nm_cond = num_mask[:, -self.config.block_size:]
            else:
                idx_cond = idx
                nv_cond = num_values
                nm_cond = num_mask

            logits, _ = self(idx_cond, num_values=nv_cond, num_mask=nm_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, idx_next], dim=1)
            # Generated tokens are text (including SME), not <NUM>, so no embedding needed
            new_nv = torch.zeros_like(idx_next, dtype=torch.float32)
            new_nm = torch.zeros_like(idx_next, dtype=torch.bool)
            num_values = torch.cat([num_values, new_nv], dim=1)
            num_mask = torch.cat([num_mask, new_nm], dim=1)

        return idx
