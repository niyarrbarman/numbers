"""
Baby Luciole (114M Nemotron3) with NumberEncoder v10 adapter.

Standalone PyTorch implementation of Nemotron3 architecture:
  - GQA (24 Q heads, 8 KV heads)
  - LayerNorm (with bias)
  - RoPE (rotary position embeddings)
  - Squared ReLU FFN
  - 12 layers, 768 hidden, 3072 FFN hidden

NumberEncoder injection is identical to GPT-2 FE:
  At <NUM> positions, replace token embedding with adapter(encoder(value)).

References:
  - Nemotron-3: https://arxiv.org/abs/2402.16819
  - Number Embedding System: ../../../np_emb_v10.py
"""

import sys
import os
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add project root so we can import NumberEncoder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from np_emb_v10 import NumberEncoder

NUM_TOKEN_ID = 50257  # GPT-2 vocab is 0..50256; 50257 = <NUM>


# =============================================================================
# Nemotron3 building blocks
# =============================================================================

def rotate_half(x):
    """Rotate half of the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embedding to Q or K."""
    # x: (B, n_head, T, head_dim), cos/sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with caching."""

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(device)
            self._sin_cached = emb.sin().to(device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE.

    24 Q heads share 8 KV heads (3 Q heads per KV group).
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head          # 24
        self.n_kv_head = config.n_kv_head    # 8
        self.head_dim = config.n_embd // config.n_head  # 32
        self.n_rep = self.n_head // self.n_kv_head      # 3

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rotary(T, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads to match Q heads for attention
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)
            k = k.reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)
            v = v.reshape(B, self.n_head, T, self.head_dim)

        # Flash attention (PyTorch >= 2.0)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y


class FeedForward(nn.Module):
    """Squared ReLU FFN (Nemotron style): down(relu(up(x))^2)."""

    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x).square()
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block with GQA + Squared ReLU FFN."""

    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


# =============================================================================
# Config and Model
# =============================================================================

@dataclass
class NemotronConfig:
    block_size: int = 1024
    vocab_size: int = 50304        # GPT-2 50257, padded for efficiency
    n_layer: int = 12
    n_head: int = 24               # query heads
    n_kv_head: int = 8             # KV heads (GQA)
    n_embd: int = 768
    ffn_hidden: int = 3072
    dropout: float = 0.0
    bias: bool = False
    # Number embedding fields
    num_emb_dim: int = 128
    num_emb_checkpoint: str = ''
    num_emb_scale_dims: int = 16
    num_emb_residue_periods: str = '2,5,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000'
    num_norm_match: bool = False
    num_blend_beta_infer: float = 1.0


class Nemotron(nn.Module):
    """Baby Luciole (114M Nemotron3) with NumberEncoder v10 adapter."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # === Transformer (no position embedding — RoPE is in attention) ===
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # === Number embedding components ===
        residue_periods = [int(p) for p in config.num_emb_residue_periods.split(',')]
        self.num_encoder = NumberEncoder(
            embedding_dim=config.num_emb_dim,
            scale_dims=config.num_emb_scale_dims,
            residue_periods=residue_periods,
        )

        self.num_adapter = nn.Sequential(
            nn.Linear(config.num_emb_dim, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # === Weight init ===
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Load encoder checkpoint AFTER init so pretrained weights aren't overwritten
        if config.num_emb_checkpoint:
            ckpt = torch.load(config.num_emb_checkpoint, map_location='cpu',
                              weights_only=False)
            variant = ckpt.get('variant', '')
            if variant in ('v10_high_fidelity_1B', 'v9_math_aware'):
                self.num_encoder.load_state_dict(ckpt['encoder_state_dict'])
                print(f"Loaded NumberEncoder ({variant}) from {config.num_emb_checkpoint}")
            else:
                print(f"WARNING: Unknown encoder variant '{variant}'")

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # No learned position embeddings in Nemotron (RoPE)
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_pretrained_nemotron(self, ckpt_path):
        """Load converted Nemotron checkpoint (from convert_nemo_ckpt.py).

        Handles vocab_size mismatch: original has 50256, ours has 50304.
        Adapter/encoder keys are skipped (loaded separately from num_emb_checkpoint).
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

        own_state = self.state_dict()
        loaded, skipped = 0, 0

        for name, param in state.items():
            if name not in own_state:
                print(f"  skip (unexpected): {name}")
                skipped += 1
                continue

            if own_state[name].shape != param.shape:
                # Vocab size mismatch (50256 → 50304): partial copy
                if 'wte.weight' in name or 'lm_head.weight' in name:
                    with torch.no_grad():
                        own_state[name][:param.shape[0]] = param
                    print(f"  partial: {name} ({param.shape[0]}→{own_state[name].shape[0]})")
                    loaded += 1
                else:
                    print(f"  skip (shape): {name} {param.shape} vs {own_state[name].shape}")
                    skipped += 1
                continue

            own_state[name].copy_(param)
            loaded += 1

        print(f"Loaded {loaded} tensors from {ckpt_path} (skipped {skipped})")

    def forward(self, idx, targets=None, num_values=None, num_mask=None,
                num_blend_beta=None, num_norm_match=None):
        """
        Args:
            idx:        (B, T) int64 token IDs
            targets:    (B, T) int64 target token IDs, or None
            num_values: (B, T) float32 number values aligned with idx
            num_mask:   (B, T) bool, True where idx == NUM_TOKEN_ID
            num_blend_beta: scalar in [0,1], blend factor
            num_norm_match: if True, match adapter norm to base embedding norm
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward seq of length {t}, block size is {self.config.block_size}"

        # Token embeddings (no position embedding — RoPE in attention)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        # Inject number embeddings at <NUM> positions
        if num_mask is not None and num_mask.any():
            num_vals_flat = num_values[num_mask]                    # (K,)
            num_emb = self.num_encoder(num_vals_flat.double())      # v10 needs float64
            num_proj = self.num_adapter(num_emb.float())            # (K, n_embd)

            tok_emb = tok_emb.clone()
            base_num = tok_emb[num_mask]
            delta_num = num_proj.to(tok_emb.dtype)

            if num_norm_match is None:
                num_norm_match = bool(getattr(self.config, 'num_norm_match', False))
            if num_norm_match:
                base_norm = base_num.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                delta_norm = delta_num.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                delta_num = delta_num * (base_norm / delta_norm).to(delta_num.dtype)

            if num_blend_beta is None:
                num_blend_beta = float(getattr(self.config, 'num_blend_beta_infer', 1.0))
            beta = max(0.0, min(1.0, float(num_blend_beta)))
            tok_emb[num_mask] = (1.0 - beta) * base_num + beta * delta_num

        x = self.transformer.drop(tok_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # RoPE cache will be rebuilt on next forward — no parameters to crop

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             adapter_lr_scale=1.0):
        """Set up optimizer with separate LR for adapter vs transformer."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        transformer_decay, transformer_nodecay = [], []
        adapter_decay, adapter_nodecay = [], []
        for name, p in param_dict.items():
            is_adapter = name.startswith('num_adapter.') or name.startswith('num_encoder.')
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
        """Estimate model flops utilization vs A100 bfloat16 peak."""
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
        """Autoregressive generation with number embedding support."""
        if num_values is None:
            num_values = torch.zeros_like(idx, dtype=torch.float32)
        if num_mask is None:
            num_mask = (idx == NUM_TOKEN_ID)

        for _ in range(max_new_tokens):
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
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)
            new_nv = torch.zeros_like(idx_next, dtype=torch.float32)
            new_nm = torch.zeros_like(idx_next, dtype=torch.bool)
            num_values = torch.cat([num_values, new_nv], dim=1)
            num_mask = torch.cat([num_mask, new_nm], dim=1)

        return idx
