"""
Baby Luciole (114M Nemotron3) with AnalyticNumberCodec + numeric decoder heads.

Standalone PyTorch implementation of Nemotron3 architecture:
  - GQA (24 Q heads, 8 KV heads)
  - LayerNorm1P ((1+weight)*norm+bias, Nemotron style)
  - RoPE (rotary position embeddings)
  - Squared ReLU FFN

Stage 1 analytic number integration:
  - Input: AnalyticNumberCodec (80-d, frozen) → trainable adapter MLP → additive
  - Output: Dual heads — frozen LM head for text, trainable numeric decoder for numbers
  - Numeric decoder: sign (2-class), exponent (65-class), 32 digits (10-class each)

References:
  - Nemotron-3: https://arxiv.org/abs/2402.16819
  - AnalyticNumberCodec: ../../../num_analytic.py
"""

import sys
import os
import math
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add project root so we can import AnalyticNumberCodec
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from num_analytic import AnalyticNumberCodec, NumberComponents
from numeric_surface import (
    SurfaceNumberComponents,
    render_surface_components,
    surface_components_from_value,
    surface_components_to_row,
    row_to_surface_components,
)
from prepare import EOT_TOKEN_ID

NUM_TOKEN_ID = 50256  # luciole_50k vocab is 0..50255; 50256 = <NUM>


# =============================================================================
# Nemotron3 building blocks (identical to model.py)
# =============================================================================

class LayerNorm1P(nn.Module):
    """LayerNorm with (1 + weight) scaling, as used in Nemotron3.

    Forward: (1 + weight) * ((x - mean) / sqrt(var + eps)) + bias
    Weight is initialized to 0 (so effective scale starts at 1).
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.float().mean(-1, keepdim=True)
        var = x.float().var(-1, keepdim=True, correction=0)
        x_norm = (x.float() - mean) / torch.sqrt(var + self.eps)
        return ((1.0 + self.weight.float()) * x_norm + self.bias.float()).type_as(x)


def rotate_half(x):
    """Rotate half of the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embedding to Q or K."""
    cos = cos.unsqueeze(0).unsqueeze(0)
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
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head

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

        cos, sin = self.rotary(T, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)
            k = k.reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)
            v = v.reshape(B, self.n_head, T, self.head_dim)

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
        self.attn_norm = LayerNorm1P(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = LayerNorm1P(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


# =============================================================================
# Config and Model
# =============================================================================

@dataclass
class NemotronAnalyticConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 24
    n_kv_head: int = 8
    n_embd: int = 768
    ffn_hidden: int = 3072
    dropout: float = 0.0
    bias: bool = False
    # Analytic number codec parameters
    analytic_K: int = 32
    analytic_exp_min: int = -32
    analytic_exp_max: int = 32
    numeric_output_mode: str = 'surface'  # 'surface' or 'exponent'
    surface_max_digits: int = 32
    surface_scale_min: int = 0
    surface_scale_max: int = 32
    surface_embed_dim: int = 16
    # Loss weights
    num_loss_lambda: float = 1.0
    num_token_text_weight: float = 1.0
    digit_loss_lambda: float = 1.0 / 32.0
    digit_decay: float = 0.85         # positional weight decay for digit positions
    exp_ordinal_weight: float = 0.1   # weight for ordinal MSE on exponent
    mantissa_weight: float = 0.5      # weight for mantissa regression MSE


class NemotronAnalytic(nn.Module):
    """Baby Luciole (114M Nemotron3) with analytic numeric IO paths.

    The legacy exponent-style numeric decoder is kept for comparison, and a new
    surface-style decoder can be selected via config.numeric_output_mode.
    """

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
            ln_f=LayerNorm1P(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # === Analytic number codec (frozen, deterministic) ===
        self.analytic_codec = AnalyticNumberCodec(
            K=config.analytic_K,
            exp_min=config.analytic_exp_min,
            exp_max=config.analytic_exp_max,
        )
        self.analytic_dim = self.analytic_codec.total_dim  # 80

        # === Numeric adapter: analytic_dim → n_embd (trainable) ===
        self.num_adapter = nn.Sequential(
            nn.Linear(self.analytic_dim, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # === Numeric decoder heads (trainable) ===
        self.num_digit_slots = max(config.analytic_K, config.surface_max_digits)
        n_exp_classes = config.analytic_exp_max - config.analytic_exp_min + 1  # 65
        n_scale_classes = config.surface_scale_max - config.surface_scale_min + 1
        self.num_decoder_sign = nn.Linear(config.n_embd, 2)
        self.num_decoder_exp = nn.Linear(config.n_embd, n_exp_classes)
        self.num_decoder_digits = nn.Linear(config.n_embd, self.num_digit_slots * 10)
        self.num_decoder_scale = nn.Linear(config.n_embd, n_scale_classes)
        self.num_decoder_len = nn.Linear(config.n_embd, config.surface_max_digits + 1)

        # === Structured numeric feedback embedding path (trainable) ===
        e = config.surface_embed_dim
        self.num_surface_sign_embed = nn.Embedding(2, e)
        self.num_surface_scale_embed = nn.Embedding(n_scale_classes, e)
        self.num_surface_len_embed = nn.Embedding(config.surface_max_digits + 1, e)
        self.num_surface_digit_embed = nn.Embedding(10, e)
        surface_flat_dim = (3 + config.surface_max_digits) * e
        self.num_surface_feedback = nn.Sequential(
            nn.Linear(surface_flat_dim, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # === Weight init ===
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_total = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_total / 1e6,))

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        Adapter/decoder keys are skipped (they are trained from scratch).
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

    @torch.compiler.disable
    def _batch_encode_analytic(self, values):
        """Encode a batch of number values using the analytic codec.

        Args:
            values: (K,) tensor of float values
        Returns:
            (K, analytic_dim) tensor of analytic embeddings
        """
        vals_np = values.detach().cpu().to(torch.float64).numpy()
        embeddings = np.stack([self.analytic_codec.encode(float(v)) for v in vals_np])
        return torch.from_numpy(embeddings).float().to(values.device)

    def _batch_embed_surface_rows(self, rows):
        """Embed structured surface rows into numeric feedback vectors."""
        rows = rows.long()
        sign_emb = self.num_surface_sign_embed(rows[:, 0])
        scale_emb = self.num_surface_scale_embed(rows[:, 1])
        len_emb = self.num_surface_len_embed(rows[:, 2].clamp(min=0, max=self.config.surface_max_digits))
        digit_emb = self.num_surface_digit_embed(rows[:, 3:3 + self.config.surface_max_digits])
        flat = torch.cat(
            [sign_emb, scale_emb, len_emb, digit_emb.reshape(rows.size(0), -1)],
            dim=-1,
        )
        return self.num_surface_feedback(flat)

    def _apply_numeric_input_embeddings(
        self,
        tok_emb,
        idx,
        num_values=None,
        num_mask=None,
        num_surface_rows=None,
        num_surface_mask=None,
    ):
        """Inject numeric side-channel embeddings at <NUM> positions."""
        input_num_mask = num_mask if num_mask is not None else (idx == NUM_TOKEN_ID)
        if not input_num_mask.any():
            return tok_emb

        tok_emb = tok_emb.clone()
        surface_mask = None
        if num_surface_mask is not None and num_surface_rows is not None and num_surface_mask.any():
            surface_mask = num_surface_mask & input_num_mask
            if surface_mask.any():
                surface_rows_flat = num_surface_rows[surface_mask]
                surface_proj = self._batch_embed_surface_rows(surface_rows_flat)
                tok_emb[surface_mask] = tok_emb[surface_mask] + surface_proj.to(tok_emb.dtype)

        analytic_mask = input_num_mask
        if surface_mask is not None:
            analytic_mask = input_num_mask & (~surface_mask)
        if analytic_mask.any() and num_values is not None:
            num_vals_flat = num_values[analytic_mask]
            analytic_emb = self._batch_encode_analytic(num_vals_flat)
            num_proj = self.num_adapter(analytic_emb)
            tok_emb[analytic_mask] = tok_emb[analytic_mask] + num_proj.to(tok_emb.dtype)
        return tok_emb

    def compute_hidden_states(
        self,
        idx,
        num_values=None,
        num_mask=None,
        num_surface_rows=None,
        num_surface_mask=None,
    ):
        """Run the transformer and return final hidden states."""
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward seq of length {t}, block size is {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        tok_emb = self._apply_numeric_input_embeddings(
            tok_emb,
            idx,
            num_values=num_values,
            num_mask=num_mask,
            num_surface_rows=num_surface_rows,
            num_surface_mask=num_surface_mask,
        )

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def compute_numeric_loss_from_hidden(
        self,
        hidden_num,
        num_target_components=None,
        num_target_surface=None,
    ):
        """Compute numeric loss from hidden states at output <NUM> positions."""
        if hidden_num.numel() == 0:
            return None

        device = hidden_num.device
        sign_logits = self.num_decoder_sign(hidden_num)
        digit_logits = self.num_decoder_digits(hidden_num)
        digit_logits = digit_logits.view(-1, self.num_digit_slots, 10)

        if self.config.numeric_output_mode == 'surface':
            assert num_target_surface is not None, "surface targets required in surface mode"
            surface = num_target_surface.long()
            sign_targets = surface[:, 0]
            scale_targets = surface[:, 1]
            len_targets = surface[:, 2].clamp(min=1, max=self.config.surface_max_digits)
            digit_targets = surface[:, 3:3 + self.config.surface_max_digits]

            scale_logits = self.num_decoder_scale(hidden_num)
            len_logits = self.num_decoder_len(hidden_num)

            sign_loss = F.cross_entropy(sign_logits, sign_targets)
            scale_loss = F.cross_entropy(scale_logits, scale_targets)
            len_loss = F.cross_entropy(len_logits, len_targets)

            K = self.config.surface_max_digits
            digit_weights = torch.tensor(
                [self.config.digit_decay ** i for i in range(K)],
                device=device, dtype=torch.float32,
            )
            per_digit_ce = F.cross_entropy(
                digit_logits[:, :K, :].reshape(-1, 10),
                digit_targets.reshape(-1),
                reduction='none',
            ).reshape(-1, K)
            active_mask = (torch.arange(K, device=device).unsqueeze(0)
                           < len_targets.unsqueeze(1)).float()
            weighted = per_digit_ce * digit_weights.unsqueeze(0) * active_mask
            denom = (digit_weights.unsqueeze(0) * active_mask).sum(dim=-1).clamp_min(1.0)
            digit_loss = (weighted.sum(dim=-1) / denom).mean()

            total_num_loss = sign_loss + scale_loss + len_loss + digit_loss
            return {
                'sign_loss': sign_loss,
                'scale_loss': scale_loss,
                'len_loss': len_loss,
                'digit_loss': digit_loss,
                'total': total_num_loss,
            }

        assert num_target_components is not None, "legacy analytic components required in exponent mode"
        components = num_target_components.long()
        sign_targets = components[:, 0]
        exp_targets = components[:, 1]
        digit_targets = components[:, 2:]
        digit_logits = digit_logits[:, :self.config.analytic_K, :]

        exp_logits = self.num_decoder_exp(hidden_num)
        sign_loss = F.cross_entropy(sign_logits, sign_targets)

        exp_ce = F.cross_entropy(exp_logits, exp_targets)
        n_exp_classes = self.config.analytic_exp_max - self.config.analytic_exp_min + 1
        exp_class_values = torch.arange(n_exp_classes, device=device, dtype=torch.float32)
        exp_probs = F.softmax(exp_logits, dim=-1)
        expected_exp = (exp_probs * exp_class_values.unsqueeze(0)).sum(dim=-1)
        exp_mse = F.mse_loss(expected_exp, exp_targets.float())
        exp_loss = exp_ce + self.config.exp_ordinal_weight * exp_mse

        K = self.config.analytic_K
        digit_weights = torch.tensor(
            [self.config.digit_decay ** i for i in range(K)],
            device=device, dtype=torch.float32,
        )
        digit_ce_per_pos = F.cross_entropy(
            digit_logits.reshape(-1, 10),
            digit_targets.reshape(-1),
            reduction='none',
        ).reshape(-1, K)
        digit_loss = (digit_ce_per_pos * digit_weights.unsqueeze(0)).sum(dim=-1)
        digit_loss = digit_loss / digit_weights.sum()
        digit_loss = digit_loss.mean()

        digit_values = torch.arange(10, device=device, dtype=torch.float32)
        digit_probs = F.softmax(digit_logits, dim=-1)
        expected_digits = (digit_probs * digit_values).sum(dim=-1)
        mantissa_powers = torch.tensor(
            [10.0 ** (-i) for i in range(K)],
            device=device, dtype=torch.float32,
        )
        expected_mantissa = (expected_digits * mantissa_powers).sum(dim=-1)
        target_mantissa = (digit_targets.float() * mantissa_powers).sum(dim=-1)
        mantissa_mse = F.mse_loss(expected_mantissa, target_mantissa)

        total_num_loss = (
            sign_loss + exp_loss +
            self.config.digit_loss_lambda * self.config.analytic_K * digit_loss +
            self.config.mantissa_weight * mantissa_mse
        )
        return {
            'sign_loss': sign_loss,
            'exp_loss': exp_loss,
            'exp_ce': exp_ce,
            'exp_mse': exp_mse,
            'digit_loss': digit_loss,
            'mantissa_mse': mantissa_mse,
            'total': total_num_loss,
        }

    def decode_numeric_components_from_hidden(self, hidden_states):
        """Decode structured numeric components from hidden states."""
        sign_logits = self.num_decoder_sign(hidden_states)
        digit_logits = self.num_decoder_digits(hidden_states)
        digit_logits = digit_logits.view(-1, self.num_digit_slots, 10)

        signs = sign_logits.argmax(dim=-1)
        digits = digit_logits.argmax(dim=-1)
        results = []

        if self.config.numeric_output_mode == 'surface':
            scale_logits = self.num_decoder_scale(hidden_states)
            len_logits = self.num_decoder_len(hidden_states)
            scales = scale_logits.argmax(dim=-1)
            lengths = len_logits.argmax(dim=-1).clamp(min=1, max=self.config.surface_max_digits)
            for i in range(hidden_states.size(0)):
                sign = 1 if signs[i].item() == 0 else -1
                scale = int(scales[i].item()) + self.config.surface_scale_min
                length = int(lengths[i].item())
                digs = tuple(int(digits[i, j].item()) for j in range(self.config.surface_max_digits))
                results.append(SurfaceNumberComponents(
                    sign=sign,
                    scale=scale,
                    length=length,
                    digits=digs,
                ))
            return results

        exps = self.num_decoder_exp(hidden_states).argmax(dim=-1)
        for i in range(hidden_states.size(0)):
            sign = 1 if signs[i].item() == 0 else -1
            exponent = exps[i].item() + self.config.analytic_exp_min
            digs = [digits[i, j].item() for j in range(self.config.analytic_K)]
            results.append(NumberComponents(sign=sign, exponent=exponent, digits=digs))
        return results

    def render_numeric_components(self, components):
        """Render decoded structured numeric components to canonical text."""
        rendered = []
        for comp in components:
            if isinstance(comp, SurfaceNumberComponents):
                rendered.append(render_surface_components(comp))
            else:
                rendered.append(self.analytic_codec.components_to_plain_string(comp))
        return rendered

    def components_to_feedback_rows(self, components, device=None):
        """Convert structured components into surface feedback rows."""
        rows = []
        for comp in components:
            if isinstance(comp, SurfaceNumberComponents):
                row = surface_components_to_row(
                    comp,
                    max_digits=self.config.surface_max_digits,
                    scale_min=self.config.surface_scale_min,
                    scale_max=self.config.surface_scale_max,
                )
            else:
                value = self.analytic_codec.components_to_plain_string(comp)
                surface = surface_components_from_value(
                    value,
                    max_digits=self.config.surface_max_digits,
                    scale_min=self.config.surface_scale_min,
                    scale_max=self.config.surface_scale_max,
                )
                row = surface_components_to_row(
                    surface,
                    max_digits=self.config.surface_max_digits,
                    scale_min=self.config.surface_scale_min,
                    scale_max=self.config.surface_scale_max,
                )
            rows.append(row)
        tensor = torch.tensor(rows, dtype=torch.long, device=device)
        return tensor

    def forward(
        self,
        idx,
        targets=None,
        num_values=None,
        num_mask=None,
        num_target_components=None,
        num_target_surface=None,
        num_surface_rows=None,
        num_surface_mask=None,
    ):
        """
        Args:
            idx:                  (B, T) int64 token IDs
            targets:              (B, T) int64 target token IDs, or None
            num_values:           (B, T) float32 number values aligned with idx
            num_mask:             (B, T) bool, True where idx == NUM_TOKEN_ID
            num_target_components: (B, T, 34) uint8 — pre-computed [sign, exp, d0..d31]
                                   Only used at training; components at positions where
                                   targets == NUM_TOKEN_ID provide structured labels.
            num_target_surface:   (B, T, 3 + K) uint8 — [sign, scale, length, digits...]
            num_surface_rows:     (B, T, 3 + K) uint8/long surface feedback rows for input NUMs
            num_surface_mask:     (B, T) bool, True where input NUMs should use structured feedback
        Returns:
            logits:        (B, T, vocab_size) text logits
            text_loss:     scalar text CE loss (or None)
            num_loss_dict: dict with sign_loss, exp_loss, digit_loss, total (or None)
        """
        x = self.compute_hidden_states(
            idx,
            num_values=num_values,
            num_mask=num_mask,
            num_surface_rows=num_surface_rows,
            num_surface_mask=num_surface_mask,
        )

        text_loss = None
        num_loss_dict = None

        if targets is not None:
            logits = self.lm_head(x)

            # --- Text loss: include <NUM> targets so the model learns to emit them ---
            target_num_mask = (targets == NUM_TOKEN_ID)
            token_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none',
                ignore_index=-1,
            ).view_as(targets)
            valid_mask = (targets != -1)
            if self.config.num_token_text_weight != 1.0:
                token_weights = valid_mask.to(token_ce.dtype)
                token_weights[target_num_mask] *= self.config.num_token_text_weight
                text_loss = (token_ce * token_weights).sum() / token_weights.sum().clamp_min(1.0)
            else:
                if valid_mask.any():
                    text_loss = token_ce[valid_mask].mean()
                else:
                    text_loss = token_ce.sum() * 0.0

            # --- Numeric loss: at output <NUM> positions ---
            if target_num_mask.any():
                hidden_num = x[target_num_mask]
                target_components = None
                target_surface = None
                if num_target_components is not None:
                    target_components = num_target_components[target_num_mask]
                if num_target_surface is not None:
                    target_surface = num_target_surface[target_num_mask]
                num_loss_dict = self.compute_numeric_loss_from_hidden(
                    hidden_num,
                    num_target_components=target_components,
                    num_target_surface=target_surface,
                )
        else:
            logits = self.lm_head(x[:, [-1], :])

        return logits, text_loss, num_loss_dict

    @torch.compiler.disable
    def decode_numeric_structured(self, hidden_states):
        """Return structured numeric components from hidden states."""
        return self.decode_numeric_components_from_hidden(hidden_states)

    @torch.compiler.disable
    def decode_numeric_output(self, hidden_states):
        """Compatibility wrapper that returns rendered strings."""
        return self.render_numeric_components(self.decode_numeric_structured(hidden_states))

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             adapter_lr_scale=1.0):
        """Set up optimizer with separate LR for adapter/decoder vs transformer."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        transformer_decay, transformer_nodecay = [], []
        adapter_decay, adapter_nodecay = [], []
        for name, p in param_dict.items():
            is_adapter = (name.startswith('num_adapter.')
                          or name.startswith('num_decoder_sign.')
                          or name.startswith('num_decoder_exp.')
                          or name.startswith('num_decoder_digits.')
                          or name.startswith('num_decoder_scale.')
                          or name.startswith('num_decoder_len.')
                          or name.startswith('num_surface_'))
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
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        num_values=None,
        num_mask=None,
        num_surface_rows=None,
        num_surface_mask=None,
        return_components=False,
    ):
        """Autoregressive generation with number embedding support.

        When the model emits <NUM>, runs the numeric decoder to reconstruct
        the number from the hidden state.
        """
        assert idx.size(0) == 1, "generate currently supports batch_size=1"
        if num_values is None:
            num_values = torch.zeros_like(idx, dtype=torch.float32)
        if num_mask is None:
            num_mask = (idx == NUM_TOKEN_ID)
        if num_surface_rows is None:
            num_surface_rows = torch.zeros(
                idx.size(0), idx.size(1), 3 + self.config.surface_max_digits,
                dtype=torch.long, device=idx.device,
            )
        if num_surface_mask is None:
            num_surface_mask = torch.zeros_like(idx, dtype=torch.bool)

        generated_numbers = []  # list of (position, decoded_string)
        generated_components = []  # list of (position, structured components)

        for step in range(max_new_tokens):
            T = idx.size(1)
            if T > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
                nv_cond = num_values[:, -self.config.block_size:]
                nm_cond = num_mask[:, -self.config.block_size:]
                ns_cond = num_surface_rows[:, -self.config.block_size:]
                nsm_cond = num_surface_mask[:, -self.config.block_size:]
            else:
                idx_cond = idx
                nv_cond = num_values
                nm_cond = num_mask
                ns_cond = num_surface_rows
                nsm_cond = num_surface_mask

            hidden = self.compute_hidden_states(
                idx_cond,
                num_values=nv_cond,
                num_mask=nm_cond,
                num_surface_rows=ns_cond,
                num_surface_mask=nsm_cond,
            )
            logits = self.lm_head(hidden[:, [-1], :])[:, -1, :]

            if temperature is None or temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # If model emitted <NUM>, decode the number from hidden state
            next_is_num = idx_next.item() == NUM_TOKEN_ID
            feedback_rows = None
            if idx_next.item() == NUM_TOKEN_ID:
                decoded_components = self.decode_numeric_structured(hidden[:, -1, :])
                decoded_strings = self.render_numeric_components(decoded_components)
                feedback_rows = self.components_to_feedback_rows(decoded_components, device=idx.device)
                generated_numbers.append((T, decoded_strings[0]))
                generated_components.append((T, decoded_components[0]))

            idx = torch.cat([idx, idx_next], dim=1)
            if next_is_num:
                new_nv = torch.zeros_like(idx_next, dtype=torch.float32)
                new_nm = torch.ones_like(idx_next, dtype=torch.bool)
                new_ns = feedback_rows.unsqueeze(1)
                new_nsm = torch.ones_like(idx_next, dtype=torch.bool)
            else:
                new_nv = torch.zeros_like(idx_next, dtype=torch.float32)
                new_nm = torch.zeros_like(idx_next, dtype=torch.bool)
                new_ns = torch.zeros(
                    idx_next.size(0), 1, 3 + self.config.surface_max_digits,
                    dtype=torch.long, device=idx.device,
                )
                new_nsm = torch.zeros_like(idx_next, dtype=torch.bool)
            num_values = torch.cat([num_values, new_nv], dim=1)
            num_mask = torch.cat([num_mask, new_nm], dim=1)
            num_surface_rows = torch.cat([num_surface_rows, new_ns], dim=1)
            num_surface_mask = torch.cat([num_surface_mask, new_nsm], dim=1)
            if idx_next.item() == EOT_TOKEN_ID:
                break

        if return_components:
            return idx, generated_numbers, generated_components
        return idx, generated_numbers
