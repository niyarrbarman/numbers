"""
Number Embedding v9 — Math-Aware Multi-Lane Encoder
=====================================================
PyTorch implementation (GPU-accelerated).

Redesigned from v8 to produce embeddings that are directly useful for
downstream mathematical reasoning, not just reversible codes.

Key changes from v8:
  1. THREE-LANE architecture (no global LayerNorm):
     - Scale lane   (16 dims): x * w — exactly additive, NOT normalized
     - Residue lane (10 dims): sin/cos at digit-aligned periods (10,100,1K,10K,100K)
     - Semantic lane(102 dims): Fourier + LogMag + Sign + Poly → proj → RMSNorm (per-dim)

  2. MULTI-OBJECTIVE pretraining:
     - L_recon:     reconstruct x from e(x)                        [existing]
     - L_compose:   predict x+y from [e(x), e(y)] via 2-layer probe [new]
     - L_order:     if x<y then readout(e(x)) < readout(e(y))      [new]
     - L_magnitude: predict floor(log10(|x|+eps)) from e(x)        [new]
     - L_spread:    anti-collapse cosine similarity penalty         [existing]

  3. OPERATION-AWARE sampling:
     - Includes (x, y, x+y, x-y) tuples alongside standard samples
     - Carry-heavy numbers (999, 9999, etc.) overrepresented
     - Arithmetic progressions

  4. PROBE-BASED evaluation:
     - Linear addition probe: R² for W @ [e(x); e(y)] → x+y
     - Linear order probe:   Spearman ρ for w @ e(x) → scalar
     - Magnitude classifier:  accuracy for exponent bucket prediction
     - Parity probe:          accuracy for x mod 2 from embedding
     - Digit probe:           accuracy for last digit (x mod 10)

Dimension budget (128 total):
  Lane 1 (Scale):    16 dims — x * learned_weight, no normalization
  Lane 2 (Residue):  10 dims — sin/cos at periods 10, 100, 1K, 10K, 100K
  Lane 3 (Semantic): 102 dims — 101 projected + 1 log_norm, per-dim RMSNorm

Usage:
    python np_emb_v9.py --max-steps 500000
    python np_emb_v9.py --max-steps 500000 --scale-dims 16 --residue-periods 10,100,1000,10000,100000
"""

import math
import time
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist


# =============================================================================
# Device Setup
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Lane 1: Scale Lane (exactly additive, NOT normalized)
# =============================================================================

class ScaleLane(nn.Module):
    """Additive scale lane: e(x) = x * w.

    Satisfies e(x+y) = e(x) + e(y) exactly by construction.
    Weights initialized with diverse scales for multi-resolution coverage.
    NO normalization — raw magnitude preserved.
    """
    def __init__(self, dims: int = 16):
        super().__init__()
        self.output_dim = dims
        # Initialize with log-spaced scales centered for task range (1-100K)
        # For x=100: values ~0.001 to ~1.0
        # For x=100K: values ~1.0 to ~1000
        init = torch.logspace(-5, -2, dims)
        # Alternate signs for diversity
        signs = torch.ones(dims)
        signs[1::2] = -1.0
        self.weight = nn.Parameter(init * signs)

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1) * self.weight  # (N, dims)


# =============================================================================
# Lane 2: Residue Lane (analytic digit features, NOT normalized)
# =============================================================================

class ResidueLane(nn.Module):
    """Modular arithmetic features at digit-aligned periods.

    For each period p, computes [sin(2π·x/p), cos(2π·x/p)].
    For integers, sin(2πx/10) cycles with the last digit,
    sin(2πx/100) with the last two digits, etc.

    These features enable:
      - Carry detection (9→0 transitions)
      - Parity (period 2)
      - Last-digit reasoning
      - Divisibility checks
    """
    def __init__(self, periods: List[int] = None):
        super().__init__()
        if periods is None:
            periods = [10, 100, 1000, 10000, 100000]
        self.register_buffer('periods',
                             torch.tensor(periods, dtype=torch.float32))
        self.output_dim = 2 * len(periods)  # sin + cos per period

    def forward(self, x: Tensor) -> Tensor:
        phases = 2.0 * math.pi * x.unsqueeze(-1) / self.periods  # (N, P)
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (N, 2P)


# =============================================================================
# Lane 3: Semantic Lane channels (analytic — no learnable params)
# =============================================================================

class FourierChannel(nn.Module):
    """Fourier encoding: 32 geometrically-spaced sin/cos pairs (64 dims)."""
    def __init__(self, num_freq: int = 32, freq_base: float = 0.1, freq_scale: float = 1.5):
        super().__init__()
        k = torch.arange(num_freq, dtype=torch.float32)
        self.register_buffer('frequencies', freq_base * (freq_scale ** k))
        self.register_buffer('amplitudes', 1.0 / torch.sqrt(1.0 + k))
        self.output_dim = 2 * num_freq

    def forward(self, x: Tensor) -> Tensor:
        phases = x.unsqueeze(-1) * self.frequencies
        sin_f = torch.sin(phases) * self.amplitudes
        cos_f = torch.cos(phases) * self.amplitudes
        return torch.cat([sin_f, cos_f], dim=-1)


class LogMagnitudeChannel(nn.Module):
    """Log-compressed magnitude: log(|x|+ε) / log(scale)."""
    def __init__(self, epsilon: float = 1e-8, log_scale: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.log_scale = math.log(log_scale)
        self.output_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        return (torch.log(torch.abs(x) + self.epsilon) / self.log_scale).unsqueeze(-1)


class SignChannel(nn.Module):
    """Smooth sign encoding: tanh(αx)."""
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.output_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.alpha * x).unsqueeze(-1)


class PolynomialChannel(nn.Module):
    """Normalized polynomial basis: per-sample norm([x, x², ..., x^p])."""
    def __init__(self, degree: int = 5):
        super().__init__()
        self.degree = degree
        self.output_dim = degree

    def forward(self, x: Tensor) -> Tensor:
        x_clamp = x.clamp(-50.0, 50.0)
        powers = torch.stack([x_clamp ** (k + 1) for k in range(self.degree)], dim=-1)
        mean = powers.mean(dim=-1, keepdim=True)
        var = powers.var(dim=-1, keepdim=True, unbiased=False)
        return (powers - mean) / torch.sqrt(var + 1e-5)


# =============================================================================
# Encoder: Three-Lane Architecture
# =============================================================================

class NumberEncoder(nn.Module):
    """Three-lane number encoder optimised for downstream math tasks.

    Lane 1 (Scale):    x * w — exactly additive, preserves magnitude
    Lane 2 (Residue):  sin/cos at digit periods — captures discrete structure
    Lane 3 (Semantic):  Fourier+LogMag+Sign+Poly → proj → RMSNorm + log_norm

    Total output: scale_dims + residue_dims + semantic_dims = embedding_dim
    """
    def __init__(self, embedding_dim: int = 128, scale_dims: int = 16,
                 residue_periods: List[int] = None,
                 num_frequencies: int = 32, poly_degree: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale_dims = scale_dims

        # Lane 1: Scale
        self.scale_lane = ScaleLane(scale_dims)

        # Lane 2: Residue
        self.residue_lane = ResidueLane(residue_periods)
        residue_dims = self.residue_lane.output_dim

        # Lane 3: Semantic — remaining dims
        self.semantic_dims = embedding_dim - scale_dims - residue_dims
        assert self.semantic_dims >= 2, \
            f"Not enough dims for semantic lane: {self.semantic_dims}"

        # Semantic sub-channels
        self.fourier = FourierChannel(num_freq=num_frequencies)
        self.log_mag = LogMagnitudeChannel()
        self.sign = SignChannel()
        self.poly = PolynomialChannel(poly_degree)
        self.raw_dim = (self.fourier.output_dim + self.log_mag.output_dim
                        + self.sign.output_dim + self.poly.output_dim)

        # Project to (semantic_dims - 1) so we can append log_norm
        proj_out = self.semantic_dims - 1
        self.proj = nn.Linear(self.raw_dim, proj_out)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

        # Per-dim RMSNorm instead of cross-dim LayerNorm
        # Learned scale per dimension, initialized to 1.0
        self.rms_scale = nn.Parameter(torch.ones(proj_out))

    def forward(self, x: Tensor) -> Tensor:
        # Lane 1: Scale (not normalized)
        scale = self.scale_lane(x)                             # (N, scale_dims)

        # Lane 2: Residue (not normalized)
        residue = self.residue_lane(x)                         # (N, residue_dims)

        # Lane 3: Semantic
        raw = torch.cat([
            self.fourier(x),
            self.log_mag(x),
            self.sign(x),
            self.poly(x),
        ], dim=-1)                                              # (N, 71)

        projected = self.proj(raw)                              # (N, semantic_dims - 1)

        # Log-norm (computed BEFORE RMSNorm)
        proj_norm = projected.norm(dim=-1, keepdim=True)
        log_norm = torch.log(proj_norm + 1e-8)                 # (N, 1)

        # Per-dim RMSNorm: divide by RMS, then multiply by learned scale
        # Unlike LayerNorm, this does NOT subtract mean — preserves relative structure
        rms = torch.sqrt(torch.mean(projected ** 2, dim=-1, keepdim=True) + 1e-8)
        normed = (projected / rms) * self.rms_scale            # (N, semantic_dims - 1)

        semantic = torch.cat([normed, log_norm], dim=-1)       # (N, semantic_dims)

        return torch.cat([scale, residue, semantic], dim=-1)   # (N, embedding_dim)


# =============================================================================
# Decoder (same structure as v8, operates on full 128-dim input)
# =============================================================================

class NumberDecoder(nn.Module):
    """MLP decoder with residual skip: emb → MLP(emb) + W_skip(emb) → (log_mag, sign_logit)"""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 192):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.w_skip = nn.Linear(embedding_dim, 2, bias=False)

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.w_skip.weight)

    def forward(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
        a1 = F.gelu(self.fc1(emb))
        a2 = F.gelu(self.fc2(a1))
        z3 = self.fc3(a2) + self.w_skip(emb)
        log_mag = z3[:, 0].clamp(-14.0, 14.0)
        sign_logit = z3[:, 1]
        recon = torch.tanh(sign_logit) * torch.exp(log_mag)
        return recon, sign_logit


# =============================================================================
# Probes (small MLPs, trained jointly, discarded after pretraining)
# =============================================================================

class AdditionProbe(nn.Module):
    """Predict x+y from [e(x); e(y)].  2-layer MLP, 1 scalar output."""
    def __init__(self, embedding_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ex: Tensor, ey: Tensor) -> Tensor:
        return self.net(torch.cat([ex, ey], dim=-1)).squeeze(-1)


class OrderProbe(nn.Module):
    """Scalar readout for ordering: w @ e(x) → scalar."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.w = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, emb: Tensor) -> Tensor:
        return self.w(emb).squeeze(-1)


class MagnitudeProbe(nn.Module):
    """Classify into exponent buckets: floor(log10(|x|+1e-8)) → class.

    Buckets: <-6, -6..-5, ..., 4..5, >5  → 13 classes
    """
    NUM_CLASSES = 13
    EXP_MIN = -6
    EXP_MAX = 5

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, self.NUM_CLASSES)

    def forward(self, emb: Tensor) -> Tensor:
        return self.fc(emb)  # (N, NUM_CLASSES) — raw logits

    @staticmethod
    def label(x: Tensor) -> Tensor:
        """Convert scalar x to bucket index."""
        exp = torch.floor(torch.log10(torch.abs(x) + 1e-8))
        bucket = (exp - MagnitudeProbe.EXP_MIN).long()
        return bucket.clamp(0, MagnitudeProbe.NUM_CLASSES - 1)


# =============================================================================
# Complete System
# =============================================================================

class NumberEmbeddingSystem(nn.Module):
    def __init__(self, embedding_dim: int = 128, scale_dims: int = 16,
                 residue_periods: List[int] = None,
                 num_frequencies: int = 32, poly_degree: int = 5,
                 device: torch.device = None):
        super().__init__()
        if device is None:
            device = get_device()
        self.device = device
        self.embedding_dim = embedding_dim
        self.scale_dims = scale_dims

        self.encoder = NumberEncoder(embedding_dim, scale_dims, residue_periods,
                                     num_frequencies, poly_degree)
        self.decoder = NumberDecoder(embedding_dim)

        # Probes (trained jointly, discarded after pretraining)
        self.addition_probe = AdditionProbe(embedding_dim)
        self.order_probe = OrderProbe(embedding_dim)
        self.magnitude_probe = MagnitudeProbe(embedding_dim)

        self.to(device)

    def encode(self, x: Tensor) -> Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def decode(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
        return self.decoder(emb)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        emb = self.encode(x)
        recon, sign_logit = self.decode(emb)
        return emb, recon, sign_logit

    def compute_loss(self, x: Tensor, emb: Tensor, recon: Tensor,
                     sign_logit: Tensor, lam_rel: float,
                     lam_compose: float, lam_order: float,
                     lam_magnitude: float) -> Tensor:
        eps_lm = 1e-8
        eps_bce = 1e-7

        # ── Term 1: Signed-log MSE (reconstruction) ──
        f_x = torch.sign(x) * torch.log1p(torch.abs(x))
        f_recon = torch.sign(recon) * torch.log1p(torch.abs(recon))
        loss_slog = F.mse_loss(f_recon, f_x)

        # ── Term 2: BCE sign loss ──
        sigma = torch.sigmoid(sign_logit)
        target_sign = torch.where(x > 0, torch.ones_like(x),
                                  torch.where(x < 0, torch.zeros_like(x),
                                              torch.full_like(x, 0.5)))
        loss_bce = F.binary_cross_entropy(sigma.clamp(eps_bce, 1 - eps_bce),
                                          target_sign)

        # ── Term 3: Log-magnitude MSE ──
        log_abs_x = torch.log(torch.abs(x) + eps_lm)
        log_abs_recon = torch.log(torch.abs(recon) + eps_lm)
        loss_lm = F.mse_loss(log_abs_recon, log_abs_x)

        # ── Term 4: Relative MSE (phase 2, ramped) ──
        loss_rel = torch.mean((recon - x) ** 2 / (x * x + 1.0))

        # ── Term 5: Spread loss ──
        n = emb.shape[0]
        idx_perm = torch.randperm(n, device=emb.device)
        emb_shuffled = emb[idx_perm]
        cos_sim = F.cosine_similarity(emb, emb_shuffled, dim=-1)
        loss_spread = torch.mean(cos_sim ** 2)

        # ── Term 6: Composition loss (addition probe) ──
        if lam_compose > 0:
            half = n // 2
            x1, x2 = x[:half], x[half:2 * half]
            emb1, emb2 = emb[:half], emb[half:2 * half]
            pred_sum = self.addition_probe(emb1, emb2)
            target_sum = x1 + x2
            # Use signed-log space for scale-invariant comparison
            f_pred = torch.sign(pred_sum) * torch.log1p(torch.abs(pred_sum))
            f_target = torch.sign(target_sum) * torch.log1p(torch.abs(target_sum))
            loss_compose = F.mse_loss(f_pred, f_target)
        else:
            loss_compose = torch.tensor(0.0, device=emb.device)

        # ── Term 7: Order loss (hinge) ──
        if lam_order > 0:
            half = n // 2
            xa, xb = x[:half], x[half:2 * half]
            emb_a, emb_b = emb[:half], emb[half:2 * half]
            score_a = self.order_probe(emb_a)
            score_b = self.order_probe(emb_b)
            # sign: +1 if xa < xb, -1 if xa > xb, 0 if equal
            diff_sign = torch.sign(xb - xa)
            # Hinge: want (score_b - score_a) * diff_sign > margin
            margin = 0.1
            violations = F.relu(margin - (score_b - score_a) * diff_sign)
            loss_order = violations.mean()
        else:
            loss_order = torch.tensor(0.0, device=emb.device)

        # ── Term 8: Magnitude classification loss ──
        if lam_magnitude > 0:
            mag_logits = self.magnitude_probe(emb)
            mag_labels = MagnitudeProbe.label(x)
            loss_mag = F.cross_entropy(mag_logits, mag_labels)
        else:
            loss_mag = torch.tensor(0.0, device=emb.device)

        return (loss_slog
                + 0.1 * loss_bce
                + 0.3 * loss_lm
                + lam_rel * loss_rel
                + 0.05 * loss_spread
                + lam_compose * loss_compose
                + lam_order * loss_order
                + lam_magnitude * loss_mag)

    def train_model(self, num_steps: int = 500000, batch_size: int = 512,
                    lr: float = 5e-4, log_interval: int = 5000,
                    warmup_steps: int = 2000, grad_clip: float = 1.0):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                       betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=1e-5)
        losses = []

        # Phase schedule
        phase2_start = int(num_steps * 0.40)   # relative MSE ramp start
        phase2_end = int(num_steps * 0.50)     # relative MSE full

        # Multi-objective ramp: 10-20% of training
        obj_start = int(num_steps * 0.10)
        obj_end = int(num_steps * 0.20)

        self.train()
        for step in range(1, num_steps + 1):
            # Relative MSE ramp
            if step < phase2_start:
                lam_rel = 0.0
            elif step < phase2_end:
                lam_rel = 0.3 * (step - phase2_start) / (phase2_end - phase2_start)
            else:
                lam_rel = 0.3

            # Multi-objective ramp
            if step < obj_start:
                obj_frac = 0.0
            elif step < obj_end:
                obj_frac = (step - obj_start) / (obj_end - obj_start)
            else:
                obj_frac = 1.0

            lam_compose = 0.3 * obj_frac
            lam_order = 0.1 * obj_frac
            lam_magnitude = 0.1 * obj_frac

            x = sample_training_numbers(batch_size, self.device)

            optimizer.zero_grad()
            emb, recon, sign_logit = self.forward(x)
            loss = self.compute_loss(x, emb, recon, sign_logit,
                                     lam_rel, lam_compose, lam_order,
                                     lam_magnitude)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            # LR schedule: linear warmup + cosine decay
            if step <= warmup_steps:
                lr_t = lr * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_t

            optimizer.step()

            losses.append(loss.item())
            if step % log_interval == 0:
                avg = sum(losses[-log_interval:]) / log_interval
                phase = ("P1" if step < phase2_start
                         else "ramp" if step < phase2_end
                         else "P2")
                w = self.encoder.scale_lane.weight
                w_min, w_max = w.min().item(), w.max().item()
                obj_tag = f" obj={obj_frac:.1f}" if obj_frac > 0 else ""
                print(f"  Step {step:>6d}/{num_steps} | Loss: {avg:.6f} "
                      f"[{phase}{obj_tag}] | lr: {lr_t:.2e} "
                      f"| scale_w: [{w_min:.2e}, {w_max:.2e}]")

        self.eval()
        return losses


# =============================================================================
# Training Data: Operation-Aware Sampling
# =============================================================================

def sample_training_numbers(batch_size: int, device: torch.device) -> Tensor:
    """Sample numbers from a distribution aligned with math tasks.

    Mix:
      30% positive log-uniform
      30% negative log-uniform
      10% near-zero
      10% integers [-1000, 1000]
      10% operation results (x+y, x-y for random x,y in [-1000,1000])
      10% carry-heavy / structured (999, 9999, powers of 10, etc.)
    """
    n_log = int(batch_size * 0.30)
    n_neg = int(batch_size * 0.30)
    n_zero = int(batch_size * 0.10)
    n_int = int(batch_size * 0.10)
    n_ops = int(batch_size * 0.10)
    n_struct = batch_size - n_log - n_neg - n_zero - n_int - n_ops

    # Standard samples
    pos = torch.exp(torch.empty(n_log, device=device).uniform_(-14, 14))
    neg = -torch.exp(torch.empty(n_neg, device=device).uniform_(-14, 14))
    zero = torch.empty(n_zero, device=device).uniform_(-0.01, 0.01)
    ints = torch.randint(-1000, 1000, (n_int,), device=device, dtype=torch.float32)

    # Operation results: x+y and x-y for random integers
    n_pairs = n_ops // 2
    a = torch.randint(-1000, 1000, (n_pairs,), device=device, dtype=torch.float32)
    b = torch.randint(-1000, 1000, (n_pairs,), device=device, dtype=torch.float32)
    ops = torch.cat([a + b, a - b])
    if ops.shape[0] < n_ops:
        ops = torch.cat([ops, (a[:n_ops - ops.shape[0]] * b[:n_ops - ops.shape[0]])])

    # Structured / carry-heavy numbers
    struct_pool = torch.tensor([
        9, 99, 999, 9999, 99999,
        10, 100, 1000, 10000, 100000,
        -9, -99, -999, -9999, -99999,
        -10, -100, -1000, -10000, -100000,
        1, -1, 0, 50000, -50000,
        11, 111, 1111, 11111,
    ], device=device, dtype=torch.float32)
    idx = torch.randint(0, len(struct_pool), (n_struct,), device=device)
    struct = struct_pool[idx]
    # Add small noise to prevent memorization of exact values
    struct = struct + torch.empty(n_struct, device=device).uniform_(-0.5, 0.5)

    samples = torch.cat([pos, neg, zero, ints, ops, struct])
    perm = torch.randperm(batch_size, device=device)
    return samples[perm]


# =============================================================================
# Helpers
# =============================================================================

def _encode_np(system: NumberEmbeddingSystem, vals) -> np.ndarray:
    with torch.no_grad():
        t = torch.tensor(np.atleast_1d(np.asarray(vals, dtype=np.float32)),
                         device=system.device)
        return system.encode(t).cpu().numpy()


def _forward_np(system: NumberEmbeddingSystem, vals):
    with torch.no_grad():
        t = torch.tensor(np.atleast_1d(np.asarray(vals, dtype=np.float32)),
                         device=system.device)
        emb, recon, _ = system.forward(t)
        return emb.cpu().numpy(), recon.cpu().numpy()


# =============================================================================
# Probe Tests (the new evaluation paradigm)
# =============================================================================

def run_probe_tests(system: NumberEmbeddingSystem, n_train: int = 10000,
                    n_test: int = 5000):
    """Evaluate how useful the embeddings are for downstream math tasks."""
    print("=" * 70)
    print("PROBE TESTS — Can simple functions of embeddings recover math?")
    print("=" * 70)

    device = system.device
    results = {}

    # ── Probe 1: Linear Addition ──
    print("\n  Probe 1: LINEAR ADDITION — W @ [e(x); e(y)] → x+y")
    print("  " + "-" * 60)

    # Generate data
    x1_train = torch.randint(-1000, 1000, (n_train,), device=device, dtype=torch.float32)
    x2_train = torch.randint(-1000, 1000, (n_train,), device=device, dtype=torch.float32)
    x1_test = torch.randint(-1000, 1000, (n_test,), device=device, dtype=torch.float32)
    x2_test = torch.randint(-1000, 1000, (n_test,), device=device, dtype=torch.float32)

    with torch.no_grad():
        e1_train = system.encode(x1_train)
        e2_train = system.encode(x2_train)
        e1_test = system.encode(x1_test)
        e2_test = system.encode(x2_test)

    # Concatenate embeddings
    X_train = torch.cat([e1_train, e2_train], dim=-1)  # (N, 256)
    y_train = (x1_train + x2_train).float()
    X_test = torch.cat([e1_test, e2_test], dim=-1)
    y_test = (x1_test + x2_test).float()

    # Least squares: W = (X^T X)^{-1} X^T y
    XtX = X_train.T @ X_train + 1e-4 * torch.eye(X_train.shape[1], device=device)
    Xty = X_train.T @ y_train
    w_add = torch.linalg.solve(XtX, Xty)

    pred_test = X_test @ w_add
    ss_res = ((y_test - pred_test) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    r2_add = 1.0 - ss_res / (ss_tot + 1e-8)
    mae_add = (y_test - pred_test).abs().mean().item()

    print(f"    R² = {r2_add:.6f}  |  MAE = {mae_add:.2f}")
    results['addition_r2'] = r2_add
    results['addition_mae'] = mae_add

    # Also test subtraction with same linear probe setup
    y_train_sub = (x1_train - x2_train).float()
    y_test_sub = (x1_test - x2_test).float()
    Xty_sub = X_train.T @ y_train_sub
    w_sub = torch.linalg.solve(XtX, Xty_sub)
    pred_sub = X_test @ w_sub
    ss_res_s = ((y_test_sub - pred_sub) ** 2).sum().item()
    ss_tot_s = ((y_test_sub - y_test_sub.mean()) ** 2).sum().item()
    r2_sub = 1.0 - ss_res_s / (ss_tot_s + 1e-8)
    mae_sub = (y_test_sub - pred_sub).abs().mean().item()
    print(f"    Subtraction:  R² = {r2_sub:.6f}  |  MAE = {mae_sub:.2f}")
    results['subtraction_r2'] = r2_sub

    # ── Probe 2: Linear Order ──
    print("\n  Probe 2: LINEAR ORDER — w @ e(x) ranks numbers correctly")
    print("  " + "-" * 60)

    x_order = torch.linspace(-10000, 10000, 2000, device=device)
    with torch.no_grad():
        emb_order = system.encode(x_order)

    # Best linear readout: least squares w @ e(x) ≈ x
    XtX_o = emb_order.T @ emb_order + 1e-4 * torch.eye(system.embedding_dim, device=device)
    Xty_o = emb_order.T @ x_order
    w_ord = torch.linalg.solve(XtX_o, Xty_o)
    pred_order = emb_order @ w_ord

    rho, _ = spearmanr(x_order.cpu().numpy(), pred_order.cpu().numpy())
    print(f"    Spearman ρ = {rho:.6f}")
    results['order_spearman'] = rho

    # ── Probe 3: Magnitude Classification ──
    print("\n  Probe 3: MAGNITUDE — predict exponent bucket from embedding")
    print("  " + "-" * 60)

    x_mag = sample_training_numbers(5000, device)
    with torch.no_grad():
        emb_mag = system.encode(x_mag)
    labels = MagnitudeProbe.label(x_mag)

    # Train a simple linear classifier
    X_mag = emb_mag
    y_mag = labels
    # Split train/test
    n_tr = 4000
    X_tr, X_te = X_mag[:n_tr], X_mag[n_tr:]
    y_tr, y_te = y_mag[:n_tr], y_mag[n_tr:]

    # One-vs-all least squares
    n_cls = MagnitudeProbe.NUM_CLASSES
    Y_oh = F.one_hot(y_tr, n_cls).float()
    XtX_m = X_tr.T @ X_tr + 1e-3 * torch.eye(system.embedding_dim, device=device)
    W_mag = torch.linalg.solve(XtX_m, X_tr.T @ Y_oh)
    pred_mag = (X_te @ W_mag).argmax(dim=-1)
    acc_mag = (pred_mag == y_te).float().mean().item()
    print(f"    Accuracy = {acc_mag:.4f} ({n_cls} classes)")
    results['magnitude_acc'] = acc_mag

    # ── Probe 4: Parity (x mod 2) ──
    print("\n  Probe 4: PARITY — predict x mod 2 from embedding")
    print("  " + "-" * 60)

    x_par = torch.randint(-5000, 5000, (5000,), device=device, dtype=torch.float32)
    with torch.no_grad():
        emb_par = system.encode(x_par)
    y_par = (x_par.long().abs() % 2).float()

    X_tr_p, X_te_p = emb_par[:4000], emb_par[4000:]
    y_tr_p, y_te_p = y_par[:4000], y_par[4000:]

    XtX_p = X_tr_p.T @ X_tr_p + 1e-3 * torch.eye(system.embedding_dim, device=device)
    w_par = torch.linalg.solve(XtX_p, X_tr_p.T @ y_tr_p)
    pred_par = (X_te_p @ w_par > 0.5).float()
    acc_par = (pred_par == y_te_p).float().mean().item()
    print(f"    Accuracy = {acc_par:.4f}")
    results['parity_acc'] = acc_par

    # ── Probe 5: Last Digit (x mod 10) ──
    print("\n  Probe 5: LAST DIGIT — predict |x| mod 10 from embedding")
    print("  " + "-" * 60)

    x_dig = torch.randint(-10000, 10000, (5000,), device=device, dtype=torch.float32)
    with torch.no_grad():
        emb_dig = system.encode(x_dig)
    y_dig = (x_dig.long().abs() % 10)

    X_tr_d, X_te_d = emb_dig[:4000], emb_dig[4000:]
    y_tr_d, y_te_d = y_dig[:4000], y_dig[4000:]

    Y_oh_d = F.one_hot(y_tr_d, 10).float()
    XtX_d = X_tr_d.T @ X_tr_d + 1e-3 * torch.eye(system.embedding_dim, device=device)
    W_dig = torch.linalg.solve(XtX_d, X_tr_d.T @ Y_oh_d)
    pred_dig = (X_te_d @ W_dig).argmax(dim=-1)
    acc_dig = (pred_dig == y_te_d).float().mean().item()
    print(f"    Accuracy = {acc_dig:.4f} (10 classes, chance=0.10)")
    results['digit_acc'] = acc_dig

    # ── Probe 6: Additivity (same as before, for comparison) ──
    print("\n  Probe 6: ADDITIVITY — ||e(x+y) - (e(x)+e(y))|| / ||e(x+y)||")
    print("  " + "-" * 60)

    K = system.scale_dims
    ranges = [
        ("small+small",     -10,      10,      -10,      10),
        ("medium+medium",   -1000,    1000,    -1000,    1000),
        ("large+large",     -100000,  100000,  -100000,  100000),
        ("small+large",     -10,      10,      -10000,   10000),
        ("integers",        -500,     500,     -500,     500),
    ]
    all_full, all_scale = [], []
    for name, lo1, hi1, lo2, hi2 in ranges:
        xa = torch.empty(1000, device=device).uniform_(lo1, hi1)
        xb = torch.empty(1000, device=device).uniform_(lo2, hi2)
        if name == "integers":
            xa = xa.round()
            xb = xb.round()
        with torch.no_grad():
            ea = system.encode(xa)
            eb = system.encode(xb)
            eab = system.encode(xa + xb)
            etarget = ea + eb

            diff_full = (eab - etarget).norm(dim=-1)
            ref_full = eab.norm(dim=-1).clamp(min=1e-8)
            rel_full = (diff_full / ref_full).mean().item()

            diff_scale = (eab[:, :K] - etarget[:, :K]).norm(dim=-1)
            ref_scale = eab[:, :K].norm(dim=-1).clamp(min=1e-8)
            rel_scale = (diff_scale / ref_scale).mean().item()

        all_full.append(rel_full)
        all_scale.append(rel_scale)
        print(f"    {name:>20}:  full={rel_full:.4f}  scale_lane={rel_scale:.6f}")

    results['additivity_full'] = sum(all_full) / len(all_full)
    results['additivity_scale'] = sum(all_scale) / len(all_scale)
    print(f"\n    Overall:  full={results['additivity_full']:.4f}  "
          f"scale_lane={results['additivity_scale']:.6f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("PROBE SUMMARY")
    print("=" * 70)
    print(f"  Addition (linear):      R² = {r2_add:.4f}, MAE = {mae_add:.1f}")
    print(f"  Subtraction (linear):   R² = {r2_sub:.4f}, MAE = {mae_sub:.1f}")
    print(f"  Order (Spearman):       ρ  = {rho:.4f}")
    print(f"  Magnitude (13-class):   acc = {acc_mag:.4f}")
    print(f"  Parity (binary):        acc = {acc_par:.4f}")
    print(f"  Last digit (10-class):  acc = {acc_dig:.4f}")
    print(f"  Additivity (scale):     err = {results['additivity_scale']:.6f}")
    print(f"  Additivity (full):      err = {results['additivity_full']:.4f}")
    print("=" * 70)

    return results


# =============================================================================
# Standard Tests (kept for backward compat)
# =============================================================================

def run_standard_tests(system: NumberEmbeddingSystem):
    """Classic v8 tests: uniqueness, continuity, reversibility, expressiveness."""
    passed = failed = total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")

    # === TEST 1: UNIQUENESS ===
    print("-" * 70)
    print("TEST 1: UNIQUENESS — Distinct numbers → distinct embeddings")
    print("-" * 70)

    nums = np.array([0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, -1.0, -100.0, 3.14159])
    embs = _encode_np(system, nums)

    norms = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
    cos_sim = norms @ norms.T
    np.fill_diagonal(cos_sim, -2.0)
    check("Max cosine sim < 0.99", cos_sim.max() < 0.99,
          f"Max cosine sim: {cos_sim.max():.4f}")
    check("All pairwise L2 distances > 0", pdist(embs).min() > 1e-4,
          f"Min L2 dist: {pdist(embs).min():.6f}")

    a, b = _encode_np(system, [1.0000]), _encode_np(system, [1.0001])
    d = np.linalg.norm(a - b)
    check("1.0000 vs 1.0001 distinguishable", d > 1e-5, f"L2 dist: {d:.8f}")

    # === TEST 2: CONTINUITY ===
    print("-" * 70)
    print("TEST 2: CONTINUITY — Small perturbations → small embedding changes")
    print("-" * 70)

    base_vals = np.array([0.0, 1.0, 10.0, -5.0, 100.0])
    emb_base = _encode_np(system, base_vals)
    prev_max_d = float('inf')
    all_decreasing = True
    eps_results = []
    for eps in [0.1, 0.01, 0.001]:
        emb_p = _encode_np(system, base_vals + eps)
        diffs = np.linalg.norm(emb_base - emb_p, axis=-1)
        max_d = diffs.max()
        rel_d = max_d / np.linalg.norm(emb_base, axis=-1).mean()
        eps_results.append((eps, max_d, rel_d))
        check(f"ε={eps}: bounded relative change", rel_d < 2.0,
              f"Max Δ: {max_d:.4f}, Relative: {rel_d:.4f}")
        if max_d > prev_max_d * 1.1:
            all_decreasing = False
        prev_max_d = max_d

    check("Smaller ε → smaller (or comparable) embedding Δ", all_decreasing,
          f"Deltas: {[f'{r[1]:.4f}' for r in eps_results]}")

    center = np.array([5.0])
    emb_c = _encode_np(system, center)
    dists = [np.linalg.norm(emb_c - _encode_np(system, center + d))
             for d in [0.01, 0.1, 1.0, 10.0]]
    mono = all(dists[i] <= dists[i + 1] * 1.1 for i in range(len(dists) - 1))
    check("Distance monotonicity", mono,
          f"Dists: {[f'{d:.4f}' for d in dists]}")

    # === TEST 3: REVERSIBILITY ===
    print("-" * 70)
    print("TEST 3: REVERSIBILITY — Round-trip reconstruction accuracy")
    print("-" * 70)

    groups = {
        "Small positive": np.array([0.01, 0.05, 0.1]),
        "Medium":         np.array([1.0, 2.5, 3.14159, 7.0]),
        "Large":          np.array([50.0, 100.0, 200.0]),
        "Negative":       np.array([-1.0, -10.0, -50.0]),
        "Near zero":      np.array([-0.01, 0.0, 0.01]),
        "Integers":       np.array([1.0, 2.0, 42.0, 99.0]),
    }
    for name, vals in groups.items():
        _, recon = _forward_np(system, vals)
        abs_err = np.abs(vals - recon)
        rel_err = abs_err / (np.abs(vals) + 1e-8)
        check(f"Reconstruction [{name}]", rel_err.max() < 1.0 or abs_err.max() < 10.0,
              f"|err|_max: {abs_err.max():.4f}, rel_max: {rel_err.max():.4f}")

    showcase = np.array([
        0.0, 1e-6, 1e-4, 0.000123, 0.001, 0.01, 0.1,
        0.5, 1.0, 2.71828, 3.14159, 7.0, 10.0, 42.0, 99.0,
        500.0, 1000.0, 9999.0, 1e5, 1e6,
        -0.001, -0.5, -1.0, -3.14159, -10.0, -42.0, -1000.0, -1234.5,
        0.33333, 0.99999, 1.00001, 123456.0, -99999.0,
    ])
    _, recon_s = _forward_np(system, showcase)
    print(f"\n  {'Input':>12}  →  {'Decoded':>14}  {'Abs Err':>10}  {'Rel Err':>10}")
    print(f"  {'─' * 12}     {'─' * 14}  {'─' * 10}  {'─' * 10}")
    for o, r in zip(showcase, recon_s):
        e = abs(o - r)
        print(f"  {o:>12.5f}  →  {r:>14.5f}  {e:>10.5f}  {e / (abs(o) + 1e-8):>9.4%}")
    print()

    # === TEST 4: EXPRESSIVENESS ===
    print("-" * 70)
    print("TEST 4: EXPRESSIVENESS — Structural properties of embedding space")
    print("-" * 70)

    # Use task-range numbers (not training range up to 1e14) to avoid
    # scale lane dominating SVD with extreme values
    task_range_nums = np.concatenate([
        np.random.uniform(-100000, 100000, 400),
        np.random.uniform(-100, 100, 100),
    ]).astype(np.float32)
    embs_s = _encode_np(system, task_range_nums)
    _, S, _ = np.linalg.svd(embs_s - embs_s.mean(axis=0), full_matrices=False)
    eff = int(np.sum(S > 0.01 * S[0]))
    check("High effective dimensionality (>20)", eff > 20,
          f"Effective dims: {eff}/{system.embedding_dim}")

    ordered = np.linspace(-10, 10, 50)
    embs_o = _encode_np(system, ordered)
    cent = embs_o - embs_o.mean(axis=0)
    _, _, Vt = np.linalg.svd(cent, full_matrices=False)
    pc1 = cent @ Vt[0]
    corr, _ = spearmanr(ordered, pc1)
    check("PC1 correlates with ordering", abs(corr) > 0.7,
          f"Spearman ρ = {corr:.4f}")

    xa = _encode_np(system, [1.0, 1.01, 1.02])
    xb = _encode_np(system, [1000.0, 1000.01, 1000.02])
    intra = max(max(np.linalg.norm(xa[i] - xa[j])
                    for j in range(3) if j != i) for i in range(3))
    inter = min(np.linalg.norm(xa[i] - xb[j])
                for i in range(3) for j in range(3))
    check("Similar numbers cluster tighter", inter > intra,
          f"Intra max: {intra:.4f}, Inter min: {inter:.4f}")

    # === TEST 5: MODEL COMPATIBILITY ===
    print("-" * 70)
    print("TEST 5: MODEL COMPATIBILITY — Standard ops & integration")
    print("-" * 70)

    emb64 = _encode_np(system, np.random.randn(64))
    check("Batch (64,) → (64, d)", emb64.shape == (64, system.embedding_dim))
    check("Single number works",
          _encode_np(system, [42.0]).shape == (1, system.embedding_dim))

    out = emb64 @ np.random.randn(system.embedding_dim, 10) * 0.01
    check("Downstream linear layer", out.shape == (64, 10))

    seq = _encode_np(system, np.random.randn(8))
    dk = system.embedding_dim
    scores = (seq @ seq.T) / np.sqrt(dk)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attn = weights @ seq
    check("Dot-product attention", attn.shape == (8, dk))

    torch.manual_seed(0)
    e1 = _encode_np(system, [1.0, 2.0])
    torch.manual_seed(0)
    e2 = _encode_np(system, [1.0, 2.0])
    check("Deterministic", np.allclose(e1, e2, atol=1e-6))

    # === TEST 6: LANE STRUCTURE ===
    print("-" * 70)
    print("TEST 6: LANE STRUCTURE — Scale lane is linear, residue lane is periodic")
    print("-" * 70)

    K = system.scale_dims
    R = system.encoder.residue_lane.output_dim

    # Scale lane linearity
    test_vals = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    test_embs = _encode_np(system, test_vals)
    add_part = test_embs[:, :K]
    ratios = add_part / test_vals[:, None]
    max_ratio_std = ratios.std(axis=0).max()
    check("Scale lane dims scale linearly with x", max_ratio_std < 0.01,
          f"Max ratio std: {max_ratio_std:.6f}")

    # Residue lane periodicity: e_residue(x) ≈ e_residue(x+10) for period-10 dims
    # ResidueLane output layout: [sin(p1),...,sin(p5), cos(p1),...,cos(p5)]
    # Period-10 is the first period, so sin at index K+0, cos at index K+n_periods
    n_periods = len(system.encoder.residue_lane.periods)
    x_base = torch.tensor([42.0, 137.0, 9999.0], device=system.device)
    x_plus10 = x_base + 10.0
    with torch.no_grad():
        e_base_full = system.encode(x_base)
        e_p10_full = system.encode(x_plus10)
        # sin(2πx/10) at index K, cos(2πx/10) at index K+n_periods
        idx_sin = K
        idx_cos = K + n_periods
        e_base = torch.stack([e_base_full[:, idx_sin], e_base_full[:, idx_cos]], dim=-1)
        e_p10 = torch.stack([e_p10_full[:, idx_sin], e_p10_full[:, idx_cos]], dim=-1)
    period10_err = (e_base - e_p10).abs().max().item()
    check("Residue period-10 dims repeat every 10", period10_err < 0.01,
          f"Max diff: {period10_err:.6f}")

    w = system.encoder.scale_lane.weight.detach().cpu().numpy()
    print(f"  Scale weight range: [{w.min():.2e}, {w.max():.2e}]")

    print("=" * 70)
    print(f"STANDARD TESTS: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 70)

    return passed, failed, total


# =============================================================================
# Checkpoint Saving
# =============================================================================

def save_checkpoint(system: NumberEmbeddingSystem, num_steps: int,
                    checkpoint_dir: str = "/tmpdir/m24047brmn/numbers/checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"np_emb_v9_{num_steps // 1000}k"
    model_path = os.path.join(checkpoint_dir, f"{tag}_model.pt")

    # Save only the encoder state dict (probes and decoder are discarded)
    torch.save({
        'encoder_state_dict': system.encoder.state_dict(),
        'full_state_dict': system.state_dict(),
        'embedding_dim': system.embedding_dim,
        'scale_dims': system.scale_dims,
        'residue_periods': system.encoder.residue_lane.periods.cpu().tolist(),
        'num_steps': num_steps,
        'variant': 'v9_math_aware',
    }, model_path)
    print(f"  Model saved: {model_path}")


# =============================================================================
# Demo
# =============================================================================

def demo(num_steps: int = 500000, scale_dims: int = 16,
         residue_periods: List[int] = None, device: torch.device = None):
    if device is None:
        device = get_device()
    print("=" * 70)
    print("NUMBER EMBEDDING v9 — MATH-AWARE MULTI-LANE ENCODER")
    print("=" * 70)
    print()

    system = NumberEmbeddingSystem(embedding_dim=128, scale_dims=scale_dims,
                                   residue_periods=residue_periods, device=device)
    enc = system.encoder
    residue_dims = enc.residue_lane.output_dim

    total_params = sum(p.numel() for p in system.parameters())
    enc_params = sum(p.numel() for p in enc.parameters())
    probe_params = total_params - enc_params - sum(p.numel() for p in system.decoder.parameters())

    print(f"Architecture:  {enc.raw_dim} raw → {system.embedding_dim}d embedding")
    print(f"  Lane 1 (Scale):    {scale_dims} dims — x * weight, additive")
    print(f"  Lane 2 (Residue):  {residue_dims} dims — sin/cos at periods "
          f"{enc.residue_lane.periods.cpu().tolist()}")
    print(f"  Lane 3 (Semantic): {enc.semantic_dims} dims — "
          f"Fourier({enc.fourier.output_dim}) + LogMag + Sign + Poly → proj → RMSNorm")
    print(f"  Encoder params:    {enc_params:,}")
    print(f"  Probe params:      {probe_params:,} (discarded after pretraining)")
    print(f"  Total params:      {total_params:,}")
    print(f"  Device:            {device}")
    print()

    K = scale_dims
    R = residue_dims
    numbers = [0.0, 1e-6, 0.001, 0.1, 1.0, 3.14159, -42.0, 100.0, 1000.0, 1e5, -1234.5]
    print("Before training:")
    for n in numbers:
        emb = _encode_np(system, [n])[0]
        s_norm = np.linalg.norm(emb[:K])
        r_norm = np.linalg.norm(emb[K:K + R])
        e_norm = np.linalg.norm(emb[K + R:])
        print(f"  {n:>11.5f} → scale={s_norm:7.3f}  residue={r_norm:5.3f}  "
              f"semantic={e_norm:7.3f}")
    print()

    print(f"Training ({num_steps} steps, batch 512)...")
    t0 = time.time()
    system.train_model(num_steps=num_steps, batch_size=512, lr=5e-4,
                       log_interval=max(1, num_steps // 100))
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("After training:")
    print(f"  {'Input':>12}  →  {'Decoded':>14}  {'Error':>10}")
    print(f"  {'─' * 12}     {'─' * 14}  {'─' * 10}")
    for n in numbers:
        _, recon = _forward_np(system, [n])
        print(f"  {n:>12.5f}  →  {recon[0]:>14.5f}  {abs(n - recon[0]):>10.5f}")
    print()

    print("Lane norms after training:")
    for n in numbers:
        emb = _encode_np(system, [n])[0]
        s_norm = np.linalg.norm(emb[:K])
        r_norm = np.linalg.norm(emb[K:K + R])
        e_norm = np.linalg.norm(emb[K + R:])
        print(f"  {n:>11.5f} → scale={s_norm:7.3f}  residue={r_norm:5.3f}  "
              f"semantic={e_norm:7.3f}")
    print()

    return system


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number Embedding v9 — Math-Aware")
    parser.add_argument("--max-steps", type=int, default=500000,
                        help="Training steps (default: 500000)")
    parser.add_argument("--scale-dims", type=int, default=16,
                        help="Scale lane dimensions (default: 16)")
    parser.add_argument("--residue-periods", type=str, default="10,100,1000,10000,100000",
                        help="Comma-separated residue periods (default: 10,100,1000,10000,100000)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load checkpoint and run tests (skip training)")
    parser.add_argument("--test-only", action="store_true",
                        help="Run tests on random (untrained) model")
    parser.add_argument("--demo-only", action="store_true",
                        help="Run demo only, skip tests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    residue_periods = [int(p) for p in args.residue_periods.split(",")]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs",
                            f"run_v9_{time.strftime('%Y%m%d_%H%M%S')}.log")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"Log: {log_path}")
    print(f"Seed: {args.seed}  |  Max steps: {args.max_steps}  |  Device: {device}")
    print(f"Scale dims: {args.scale_dims}  |  Residue periods: {residue_periods}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    try:
        if args.load:
            # Load a saved checkpoint and run tests
            ckpt = torch.load(args.load, map_location=device, weights_only=False)
            scale_dims = ckpt.get('scale_dims', args.scale_dims)
            rp = ckpt.get('residue_periods', residue_periods)
            rp = [int(p) for p in rp]
            print(f"Loading checkpoint: {args.load}")
            print(f"  scale_dims={scale_dims}, residue_periods={rp}")
            print(f"  trained for {ckpt.get('num_steps', '?')} steps")
            print()

            system = NumberEmbeddingSystem(
                embedding_dim=ckpt.get('embedding_dim', 128),
                scale_dims=scale_dims, residue_periods=rp, device=device)
            system.load_state_dict(ckpt['full_state_dict'])
            system.eval()

            run_standard_tests(system)
            print()
            run_probe_tests(system)

        elif args.test_only:
            system = NumberEmbeddingSystem(
                embedding_dim=128, scale_dims=args.scale_dims,
                residue_periods=residue_periods, device=device)
            run_standard_tests(system)
            run_probe_tests(system)
        elif args.demo_only:
            system = demo(num_steps=args.max_steps, scale_dims=args.scale_dims,
                          residue_periods=residue_periods, device=device)
            save_checkpoint(system, args.max_steps)
        else:
            system = demo(num_steps=args.max_steps, scale_dims=args.scale_dims,
                          residue_periods=residue_periods, device=device)
            save_checkpoint(system, args.max_steps)
            print()
            run_standard_tests(system)
            print()
            run_probe_tests(system)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"\nOutput saved to: {log_path}")
