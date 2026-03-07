"""
Number Embedding v10 — High-Fidelity Math-Aware Encoder (1B Range)
==================================================================
PyTorch implementation (GPU-accelerated).

Major changes from v9:
  1. EXTENDED RESIDUE LANE (22 dims, 11 periods: 2,5,10,...,1e9):
     - Float64 computation for exact modular arithmetic at 1e9
     - Period 2 (parity) and period 5 (mod 5) added
     - Full digit coverage up to 10 digits

  2. FIXED FOURIER CHANNEL (operates on log10(|x|+1)):
     - No more float32 precision catastrophe at large x
     - 33 log-spaced frequencies from 0.5 to 500
     - Phase stays bounded: max = 9 * 500 = 4500 (safe for float32)

  3. POLYNOMIAL REMOVED:
     - Was dead for |x|>50 (clamped), contributed nothing
     - 5 freed dims redistributed to residue (+4) and Fourier (+1 freq)

  4. NO RMSNorm ON CONCATENATION:
     - v9 RMSNorm caused dimension collapse (3/128 effective dims)
     - Replaced with per-dim learned scale (no division by RMS)

  5. LANE-SPECIFIC AUXILIARY LOSSES:
     - L_digit: predict each digit from RESIDUE lane only (forces utilization)
     - L_decorr: decorrelation penalty (prevents dimension collapse)
     - L_subtraction: subtraction probe (complements addition)

  6. EXPANDED TRAINING:
     - 2M steps default (4x more than v9)
     - Digit-uniform sampling: equal probability for 1-9 digit numbers
     - Full [0, 1e9] range with carry-heavy/structured at all scales

  7. COMPREHENSIVE TEST SUITE (18 tests):
     - Per-digit accuracy, parity, cross-scale generalization
     - Float32 stability, lane independence, discriminability
     - MIN/MAX probes, carry detection

Dimension budget (128 total):
  Lane 1 (Scale):    16 dims — x * learned_weight, no normalization
  Lane 2 (Residue):  22 dims — sin/cos at 11 periods (2,5,10,...,1e9), float64
  Lane 3 (Semantic): 90 dims — Fourier(66d) + LogMag(1d) + Sign(1d) → proj(89d) + log_norm(1d)

Usage:
    python np_emb_v10.py --max-steps 2000000
    python np_emb_v10.py --load checkpoint.pt    # test only
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
from typing import Tuple, List

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
        init = torch.logspace(-5, -2, dims)
        signs = torch.ones(dims)
        signs[1::2] = -1.0
        self.weight = nn.Parameter(init * signs)

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1) * self.weight  # (N, dims)


# =============================================================================
# Lane 2: Residue Lane (float64 for exact modular arithmetic)
# =============================================================================

class ResidueLane(nn.Module):
    """Modular arithmetic features at digit-aligned periods.

    For each period p, computes [sin(2π·x/p), cos(2π·x/p)] in float64
    for exact integer representation up to 2^53 (9e15), then casts to float32.

    Periods: {2, 5, 10, 100, 1K, 10K, 100K, 1M, 10M, 100M, 1B}
    - Period 2: parity (odd/even)
    - Period 5: mod-5 structure
    - Period 10-1B: individual digit extraction for up to 10-digit numbers
    """
    def __init__(self, periods: List[int] = None):
        super().__init__()
        if periods is None:
            periods = [2, 5, 10, 100, 1000, 10000, 100000,
                       1000000, 10000000, 100000000, 1000000000]
        # Store as float64 for precise division
        self.register_buffer('periods',
                             torch.tensor(periods, dtype=torch.float64))
        self.output_dim = 2 * len(periods)  # sin + cos per period

    def forward(self, x: Tensor) -> Tensor:
        # Cast to float64 for precise modular arithmetic at large x
        # float32 fails for x=1e9: phase = 2π*1e9/10 = 6.28e8, precision ±75 radians
        # float64 is exact for integers up to 2^53 ≈ 9e15
        x_64 = x.double().unsqueeze(-1)
        phases = 2.0 * math.pi * x_64 / self.periods  # (N, P)
        result = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return result.float()  # (N, 2P), back to float32


# =============================================================================
# Lane 3: Semantic Lane channels
# =============================================================================

class FourierChannel(nn.Module):
    """Fourier encoding on log-magnitude: sin/cos(log10(|x|+1) * freq).

    Operates on log10(|x|+1) instead of raw x to avoid float32 precision
    catastrophe. For x up to 1e9, log10(|x|+1) ∈ [0, 9], so max phase
    = 9 * 500 = 4500 — well within float32 precision.

    33 log-spaced frequencies from 0.5 to 500 with amplitude decay.
    """
    def __init__(self, num_freq: int = 33, freq_min: float = 0.5,
                 freq_max: float = 500.0):
        super().__init__()
        freqs = torch.logspace(math.log10(freq_min), math.log10(freq_max),
                               num_freq)
        self.register_buffer('frequencies', freqs)
        k = torch.arange(num_freq, dtype=torch.float32)
        self.register_buffer('amplitudes', 1.0 / torch.sqrt(1.0 + k))
        self.output_dim = 2 * num_freq  # sin + cos

    def forward(self, x: Tensor) -> Tensor:
        log_x = torch.log10(torch.abs(x) + 1.0)  # [0, ~9] for |x| up to 1e9
        phases = log_x.unsqueeze(-1) * self.frequencies  # (N, F)
        sin_f = torch.sin(phases) * self.amplitudes
        cos_f = torch.cos(phases) * self.amplitudes
        return torch.cat([sin_f, cos_f], dim=-1)  # (N, 2F)


class LogMagnitudeChannel(nn.Module):
    """Log-compressed magnitude: log10(|x|+1), output ∈ [0, ~9]."""
    def __init__(self):
        super().__init__()
        self.output_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        return torch.log10(torch.abs(x) + 1.0).unsqueeze(-1)


class SignChannel(nn.Module):
    """Smooth sign encoding: tanh(αx)."""
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.output_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.alpha * x).unsqueeze(-1)


# =============================================================================
# Encoder: Three-Lane Architecture (v10)
# =============================================================================

class NumberEncoder(nn.Module):
    """Three-lane number encoder for downstream math tasks (1B range).

    Lane 1 (Scale):    x * w — exactly additive, preserves magnitude
    Lane 2 (Residue):  sin/cos at 11 periods — digit structure, float64
    Lane 3 (Semantic):  Fourier(log-space) + LogMag + Sign → proj + scale + log_norm

    No RMSNorm. Per-dim learned scaling on semantic projection instead.
    """
    def __init__(self, embedding_dim: int = 128, scale_dims: int = 16,
                 residue_periods: List[int] = None,
                 num_frequencies: int = 33):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale_dims = scale_dims

        # Lane 1: Scale
        self.scale_lane = ScaleLane(scale_dims)

        # Lane 2: Residue
        self.residue_lane = ResidueLane(residue_periods)
        self.residue_dims = self.residue_lane.output_dim

        # Lane 3: Semantic — remaining dims
        self.semantic_dims = embedding_dim - scale_dims - self.residue_dims
        assert self.semantic_dims >= 2, \
            f"Not enough dims for semantic lane: {self.semantic_dims}"

        # Semantic sub-channels (no polynomial)
        self.fourier = FourierChannel(num_freq=num_frequencies)
        self.log_mag = LogMagnitudeChannel()
        self.sign = SignChannel()
        self.raw_dim = (self.fourier.output_dim + self.log_mag.output_dim
                        + self.sign.output_dim)

        # Project to (semantic_dims - 1) so we can append log_norm
        proj_out = self.semantic_dims - 1
        self.proj = nn.Linear(self.raw_dim, proj_out)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

        # Per-dim learned scale (NOT RMSNorm — no division by RMS)
        # Acts as soft attention over projected features
        self.dim_scale = nn.Parameter(torch.ones(proj_out))

    def forward(self, x: Tensor) -> Tensor:
        # Lane 1: Scale (not normalized)
        scale = self.scale_lane(x)                             # (N, 16)

        # Lane 2: Residue (float64 internally, float32 output)
        residue = self.residue_lane(x)                         # (N, 22)

        # Lane 3: Semantic (no polynomial, no RMSNorm)
        raw = torch.cat([
            self.fourier(x),
            self.log_mag(x),
            self.sign(x),
        ], dim=-1)                                              # (N, 68)

        projected = self.proj(raw)                              # (N, 89)

        # Per-dim scale (learned, no normalization)
        scaled = projected * self.dim_scale                     # (N, 89)

        # Log-norm captures magnitude of projection
        proj_norm = projected.norm(dim=-1, keepdim=True)
        log_norm = torch.log(proj_norm + 1e-8)                 # (N, 1)

        semantic = torch.cat([scaled, log_norm], dim=-1)       # (N, 90)

        return torch.cat([scale, residue, semantic], dim=-1)   # (N, 128)


# =============================================================================
# Decoder
# =============================================================================

class NumberDecoder(nn.Module):
    """MLP decoder: emb → (log_mag, sign_logit) → reconstructed scalar.

    Wider hidden layer (256) for 1e9 range reconstruction.
    """
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
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
        log_mag = z3[:, 0].clamp(-25.0, 25.0)  # Extended for 1e9
        sign_logit = z3[:, 1]
        recon = torch.tanh(sign_logit) * torch.exp(log_mag)
        return recon, sign_logit


# =============================================================================
# Probes (trained jointly, discarded after pretraining)
# =============================================================================

class DigitProbe(nn.Module):
    """Predict individual digits of |x| from the RESIDUE lane only.

    One 10-class classifier per digit position (ones, tens, ..., billions).
    Input is the residue lane slice of the embedding, not the full embedding.
    This forces the residue lane to encode digit structure.
    """
    def __init__(self, residue_dim: int = 22, num_positions: int = 10):
        super().__init__()
        self.num_positions = num_positions
        self.classifiers = nn.ModuleList([
            nn.Linear(residue_dim, 10) for _ in range(num_positions)
        ])

    def forward(self, residue_emb: Tensor) -> List[Tensor]:
        """Returns list of (N, 10) logits, one per digit position."""
        return [clf(residue_emb) for clf in self.classifiers]

    @staticmethod
    def labels(x: Tensor, num_positions: int = 10) -> List[Tensor]:
        """Extract digit labels: digit_d = floor(|x|) // 10^d % 10."""
        abs_x = x.abs().long()
        digits = []
        for d in range(num_positions):
            digit = (abs_x // (10 ** d)) % 10
            digits.append(digit)
        return digits


class AdditionProbe(nn.Module):
    """Predict x+y from [e(x); e(y)]. 2-layer MLP, 1 scalar output."""
    def __init__(self, embedding_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ex: Tensor, ey: Tensor) -> Tensor:
        return self.net(torch.cat([ex, ey], dim=-1)).squeeze(-1)


class SubtractionProbe(nn.Module):
    """Predict x-y from [e(x); e(y)]. 2-layer MLP, 1 scalar output."""
    def __init__(self, embedding_dim: int = 128, hidden: int = 256):
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
    """Classify into exponent buckets: floor(log10(|x|+ε)) → class.

    For 1e9 range: exponents -2 to 9, plus overflow.
    Buckets: <-2, [-2,-1), [-1,0), [0,1), ..., [8,9), >=9  → 13 classes
    """
    NUM_CLASSES = 13
    EXP_MIN = -2
    EXP_MAX = 9

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, self.NUM_CLASSES)

    def forward(self, emb: Tensor) -> Tensor:
        return self.fc(emb)

    @staticmethod
    def label(x: Tensor) -> Tensor:
        exp = torch.floor(torch.log10(torch.abs(x) + 1e-8))
        bucket = (exp - MagnitudeProbe.EXP_MIN).long()
        return bucket.clamp(0, MagnitudeProbe.NUM_CLASSES - 1)


# =============================================================================
# Training Data: Digit-Uniform Sampling for 1B Range
# =============================================================================

def sample_training_numbers(batch_size: int, device: torch.device) -> Tensor:
    """Sample numbers from a distribution covering full [0, 1e9] range.

    Mix:
      35% log-uniform [1, 1e9] with random sign
      10% small numbers [-100, 100]
       5% near-zero [-1, 1]
      10% exact integers (by digit count, for digit probes)
      15% operation results at scale (x±y)
      10% carry-heavy / structured at all scales
      15% digit-uniform exact integers [1, 1e9]
    """
    n_logu = int(batch_size * 0.35)
    n_small = int(batch_size * 0.10)
    n_zero = int(batch_size * 0.05)
    n_int = int(batch_size * 0.10)
    n_ops = int(batch_size * 0.15)
    n_struct = int(batch_size * 0.10)
    n_digit = batch_size - n_logu - n_small - n_zero - n_int - n_ops - n_struct

    # Log-uniform with random sign: 10^U(0,9) ∈ [1, 1e9]
    exp_u = torch.empty(n_logu, device=device).uniform_(0.0, 9.0)
    logu = 10.0 ** exp_u
    logu_signs = 2 * torch.randint(0, 2, (n_logu,), device=device,
                                    dtype=torch.float32) - 1
    logu = logu * logu_signs

    # Small numbers
    small = torch.empty(n_small, device=device).uniform_(-100.0, 100.0)

    # Near-zero
    zero = torch.empty(n_zero, device=device).uniform_(-1.0, 1.0)

    # Exact integers by digit count (important for digit probes)
    ints = torch.zeros(n_int, device=device)
    int_ndigits = torch.randint(1, 10, (n_int,), device=device)
    for d_val in range(1, 10):
        mask = (int_ndigits == d_val)
        count = mask.sum().item()
        if count > 0:
            lo = 10 ** (d_val - 1) if d_val > 1 else 0
            hi = 10 ** d_val
            vals = torch.randint(lo, hi, (count,), device=device,
                                 dtype=torch.float32)
            ints[mask] = vals
    int_signs = 2 * torch.randint(0, 2, (n_int,), device=device,
                                   dtype=torch.float32) - 1
    ints = ints * int_signs

    # Operation results at scale: x±y where x,y are log-uniform
    n_pairs = n_ops // 2
    exp_a = torch.empty(n_pairs, device=device).uniform_(0.0, 9.0)
    exp_b = torch.empty(n_pairs, device=device).uniform_(0.0, 9.0)
    a = 10.0 ** exp_a
    b = 10.0 ** exp_b
    a_signs = 2 * torch.randint(0, 2, (n_pairs,), device=device,
                                 dtype=torch.float32) - 1
    b_signs = 2 * torch.randint(0, 2, (n_pairs,), device=device,
                                 dtype=torch.float32) - 1
    a = a * a_signs
    b = b * b_signs
    ops = torch.cat([a + b, a - b])[:n_ops]

    # Structured / carry-heavy at all scales
    struct_pool = []
    for e in range(10):  # 10^0 through 10^9
        base = 10 ** e
        struct_pool.extend([base - 1, base, -(base - 1), -base])
    struct_pool.extend([0, 1, -1, 2, -2, 5, -5])
    # Repunits and all-nines
    for d in range(1, 10):
        struct_pool.append(int('1' * d))
        struct_pool.append(int('9' * d))
    # Powers of 10
    for e in range(10):
        struct_pool.append(10 ** e)
    struct_pool = torch.tensor(struct_pool, device=device, dtype=torch.float32)
    idx = torch.randint(0, len(struct_pool), (n_struct,), device=device)
    struct = struct_pool[idx]
    struct = struct + torch.empty(n_struct, device=device).uniform_(-0.5, 0.5)

    # Digit-uniform exact integers (extra batch for digit training)
    digit_nums = torch.zeros(n_digit, device=device)
    digit_ndigits = torch.randint(1, 10, (n_digit,), device=device)
    for d_val in range(1, 10):
        mask = (digit_ndigits == d_val)
        count = mask.sum().item()
        if count > 0:
            lo = 10 ** (d_val - 1) if d_val > 1 else 0
            hi = 10 ** d_val
            vals = torch.randint(lo, hi, (count,), device=device,
                                 dtype=torch.float32)
            digit_nums[mask] = vals
    d_signs = 2 * torch.randint(0, 2, (n_digit,), device=device,
                                 dtype=torch.float32) - 1
    digit_nums = digit_nums * d_signs

    samples = torch.cat([logu, small, zero, ints, ops, struct, digit_nums])
    return samples[torch.randperm(batch_size, device=device)]


# =============================================================================
# Complete System
# =============================================================================

class NumberEmbeddingSystem(nn.Module):
    def __init__(self, embedding_dim: int = 128, scale_dims: int = 16,
                 residue_periods: List[int] = None,
                 num_frequencies: int = 33,
                 device: torch.device = None):
        super().__init__()
        if device is None:
            device = get_device()
        self.device = device
        self.embedding_dim = embedding_dim
        self.scale_dims = scale_dims

        self.encoder = NumberEncoder(embedding_dim, scale_dims,
                                     residue_periods, num_frequencies)
        self.decoder = NumberDecoder(embedding_dim)
        self.residue_dims = self.encoder.residue_dims

        # Probes (trained jointly, discarded after pretraining)
        self.addition_probe = AdditionProbe(embedding_dim)
        self.subtraction_probe = SubtractionProbe(embedding_dim)
        self.order_probe = OrderProbe(embedding_dim)
        self.magnitude_probe = MagnitudeProbe(embedding_dim)
        self.digit_probe = DigitProbe(self.residue_dims)

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

    def _get_residue_slice(self, emb: Tensor) -> Tensor:
        """Extract residue lane dims from full embedding."""
        k = self.scale_dims
        return emb[:, k:k + self.residue_dims]

    def compute_loss(self, x: Tensor, emb: Tensor, recon: Tensor,
                     sign_logit: Tensor, lam_rel: float,
                     lam_compose: float, lam_order: float,
                     lam_magnitude: float, lam_digit: float,
                     lam_decorr: float, lam_sub: float) -> Tensor:
        eps_lm = 1e-8
        eps_bce = 1e-7

        # ── Core losses (same as v9) ──

        # 1. Signed-log MSE (reconstruction)
        f_x = torch.sign(x) * torch.log1p(torch.abs(x))
        f_recon = torch.sign(recon) * torch.log1p(torch.abs(recon))
        loss_slog = F.mse_loss(f_recon, f_x)

        # 2. BCE sign loss
        sigma = torch.sigmoid(sign_logit)
        target_sign = torch.where(x > 0, torch.ones_like(x),
                                  torch.where(x < 0, torch.zeros_like(x),
                                              torch.full_like(x, 0.5)))
        loss_bce = F.binary_cross_entropy(sigma.clamp(eps_bce, 1 - eps_bce),
                                          target_sign)

        # 3. Log-magnitude MSE
        log_abs_x = torch.log(torch.abs(x) + eps_lm)
        log_abs_recon = torch.log(torch.abs(recon) + eps_lm)
        loss_lm = F.mse_loss(log_abs_recon, log_abs_x)

        # 4. Relative MSE (phase 2, ramped)
        loss_rel = torch.mean((recon - x) ** 2 / (x * x + 1.0))

        # 5. Spread loss
        n = emb.shape[0]
        idx_perm = torch.randperm(n, device=emb.device)
        emb_shuffled = emb[idx_perm]
        cos_sim = F.cosine_similarity(emb, emb_shuffled, dim=-1)
        loss_spread = torch.mean(cos_sim ** 2)

        # ── Multi-objective losses ──

        # 6. Composition loss (addition probe)
        if lam_compose > 0:
            half = n // 2
            x1, x2 = x[:half], x[half:2 * half]
            emb1, emb2 = emb[:half], emb[half:2 * half]
            pred_sum = self.addition_probe(emb1, emb2)
            target_sum = x1 + x2
            f_pred = torch.sign(pred_sum) * torch.log1p(torch.abs(pred_sum))
            f_target = torch.sign(target_sum) * torch.log1p(torch.abs(target_sum))
            loss_compose = F.mse_loss(f_pred, f_target)
        else:
            loss_compose = torch.tensor(0.0, device=emb.device)

        # 7. Order loss (hinge)
        if lam_order > 0:
            half = n // 2
            xa, xb = x[:half], x[half:2 * half]
            emb_a, emb_b = emb[:half], emb[half:2 * half]
            score_a = self.order_probe(emb_a)
            score_b = self.order_probe(emb_b)
            diff_sign = torch.sign(xb - xa)
            margin = 0.1
            violations = F.relu(margin - (score_b - score_a) * diff_sign)
            loss_order = violations.mean()
        else:
            loss_order = torch.tensor(0.0, device=emb.device)

        # 8. Magnitude classification
        if lam_magnitude > 0:
            mag_logits = self.magnitude_probe(emb)
            mag_labels = MagnitudeProbe.label(x)
            loss_mag = F.cross_entropy(mag_logits, mag_labels)
        else:
            loss_mag = torch.tensor(0.0, device=emb.device)

        # ── NEW: Lane-specific losses ──

        # 9. Digit classification from RESIDUE lane only
        if lam_digit > 0:
            residue_emb = self._get_residue_slice(emb)
            digit_logits = self.digit_probe(residue_emb)
            digit_labels = DigitProbe.labels(x, self.digit_probe.num_positions)
            loss_digit = torch.tensor(0.0, device=emb.device)
            for pos, (logits, labels) in enumerate(
                    zip(digit_logits, digit_labels)):
                loss_digit = loss_digit + F.cross_entropy(logits, labels)
            loss_digit = loss_digit / len(digit_logits)  # Average over positions
        else:
            loss_digit = torch.tensor(0.0, device=emb.device)

        # 10. Decorrelation loss (prevents dimension collapse)
        if lam_decorr > 0:
            emb_c = emb - emb.mean(dim=0, keepdim=True)
            cov = (emb_c.T @ emb_c) / (n - 1)
            std = torch.sqrt(torch.diag(cov).clamp(min=1e-8))
            corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
            d = corr.shape[0]
            off_diag = corr.pow(2).sum() - d
            loss_decorr = off_diag / (d * (d - 1))
        else:
            loss_decorr = torch.tensor(0.0, device=emb.device)

        # 11. Subtraction probe
        if lam_sub > 0:
            half = n // 2
            x1, x2 = x[:half], x[half:2 * half]
            emb1, emb2 = emb[:half], emb[half:2 * half]
            pred_diff = self.subtraction_probe(emb1, emb2)
            target_diff = x1 - x2
            f_pred_d = torch.sign(pred_diff) * torch.log1p(
                torch.abs(pred_diff))
            f_target_d = torch.sign(target_diff) * torch.log1p(
                torch.abs(target_diff))
            loss_sub = F.mse_loss(f_pred_d, f_target_d)
        else:
            loss_sub = torch.tensor(0.0, device=emb.device)

        return (loss_slog
                + 0.1 * loss_bce
                + 0.3 * loss_lm
                + lam_rel * loss_rel
                + 0.05 * loss_spread
                + lam_compose * loss_compose
                + lam_order * loss_order
                + lam_magnitude * loss_mag
                + lam_digit * loss_digit
                + lam_decorr * loss_decorr
                + lam_sub * loss_sub)

    def train_model(self, num_steps: int = 2000000, batch_size: int = 512,
                    lr: float = 5e-4, log_interval: int = 10000,
                    warmup_steps: int = 5000, grad_clip: float = 1.0):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                       betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=1e-5)
        losses = []

        # Phase schedule
        # Phase 1 (0-10%): core losses only
        # Phase 2 (10-30%): ramp in multi-objective + decorrelation
        # Phase 3 (30-50%): ramp in digit loss
        # Phase 4 (50-60%): ramp in relative MSE
        # Phase 5 (60-100%): all losses at full weight

        obj_start = int(num_steps * 0.10)
        obj_end = int(num_steps * 0.30)
        digit_start = int(num_steps * 0.20)
        digit_end = int(num_steps * 0.40)
        rel_start = int(num_steps * 0.50)
        rel_end = int(num_steps * 0.60)

        self.train()
        for step in range(1, num_steps + 1):
            # Relative MSE ramp
            if step < rel_start:
                lam_rel = 0.0
            elif step < rel_end:
                lam_rel = 0.3 * (step - rel_start) / (rel_end - rel_start)
            else:
                lam_rel = 0.3

            # Multi-objective ramp (compose, order, magnitude, sub, decorr)
            if step < obj_start:
                obj_frac = 0.0
            elif step < obj_end:
                obj_frac = (step - obj_start) / (obj_end - obj_start)
            else:
                obj_frac = 1.0

            lam_compose = 0.3 * obj_frac
            lam_order = 0.1 * obj_frac
            lam_magnitude = 0.1 * obj_frac
            lam_sub = 0.2 * obj_frac
            lam_decorr = 0.1 * obj_frac

            # Digit loss ramp (later start)
            if step < digit_start:
                lam_digit = 0.0
            elif step < digit_end:
                lam_digit = 1.0 * (step - digit_start) / (
                    digit_end - digit_start)
            else:
                lam_digit = 1.0

            x = sample_training_numbers(batch_size, self.device)

            optimizer.zero_grad()
            emb, recon, sign_logit = self.forward(x)
            loss = self.compute_loss(x, emb, recon, sign_logit,
                                     lam_rel, lam_compose, lam_order,
                                     lam_magnitude, lam_digit,
                                     lam_decorr, lam_sub)
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
                w = self.encoder.scale_lane.weight
                w_min, w_max = w.min().item(), w.max().item()

                # Phase tag
                if step < obj_start:
                    phase = "P1:core"
                elif step < digit_start:
                    phase = "P2:obj"
                elif step < rel_start:
                    phase = "P3:digit"
                elif step < rel_end:
                    phase = "P4:rel"
                else:
                    phase = "P5:full"

                # Lane norms for a sample
                with torch.no_grad():
                    sample_x = torch.tensor([1.0, 1000.0, 1e6, 1e9],
                                            device=self.device)
                    sample_emb = self.encode(sample_x)
                    k = self.scale_dims
                    r = self.residue_dims
                    s_norms = sample_emb[:, :k].norm(dim=-1)
                    r_norms = sample_emb[:, k:k+r].norm(dim=-1)
                    e_norms = sample_emb[:, k+r:].norm(dim=-1)

                print(f"  Step {step:>7d}/{num_steps} | Loss: {avg:.6f} "
                      f"[{phase}] | lr: {lr_t:.2e} "
                      f"| scale_w: [{w_min:.2e}, {w_max:.2e}]")
                print(f"    Lane norms (1, 1K, 1M, 1B): "
                      f"scale=[{s_norms[0]:.2f},{s_norms[1]:.2f},"
                      f"{s_norms[2]:.2f},{s_norms[3]:.2f}] "
                      f"residue=[{r_norms[0]:.2f},{r_norms[1]:.2f},"
                      f"{r_norms[2]:.2f},{r_norms[3]:.2f}] "
                      f"semantic=[{e_norms[0]:.2f},{e_norms[1]:.2f},"
                      f"{e_norms[2]:.2f},{e_norms[3]:.2f}]")

        self.eval()
        return losses


# =============================================================================
# Helpers
# =============================================================================

def _encode_np(system: NumberEmbeddingSystem, vals) -> np.ndarray:
    with torch.no_grad():
        t = torch.tensor(np.atleast_1d(np.asarray(vals, dtype=np.float64)),
                         dtype=torch.float32, device=system.device)
        return system.encode(t).cpu().numpy()


def _forward_np(system: NumberEmbeddingSystem, vals):
    with torch.no_grad():
        t = torch.tensor(np.atleast_1d(np.asarray(vals, dtype=np.float64)),
                         dtype=torch.float32, device=system.device)
        emb, recon, _ = system.forward(t)
        return emb.cpu().numpy(), recon.cpu().numpy()


# =============================================================================
# Probe Tests (expanded for v10)
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

    for label, lo, hi in [("[-1K,1K]", -1000, 1000),
                           ("[-1M,1M]", -1000000, 1000000)]:
        x1_tr = torch.empty(n_train, device=device).uniform_(lo, hi)
        x2_tr = torch.empty(n_train, device=device).uniform_(lo, hi)
        x1_te = torch.empty(n_test, device=device).uniform_(lo, hi)
        x2_te = torch.empty(n_test, device=device).uniform_(lo, hi)

        with torch.no_grad():
            e1_tr = system.encode(x1_tr)
            e2_tr = system.encode(x2_tr)
            e1_te = system.encode(x1_te)
            e2_te = system.encode(x2_te)

        X_tr = torch.cat([e1_tr, e2_tr], dim=-1)
        y_tr = (x1_tr + x2_tr).float()
        X_te = torch.cat([e1_te, e2_te], dim=-1)
        y_te = (x1_te + x2_te).float()

        XtX = X_tr.T @ X_tr + 1e-4 * torch.eye(X_tr.shape[1], device=device)
        w = torch.linalg.solve(XtX, X_tr.T @ y_tr)
        pred = X_te @ w
        ss_res = ((y_te - pred) ** 2).sum().item()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        mae = (y_te - pred).abs().mean().item()
        print(f"    {label}: R² = {r2:.6f}  |  MAE = {mae:.2f}")
        results[f'addition_r2_{label}'] = r2

    # ── Probe 2: Linear Subtraction ──
    print("\n  Probe 2: LINEAR SUBTRACTION — W @ [e(x); e(y)] → x-y")
    print("  " + "-" * 60)

    x1_tr = torch.empty(n_train, device=device).uniform_(-1000, 1000)
    x2_tr = torch.empty(n_train, device=device).uniform_(-1000, 1000)
    x1_te = torch.empty(n_test, device=device).uniform_(-1000, 1000)
    x2_te = torch.empty(n_test, device=device).uniform_(-1000, 1000)

    with torch.no_grad():
        e1_tr = system.encode(x1_tr)
        e2_tr = system.encode(x2_tr)
        e1_te = system.encode(x1_te)
        e2_te = system.encode(x2_te)

    X_tr = torch.cat([e1_tr, e2_tr], dim=-1)
    y_tr = (x1_tr - x2_tr).float()
    X_te = torch.cat([e1_te, e2_te], dim=-1)
    y_te = (x1_te - x2_te).float()

    XtX = X_tr.T @ X_tr + 1e-4 * torch.eye(X_tr.shape[1], device=device)
    w_sub = torch.linalg.solve(XtX, X_tr.T @ y_tr)
    pred_sub = X_te @ w_sub
    ss_res = ((y_te - pred_sub) ** 2).sum().item()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
    r2_sub = 1.0 - ss_res / (ss_tot + 1e-8)
    mae_sub = (y_te - pred_sub).abs().mean().item()
    print(f"    R² = {r2_sub:.6f}  |  MAE = {mae_sub:.2f}")
    results['subtraction_r2'] = r2_sub

    # ── Probe 3: Linear Order ──
    print("\n  Probe 3: LINEAR ORDER — w @ e(x) ranks numbers correctly")
    print("  " + "-" * 60)

    x_order = torch.linspace(-1e6, 1e6, 2000, device=device)
    with torch.no_grad():
        emb_order = system.encode(x_order)

    XtX_o = emb_order.T @ emb_order + 1e-4 * torch.eye(
        system.embedding_dim, device=device)
    Xty_o = emb_order.T @ x_order
    w_ord = torch.linalg.solve(XtX_o, Xty_o)
    pred_order = emb_order @ w_ord

    rho, _ = spearmanr(x_order.cpu().numpy(), pred_order.cpu().numpy())
    print(f"    Spearman ρ = {rho:.6f}")
    results['order_spearman'] = rho

    # ── Probe 4: Magnitude Classification ──
    print("\n  Probe 4: MAGNITUDE — predict exponent bucket from embedding")
    print("  " + "-" * 60)

    x_mag = sample_training_numbers(5000, device)
    with torch.no_grad():
        emb_mag = system.encode(x_mag)
    labels = MagnitudeProbe.label(x_mag)

    n_tr = 4000
    X_tr, X_te = emb_mag[:n_tr], emb_mag[n_tr:]
    y_tr, y_te = labels[:n_tr], labels[n_tr:]

    n_cls = MagnitudeProbe.NUM_CLASSES
    Y_oh = F.one_hot(y_tr, n_cls).float()
    XtX_m = X_tr.T @ X_tr + 1e-3 * torch.eye(system.embedding_dim,
                                                device=device)
    W_mag = torch.linalg.solve(XtX_m, X_tr.T @ Y_oh)
    pred_mag = (X_te @ W_mag).argmax(dim=-1)
    acc_mag = (pred_mag == y_te).float().mean().item()
    print(f"    Accuracy = {acc_mag:.4f} ({n_cls} classes)")
    results['magnitude_acc'] = acc_mag

    # ── Probe 5: Parity ──
    print("\n  Probe 5: PARITY — predict x mod 2 from embedding")
    print("  " + "-" * 60)

    x_par = torch.randint(-5000, 5000, (5000,), device=device,
                           dtype=torch.float32)
    with torch.no_grad():
        emb_par = system.encode(x_par)
    y_par = (x_par.long().abs() % 2).float()

    X_tr_p, X_te_p = emb_par[:4000], emb_par[4000:]
    y_tr_p, y_te_p = y_par[:4000], y_par[4000:]

    XtX_p = X_tr_p.T @ X_tr_p + 1e-3 * torch.eye(
        system.embedding_dim, device=device)
    w_par = torch.linalg.solve(XtX_p, X_tr_p.T @ y_tr_p)
    pred_par = (X_te_p @ w_par > 0.5).float()
    acc_par = (pred_par == y_te_p).float().mean().item()
    print(f"    Accuracy = {acc_par:.4f}")
    results['parity_acc'] = acc_par

    # ── Probe 6: Last Digit ──
    print("\n  Probe 6: LAST DIGIT — predict |x| mod 10 from embedding")
    print("  " + "-" * 60)

    x_dig = torch.randint(-10000, 10000, (5000,), device=device,
                           dtype=torch.float32)
    with torch.no_grad():
        emb_dig = system.encode(x_dig)
    y_dig = (x_dig.long().abs() % 10)

    X_tr_d, X_te_d = emb_dig[:4000], emb_dig[4000:]
    y_tr_d, y_te_d = y_dig[:4000], y_dig[4000:]

    Y_oh_d = F.one_hot(y_tr_d, 10).float()
    XtX_d = X_tr_d.T @ X_tr_d + 1e-3 * torch.eye(
        system.embedding_dim, device=device)
    W_dig = torch.linalg.solve(XtX_d, X_tr_d.T @ Y_oh_d)
    pred_dig = (X_te_d @ W_dig).argmax(dim=-1)
    acc_dig = (pred_dig == y_te_d).float().mean().item()
    print(f"    Accuracy = {acc_dig:.4f} (10 classes, chance=0.10)")
    results['digit_acc'] = acc_dig

    # ── Probe 7: All Digits Per-Position ──
    print("\n  Probe 7: ALL DIGITS — per-position accuracy from residue lane")
    print("  " + "-" * 60)

    x_all = torch.randint(0, 1000000000, (8000,), device=device,
                           dtype=torch.float32)
    with torch.no_grad():
        emb_all = system.encode(x_all)
    k = system.scale_dims
    r = system.residue_dims
    residue_all = emb_all[:, k:k + r]

    digit_labels = DigitProbe.labels(x_all, 10)
    X_tr_a, X_te_a = residue_all[:6000], residue_all[6000:]

    per_pos_acc = []
    for pos in range(10):
        y_tr_a = digit_labels[pos][:6000]
        y_te_a = digit_labels[pos][6000:]
        Y_oh_a = F.one_hot(y_tr_a, 10).float()
        XtX_a = X_tr_a.T @ X_tr_a + 1e-3 * torch.eye(r, device=device)
        W_a = torch.linalg.solve(XtX_a, X_tr_a.T @ Y_oh_a)
        pred_a = (X_te_a @ W_a).argmax(dim=-1)
        acc_a = (pred_a == y_te_a).float().mean().item()
        per_pos_acc.append(acc_a)
        digit_name = ['ones', 'tens', 'hundreds', '1K', '10K', '100K',
                       '1M', '10M', '100M', '1B'][pos]
        print(f"    Position {pos} ({digit_name:>6}): {acc_a:.4f}")
    results['digit_per_pos'] = per_pos_acc

    # ── Probe 8: Additivity ──
    print("\n  Probe 8: ADDITIVITY — ||e(x+y) - (e(x)+e(y))|| / ||e(x+y)||")
    print("  " + "-" * 60)

    K = system.scale_dims
    ranges = [
        ("small+small",     -10,      10,      -10,      10),
        ("medium+medium",   -1000,    1000,    -1000,    1000),
        ("large+large",     -100000,  100000,  -100000,  100000),
        ("very large",      -1e8,     1e8,     -1e8,     1e8),
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
        print(f"    {name:>20}:  full={rel_full:.4f}  "
              f"scale_lane={rel_scale:.6f}")

    results['additivity_full'] = sum(all_full) / len(all_full)
    results['additivity_scale'] = sum(all_scale) / len(all_scale)
    print(f"\n    Overall:  full={results['additivity_full']:.4f}  "
          f"scale_lane={results['additivity_scale']:.6f}")

    # ── Probe 9: MIN/MAX ──
    print("\n  Probe 9: MIN/MAX — predict min(x,y) and max(x,y) from "
          "[e(x); e(y)]")
    print("  " + "-" * 60)

    x1_mm = torch.empty(n_train, device=device).uniform_(-10000, 10000)
    x2_mm = torch.empty(n_train, device=device).uniform_(-10000, 10000)
    x1_mm_te = torch.empty(n_test, device=device).uniform_(-10000, 10000)
    x2_mm_te = torch.empty(n_test, device=device).uniform_(-10000, 10000)

    with torch.no_grad():
        e1_mm = system.encode(x1_mm)
        e2_mm = system.encode(x2_mm)
        e1_mm_te = system.encode(x1_mm_te)
        e2_mm_te = system.encode(x2_mm_te)

    X_mm_tr = torch.cat([e1_mm, e2_mm], dim=-1)
    X_mm_te = torch.cat([e1_mm_te, e2_mm_te], dim=-1)

    for op_name, op_fn in [("MIN", torch.minimum), ("MAX", torch.maximum)]:
        y_tr_mm = op_fn(x1_mm, x2_mm).float()
        y_te_mm = op_fn(x1_mm_te, x2_mm_te).float()
        XtX_mm = X_mm_tr.T @ X_mm_tr + 1e-4 * torch.eye(
            X_mm_tr.shape[1], device=device)
        w_mm = torch.linalg.solve(XtX_mm, X_mm_tr.T @ y_tr_mm)
        pred_mm = X_mm_te @ w_mm
        ss_res = ((y_te_mm - pred_mm) ** 2).sum().item()
        ss_tot = ((y_te_mm - y_te_mm.mean()) ** 2).sum().item()
        r2_mm = 1.0 - ss_res / (ss_tot + 1e-8)
        print(f"    {op_name}: R² = {r2_mm:.6f}")
        results[f'{op_name.lower()}_r2'] = r2_mm

    # ── Probe 10: Cross-Scale Generalization ──
    print("\n  Probe 10: CROSS-SCALE — train on [0,1K], test on [1M,1B]")
    print("  " + "-" * 60)

    x1_cs_tr = torch.empty(n_train, device=device).uniform_(-1000, 1000)
    x2_cs_tr = torch.empty(n_train, device=device).uniform_(-1000, 1000)
    # Test on MUCH larger numbers
    x1_cs_te = torch.empty(n_test, device=device).uniform_(-1e9, 1e9)
    x2_cs_te = torch.empty(n_test, device=device).uniform_(-1e9, 1e9)

    with torch.no_grad():
        e1_cs_tr = system.encode(x1_cs_tr)
        e2_cs_tr = system.encode(x2_cs_tr)
        e1_cs_te = system.encode(x1_cs_te)
        e2_cs_te = system.encode(x2_cs_te)

    X_cs_tr = torch.cat([e1_cs_tr, e2_cs_tr], dim=-1)
    X_cs_te = torch.cat([e1_cs_te, e2_cs_te], dim=-1)
    y_cs_tr = (x1_cs_tr + x2_cs_tr).float()
    y_cs_te = (x1_cs_te + x2_cs_te).float()

    XtX_cs = X_cs_tr.T @ X_cs_tr + 1e-4 * torch.eye(
        X_cs_tr.shape[1], device=device)
    w_cs = torch.linalg.solve(XtX_cs, X_cs_tr.T @ y_cs_tr)
    pred_cs = X_cs_te @ w_cs
    ss_res_cs = ((y_cs_te - pred_cs) ** 2).sum().item()
    ss_tot_cs = ((y_cs_te - y_cs_te.mean()) ** 2).sum().item()
    r2_cs = 1.0 - ss_res_cs / (ss_tot_cs + 1e-8)
    print(f"    Addition R² (cross-scale) = {r2_cs:.6f}")
    results['cross_scale_r2'] = r2_cs

    # ── Summary ──
    print("\n" + "=" * 70)
    print("PROBE SUMMARY")
    print("=" * 70)
    print(f"  Addition [-1K,1K]:     R² = "
          f"{results.get('addition_r2_[-1K,1K]', 0):.4f}")
    print(f"  Addition [-1M,1M]:     R² = "
          f"{results.get('addition_r2_[-1M,1M]', 0):.4f}")
    print(f"  Subtraction:           R² = {r2_sub:.4f}")
    print(f"  Order (Spearman):      ρ  = {rho:.4f}")
    print(f"  Magnitude ({n_cls}-cls):    acc = {acc_mag:.4f}")
    print(f"  Parity (binary):       acc = {acc_par:.4f}")
    print(f"  Last digit (10-cls):   acc = {acc_dig:.4f}")
    digit_avg = sum(per_pos_acc) / len(per_pos_acc)
    print(f"  Digit avg (residue):   acc = {digit_avg:.4f}")
    print(f"  Additivity (scale):    err = {results['additivity_scale']:.6f}")
    print(f"  Additivity (full):     err = {results['additivity_full']:.4f}")
    print(f"  MIN R²:                     {results['min_r2']:.4f}")
    print(f"  MAX R²:                     {results['max_r2']:.4f}")
    print(f"  Cross-scale R²:             {r2_cs:.4f}")
    print("=" * 70)

    return results


# =============================================================================
# Standard Tests (extended for v10)
# =============================================================================

def run_standard_tests(system: NumberEmbeddingSystem):
    """Extended tests: uniqueness, continuity, reversibility, expressiveness,
    compatibility, lane structure, float32 stability, discriminability,
    lane independence."""
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

    nums = np.array([0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, -1.0,
                     -100.0, 3.14159, 1e6, 1e9, -1e9])
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
    check("1.0000 vs 1.0001 distinguishable", d > 1e-5,
          f"L2 dist: {d:.8f}")

    # === TEST 2: CONTINUITY ===
    print("-" * 70)
    print("TEST 2: CONTINUITY — Small perturbations → small embedding changes")
    print("-" * 70)

    base_vals = np.array([0.0, 1.0, 10.0, -5.0, 100.0, 1e6, 1e9])
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

    # === TEST 3: REVERSIBILITY (extended to 1e9) ===
    print("-" * 70)
    print("TEST 3: REVERSIBILITY — Reconstruction accuracy (extended range)")
    print("-" * 70)

    groups = {
        "Small positive": np.array([0.01, 0.05, 0.1]),
        "Medium":         np.array([1.0, 2.5, 3.14159, 7.0]),
        "Large":          np.array([50.0, 100.0, 200.0]),
        "Very large":     np.array([1e5, 1e6, 1e7]),
        "Huge":           np.array([1e8, 5e8, 1e9]),
        "Negative":       np.array([-1.0, -10.0, -50.0, -1e6]),
        "Near zero":      np.array([-0.01, 0.0, 0.01]),
        "Integers":       np.array([1.0, 2.0, 42.0, 99.0, 999999.0]),
    }
    for name, vals in groups.items():
        _, recon = _forward_np(system, vals)
        abs_err = np.abs(vals - recon)
        rel_err = abs_err / (np.abs(vals) + 1e-8)
        check(f"Reconstruction [{name}]",
              rel_err.max() < 1.0 or abs_err.max() < 100.0,
              f"|err|_max: {abs_err.max():.2f}, rel_max: {rel_err.max():.4f}")

    # Showcase table
    showcase = np.array([
        0.0, 1e-4, 0.01, 0.1, 1.0, 3.14159, 10.0, 42.0, 100.0,
        1000.0, 9999.0, 1e5, 1e6, 1e7, 1e8, 1e9,
        -0.1, -1.0, -42.0, -1000.0, -1e6, -1e9,
    ])
    _, recon_s = _forward_np(system, showcase)
    print(f"\n  {'Input':>14}  →  {'Decoded':>16}  {'Abs Err':>12}  "
          f"{'Rel Err':>10}")
    print(f"  {'─' * 14}     {'─' * 16}  {'─' * 12}  {'─' * 10}")
    for o, r in zip(showcase, recon_s):
        e = abs(o - r)
        print(f"  {o:>14.2f}  →  {r:>16.2f}  {e:>12.2f}  "
              f"{e / (abs(o) + 1e-8):>9.4%}")
    print()

    # === TEST 4: EXPRESSIVENESS (effective dimensionality) ===
    print("-" * 70)
    print("TEST 4: EXPRESSIVENESS — Effective dimensionality & structure")
    print("-" * 70)

    task_range_nums = np.concatenate([
        np.random.uniform(-1e9, 1e9, 400),
        np.random.uniform(-1000, 1000, 100),
    ]).astype(np.float32)
    embs_s = _encode_np(system, task_range_nums)
    _, S, _ = np.linalg.svd(embs_s - embs_s.mean(axis=0), full_matrices=False)
    eff = int(np.sum(S > 0.01 * S[0]))
    check("Effective dimensionality > 40", eff > 40,
          f"Effective dims: {eff}/{system.embedding_dim}")

    ordered = np.linspace(-1e6, 1e6, 50)
    embs_o = _encode_np(system, ordered)
    cent = embs_o - embs_o.mean(axis=0)
    _, _, Vt = np.linalg.svd(cent, full_matrices=False)
    pc1 = cent @ Vt[0]
    corr, _ = spearmanr(ordered, pc1)
    check("PC1 correlates with ordering", abs(corr) > 0.7,
          f"Spearman ρ = {corr:.4f}")

    # === TEST 5: MODEL COMPATIBILITY ===
    print("-" * 70)
    print("TEST 5: MODEL COMPATIBILITY — Standard ops & integration")
    print("-" * 70)

    emb64 = _encode_np(system, np.random.randn(64) * 1e6)
    check("Batch (64,) → (64, d)", emb64.shape == (64, system.embedding_dim))
    check("Single number works",
          _encode_np(system, [42.0]).shape == (1, system.embedding_dim))

    out = emb64 @ np.random.randn(system.embedding_dim, 10) * 0.01
    check("Downstream linear layer", out.shape == (64, 10))

    seq = _encode_np(system, np.random.randn(8) * 1000)
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
    print("TEST 6: LANE STRUCTURE — Scale=linear, Residue=periodic")
    print("-" * 70)

    K = system.scale_dims
    R = system.residue_dims

    # Scale lane linearity
    test_vals = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    test_embs = _encode_np(system, test_vals)
    add_part = test_embs[:, :K]
    ratios = add_part / test_vals[:, None]
    max_ratio_std = ratios.std(axis=0).max()
    check("Scale lane dims scale linearly with x", max_ratio_std < 0.01,
          f"Max ratio std: {max_ratio_std:.6f}")

    # Residue lane periodicity: period-10 features repeat every 10
    n_periods = len(system.encoder.residue_lane.periods)
    x_base = torch.tensor([42.0, 137.0, 9999.0, 1e8], device=system.device)
    x_plus10 = x_base + 10.0
    with torch.no_grad():
        e_base_full = system.encode(x_base)
        e_p10_full = system.encode(x_plus10)
        # Period 10 is at index 2 (after 2, 5), sin at K+2, cos at K+n_periods+2
        idx_sin = K + 2
        idx_cos = K + n_periods + 2
        e_base_r = torch.stack([e_base_full[:, idx_sin],
                                e_base_full[:, idx_cos]], dim=-1)
        e_p10_r = torch.stack([e_p10_full[:, idx_sin],
                               e_p10_full[:, idx_cos]], dim=-1)
    period10_err = (e_base_r - e_p10_r).abs().max().item()
    check("Residue period-10 repeats every 10", period10_err < 0.01,
          f"Max diff: {period10_err:.6f}")

    # Parity: period-2 features differ for odd vs even
    x_even = torch.tensor([2.0, 100.0, 1000.0], device=system.device)
    x_odd = torch.tensor([3.0, 101.0, 1001.0], device=system.device)
    with torch.no_grad():
        e_even = system.encode(x_even)
        e_odd = system.encode(x_odd)
        # Period 2 is at index 0, sin at K+0, cos at K+n_periods+0
        even_sin = e_even[:, K + 0]
        odd_sin = e_odd[:, K + 0]
    parity_diff = (even_sin - odd_sin).abs().mean().item()
    check("Period-2 distinguishes even/odd", parity_diff > 0.5,
          f"Mean |sin diff|: {parity_diff:.4f}")

    w = system.encoder.scale_lane.weight.detach().cpu().numpy()
    print(f"  Scale weight range: [{w.min():.2e}, {w.max():.2e}]")

    # === TEST 7: FLOAT32 STABILITY ===
    print("-" * 70)
    print("TEST 7: FLOAT32 STABILITY — No NaN/Inf, distinct outputs at 1e9")
    print("-" * 70)

    extreme_vals = np.array([0.0, 1e-8, 1.0, 1e3, 1e6, 1e9, -1e9,
                             999999999.0, 1000000001.0])
    embs_ext = _encode_np(system, extreme_vals)
    check("No NaN in embeddings", not np.any(np.isnan(embs_ext)))
    check("No Inf in embeddings", not np.any(np.isinf(embs_ext)))

    # Distinct outputs for close large numbers
    a_big = _encode_np(system, [999999998.0])
    b_big = _encode_np(system, [999999999.0])
    d_big = np.linalg.norm(a_big - b_big)
    check("999999998 vs 999999999 distinguishable", d_big > 1e-4,
          f"L2 dist: {d_big:.6f}")

    # Residue features are numerically meaningful at 1e9
    with torch.no_grad():
        e_1b = system.encode(torch.tensor([1000000000.0],
                                           device=system.device))
        e_1b_p1 = system.encode(torch.tensor([1000000001.0],
                                              device=system.device))
        residue_1b = e_1b[:, K:K + R]
        residue_1b_p1 = e_1b_p1[:, K:K + R]
    residue_diff = (residue_1b - residue_1b_p1).abs().max().item()
    check("Residue features change at 1e9 boundary", residue_diff > 0.01,
          f"Max residue diff: {residue_diff:.4f}")

    # === TEST 8: DISCRIMINABILITY ===
    print("-" * 70)
    print("TEST 8: DISCRIMINABILITY — Nearest-neighbor accuracy")
    print("-" * 70)

    # 500 random numbers, check if embedding nearest neighbor matches
    # number nearest neighbor
    disc_nums = np.sort(np.random.uniform(-1e6, 1e6, 500)).astype(np.float32)
    disc_embs = _encode_np(system, disc_nums)

    # Number nearest neighbor: adjacent in sorted order
    n_correct = 0
    n_total = len(disc_nums) - 2
    for i in range(1, len(disc_nums) - 1):
        dists = np.linalg.norm(disc_embs - disc_embs[i], axis=-1)
        dists[i] = np.inf
        nn_idx = dists.argmin()
        # True nearest is i-1 or i+1 (sorted)
        if nn_idx in (i - 1, i + 1):
            n_correct += 1
    nn_acc = n_correct / n_total
    check("Nearest-neighbor accuracy > 0.80", nn_acc > 0.80,
          f"NN accuracy: {nn_acc:.4f}")
    results_disc = {'nn_acc': nn_acc}

    # === TEST 9: LANE INDEPENDENCE ===
    print("-" * 70)
    print("TEST 9: LANE INDEPENDENCE — Low correlation between lanes")
    print("-" * 70)

    ind_nums = np.random.uniform(-1e6, 1e6, 1000).astype(np.float32)
    ind_embs = _encode_np(system, ind_nums)
    scale_part = ind_embs[:, :K]
    residue_part = ind_embs[:, K:K + R]
    semantic_part = ind_embs[:, K + R:]

    # Mean absolute correlation between lane centroids
    def lane_corr(a, b):
        a_c = a - a.mean(axis=0)
        b_c = b - b.mean(axis=0)
        a_std = a_c.std(axis=0).mean()
        b_std = b_c.std(axis=0).mean()
        cov = np.abs(a_c.T @ b_c).mean() / len(a)
        return cov / (a_std * b_std + 1e-8)

    sr_corr = lane_corr(scale_part, residue_part)
    ss_corr = lane_corr(scale_part, semantic_part)
    rs_corr = lane_corr(residue_part, semantic_part)
    check("Scale-Residue correlation < 0.3", sr_corr < 0.3,
          f"Corr: {sr_corr:.4f}")
    check("Scale-Semantic correlation < 0.3", ss_corr < 0.3,
          f"Corr: {ss_corr:.4f}")
    check("Residue-Semantic correlation < 0.3", rs_corr < 0.3,
          f"Corr: {rs_corr:.4f}")

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
    tag = f"np_emb_v10_{num_steps // 1000}k"
    model_path = os.path.join(checkpoint_dir, f"{tag}_model.pt")

    torch.save({
        'encoder_state_dict': system.encoder.state_dict(),
        'full_state_dict': system.state_dict(),
        'embedding_dim': system.embedding_dim,
        'scale_dims': system.scale_dims,
        'residue_periods': system.encoder.residue_lane.periods.cpu().tolist(),
        'residue_dims': system.residue_dims,
        'num_frequencies': system.encoder.fourier.frequencies.shape[0],
        'num_steps': num_steps,
        'variant': 'v10_high_fidelity_1B',
    }, model_path)
    print(f"  Model saved: {model_path}")


# =============================================================================
# Demo
# =============================================================================

def demo(num_steps: int = 2000000, scale_dims: int = 16,
         residue_periods: List[int] = None, device: torch.device = None):
    if device is None:
        device = get_device()
    print("=" * 70)
    print("NUMBER EMBEDDING v10 — HIGH-FIDELITY MATH-AWARE ENCODER (1B)")
    print("=" * 70)
    print()

    system = NumberEmbeddingSystem(embedding_dim=128, scale_dims=scale_dims,
                                   residue_periods=residue_periods,
                                   device=device)
    enc = system.encoder

    total_params = sum(p.numel() for p in system.parameters())
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in system.decoder.parameters())
    probe_params = total_params - enc_params - dec_params

    K = scale_dims
    R = system.residue_dims

    print(f"Architecture:  {enc.raw_dim} raw → {system.embedding_dim}d "
          f"embedding")
    print(f"  Lane 1 (Scale):    {K} dims — x * weight, additive")
    print(f"  Lane 2 (Residue):  {R} dims — sin/cos at "
          f"{len(enc.residue_lane.periods)} periods (float64), "
          f"periods={[int(p) for p in enc.residue_lane.periods.cpu().tolist()]}")
    print(f"  Lane 3 (Semantic): {enc.semantic_dims} dims — "
          f"Fourier({enc.fourier.output_dim}d, log-space) + LogMag + Sign "
          f"→ proj → scale")
    print(f"  Encoder params:    {enc_params:,}")
    print(f"  Decoder params:    {dec_params:,}")
    print(f"  Probe params:      {probe_params:,} "
          f"(discarded after pretraining)")
    print(f"  Total params:      {total_params:,}")
    print(f"  Device:            {device}")
    print()

    numbers = [0.0, 1e-4, 0.1, 1.0, 3.14159, -42.0, 100.0, 1000.0,
               1e5, 1e6, 1e8, 1e9, -1234.5, -1e9]
    print("Before training:")
    for n in numbers:
        emb = _encode_np(system, [n])[0]
        s_norm = np.linalg.norm(emb[:K])
        r_norm = np.linalg.norm(emb[K:K + R])
        e_norm = np.linalg.norm(emb[K + R:])
        print(f"  {n:>12.1f} → scale={s_norm:9.3f}  residue={r_norm:5.3f}  "
              f"semantic={e_norm:7.3f}")
    print()

    print(f"Training ({num_steps:,} steps, batch 512)...")
    t0 = time.time()
    system.train_model(num_steps=num_steps, batch_size=512, lr=5e-4,
                       log_interval=max(1, num_steps // 100))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/3600:.1f}h)\n")

    print("After training:")
    print(f"  {'Input':>14}  →  {'Decoded':>16}  {'Error':>12}")
    print(f"  {'─' * 14}     {'─' * 16}  {'─' * 12}")
    for n in numbers:
        _, recon = _forward_np(system, [n])
        print(f"  {n:>14.1f}  →  {recon[0]:>16.2f}  "
              f"{abs(n - recon[0]):>12.2f}")
    print()

    print("Lane norms after training:")
    for n in numbers:
        emb = _encode_np(system, [n])[0]
        s_norm = np.linalg.norm(emb[:K])
        r_norm = np.linalg.norm(emb[K:K + R])
        e_norm = np.linalg.norm(emb[K + R:])
        print(f"  {n:>12.1f} → scale={s_norm:9.3f}  residue={r_norm:5.3f}  "
              f"semantic={e_norm:7.3f}")
    print()

    return system


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Number Embedding v10 — High-Fidelity (1B Range)")
    parser.add_argument("--max-steps", type=int, default=2000000,
                        help="Training steps (default: 2000000)")
    parser.add_argument("--scale-dims", type=int, default=16,
                        help="Scale lane dimensions (default: 16)")
    parser.add_argument("--residue-periods", type=str,
                        default="2,5,10,100,1000,10000,100000,"
                                "1000000,10000000,100000000,1000000000",
                        help="Comma-separated residue periods")
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
                            f"run_v10_{time.strftime('%Y%m%d_%H%M%S')}.log")

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
    print(f"Seed: {args.seed}  |  Max steps: {args.max_steps:,}  |  "
          f"Device: {device}")
    print(f"Scale dims: {args.scale_dims}  |  Residue periods: "
          f"{residue_periods}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    try:
        if args.load:
            ckpt = torch.load(args.load, map_location=device,
                              weights_only=False)
            scale_dims = ckpt.get('scale_dims', args.scale_dims)
            rp = ckpt.get('residue_periods', residue_periods)
            rp = [int(p) for p in rp]
            nf = ckpt.get('num_frequencies', 33)
            print(f"Loading checkpoint: {args.load}")
            print(f"  scale_dims={scale_dims}, residue_periods={rp}, "
                  f"num_freq={nf}")
            print(f"  trained for {ckpt.get('num_steps', '?'):,} steps")
            print()

            system = NumberEmbeddingSystem(
                embedding_dim=ckpt.get('embedding_dim', 128),
                scale_dims=scale_dims, residue_periods=rp,
                num_frequencies=nf, device=device)
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
            system = demo(num_steps=args.max_steps,
                          scale_dims=args.scale_dims,
                          residue_periods=residue_periods, device=device)
            save_checkpoint(system, args.max_steps)
        else:
            system = demo(num_steps=args.max_steps,
                          scale_dims=args.scale_dims,
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
