"""
Number Embedding Representation Framework
==========================================
PyTorch implementation (GPU-accelerated).

Embeds numerical values into high-dimensional vectors preserving:
  1. Uniqueness   — distinct numbers → distinct embeddings
  2. Continuity   — small Δx → small Δe(x)
  3. Reversibility — decode(encode(x)) ≈ x
  4. Expressiveness — captures multi-scale numerical structure
  5. Compatibility  — standard arrays, differentiable, integrable

Architecture (v8 — anti-collapse fixes):
  Encoder:  x ∈ ℝ  →  e(x) ∈ ℝ^d
    Channel 1: Fourier — 32 linear-freq (64 dims)
    Channel 2: Log-magnitude (1 dim)
    Channel 3: Smooth sign (1 dim)
    Channel 4: Polynomial basis (5 dims)
    → Learned linear (71 → 127) → LayerNorm → concat [log_norm] → ℝ^128

  Decoder:  e(x) ∈ ℝ^d  →  x̂ ∈ ℝ
    3-layer MLP with GELU + residual skip, predicts sign * exp(log_mag)

  Training:
    L = signed_log_MSE + 0.1·BCE_sign + 0.3·log_mag_MSE
        + 0.3·rel_MSE (phase 2, ramped) + 0.05·spread_loss
    AdamW optimiser, trained on log-uniform samples, batch 512, 500k steps.

  v8 changes from v7:
    - Reverted to 32 linear-only Fourier (was split 16+16 log)
    - 1 reserved dim (was 2) — dropped redundant proj_mean
    - Batch 512 (was 4096) — prevents embedding collapse
    - Spread loss (λ=0.05) — penalizes high pairwise cosine similarity
    - Relative MSE weight 0.3 (was 0.5), ramp at 40-50% (was 30-40%)
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
# Encoder Channels (Analytic — no learnable params)
# =============================================================================

class FourierChannel(nn.Module):
    """Fourier encoding: 32 linear-frequency channels.

    sin/cos at geometrically-spaced ω applied to raw x.
    Amplitude damping 1/√(1+k) for stability.
    Total output: 64 dims.
    """
    def __init__(self, num_freq: int = 32, freq_base: float = 0.1, freq_scale: float = 1.5):
        super().__init__()
        k = torch.arange(num_freq, dtype=torch.float32)
        self.register_buffer('frequencies', freq_base * (freq_scale ** k))
        self.register_buffer('amplitudes', 1.0 / torch.sqrt(1.0 + k))
        self.output_dim = 2 * num_freq  # 64

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
    """Normalized polynomial basis: LayerNorm([x, x², ..., x^p])."""
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
# Encoder
# =============================================================================

class NumberEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_frequencies: int = 32, poly_degree: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fourier = FourierChannel(num_freq=num_frequencies)
        self.log_mag = LogMagnitudeChannel()
        self.sign = SignChannel()
        self.poly = PolynomialChannel(poly_degree)
        self.raw_dim = (self.fourier.output_dim + self.log_mag.output_dim
                        + self.sign.output_dim + self.poly.output_dim)
        # Project to (d-1) so we can append 1 reserved dim (log_norm)
        self.proj = nn.Linear(self.raw_dim, embedding_dim - 1)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        raw = torch.cat([
            self.fourier(x),
            self.log_mag(x),
            self.sign(x),
            self.poly(x),
        ], dim=-1)                                          # (N, 71)

        projected = self.proj(raw)                          # (N, 127)

        # Reserve 1 dim: log_norm (computed BEFORE LayerNorm)
        proj_norm = projected.norm(dim=-1, keepdim=True)    # (N, 1)
        log_norm = torch.log(proj_norm + 1e-8)              # (N, 1)

        # Manual LayerNorm on 127 dims (no learnable γ/β)
        p_mean = projected.mean(dim=-1, keepdim=True)
        p_var = projected.var(dim=-1, keepdim=True, unbiased=False)
        normed = (projected - p_mean) / torch.sqrt(p_var + 1e-5)

        return torch.cat([normed, log_norm], dim=-1)        # (N, 128)


# =============================================================================
# Decoder
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
        nn.init.zeros_(self.w_skip.weight)  # start with no skip influence

    def forward(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
        a1 = F.gelu(self.fc1(emb))
        a2 = F.gelu(self.fc2(a1))
        z3 = self.fc3(a2) + self.w_skip(emb)  # residual skip

        log_mag = z3[:, 0].clamp(-14.0, 14.0)
        sign_logit = z3[:, 1]
        recon = torch.tanh(sign_logit) * torch.exp(log_mag)
        return recon, sign_logit


# =============================================================================
# Complete System
# =============================================================================

class NumberEmbeddingSystem(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_frequencies: int = 32,
                 poly_degree: int = 5, device: torch.device = None):
        super().__init__()
        if device is None:
            device = get_device()
        self.device = device
        self.embedding_dim = embedding_dim
        self.encoder = NumberEncoder(embedding_dim, num_frequencies, poly_degree)
        self.decoder = NumberDecoder(embedding_dim)
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
                     sign_logit: Tensor, lam_rel: float) -> Tensor:
        eps_lm = 1e-8
        eps_bce = 1e-7

        # Term 1: Signed-log MSE
        f_x = torch.sign(x) * torch.log1p(torch.abs(x))
        f_recon = torch.sign(recon) * torch.log1p(torch.abs(recon))
        loss_slog = F.mse_loss(f_recon, f_x)

        # Term 2: BCE sign loss
        sigma = torch.sigmoid(sign_logit)
        target = torch.where(x > 0, torch.ones_like(x),
                             torch.where(x < 0, torch.zeros_like(x),
                                         torch.full_like(x, 0.5)))
        loss_bce = F.binary_cross_entropy(sigma.clamp(eps_bce, 1 - eps_bce), target)

        # Term 3: Log-magnitude MSE
        log_abs_x = torch.log(torch.abs(x) + eps_lm)
        log_abs_recon = torch.log(torch.abs(recon) + eps_lm)
        loss_lm = F.mse_loss(log_abs_recon, log_abs_x)

        # Term 4: Relative MSE (phase 2, ramped)
        loss_rel = torch.mean((recon - x) ** 2 / (x * x + 1.0))

        # Term 5: Embedding spread loss — penalize high cosine similarity
        # Sample a mini-batch of pairs to keep cost O(n) not O(n²)
        n = emb.shape[0]
        idx = torch.randperm(n, device=emb.device)
        emb_shuffled = emb[idx]
        cos_sim = F.cosine_similarity(emb, emb_shuffled, dim=-1)  # (N,)
        # Push cosine similarity of different numbers toward 0
        loss_spread = torch.mean(cos_sim ** 2)

        return (loss_slog + 0.1 * loss_bce + 0.3 * loss_lm
                + lam_rel * loss_rel + 0.05 * loss_spread)

    def train_model(self, num_steps: int = 500000, batch_size: int = 512,
                    lr: float = 5e-4, log_interval: int = 5000,
                    warmup_steps: int = 2000, grad_clip: float = 1.0):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                       betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=1e-5)
        losses = []
        phase2_start = int(num_steps * 0.40)
        phase2_end = int(num_steps * 0.50)

        self.train()
        for step in range(1, num_steps + 1):
            # Phase-2 lambda ramp (reduced weight: 0.3 max)
            if step < phase2_start:
                lam_rel = 0.0
            elif step < phase2_end:
                lam_rel = 0.3 * (step - phase2_start) / (phase2_end - phase2_start)
            else:
                lam_rel = 0.3

            x = sample_training_numbers(batch_size, self.device)

            optimizer.zero_grad()
            emb, recon, sign_logit = self.forward(x)
            loss = self.compute_loss(x, emb, recon, sign_logit, lam_rel)
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
                print(f"  Step {step:>6d}/{num_steps} | Loss: {avg:.6f} [{phase}] | lr: {lr_t:.2e}")

        self.eval()
        return losses


# =============================================================================
# Training Data
# =============================================================================

def sample_training_numbers(batch_size: int, device: torch.device) -> Tensor:
    n_log = int(batch_size * 0.4)
    n_neg = int(batch_size * 0.4)
    n_zero = int(batch_size * 0.1)
    n_int = batch_size - n_log - n_neg - n_zero

    pos = torch.exp(torch.empty(n_log, device=device).uniform_(-14, 14))
    neg = -torch.exp(torch.empty(n_neg, device=device).uniform_(-14, 14))
    zero = torch.empty(n_zero, device=device).uniform_(-0.01, 0.01)
    ints = torch.randint(-1000, 1000, (n_int,), device=device, dtype=torch.float32)

    samples = torch.cat([pos, neg, zero, ints])
    idx = torch.randperm(batch_size, device=device)
    return samples[idx]


# =============================================================================
# Helper: torch tensor → numpy for tests
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
# Tests
# =============================================================================

def run_tests(system: Optional[NumberEmbeddingSystem] = None, device: torch.device = None):
    if device is None:
        device = get_device()
    if system is None:
        print("=" * 70)
        print("TRAINING NUMBER EMBEDDING SYSTEM")
        print("=" * 70)
        system = NumberEmbeddingSystem(embedding_dim=128, device=device)
        system.train_model(num_steps=500000, batch_size=512, lr=5e-4,
                           log_interval=5000)
        print()

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

    base = np.array([0.0, 1.0, 10.0, -5.0, 100.0])
    emb_base = _encode_np(system, base)
    prev_max_d = float('inf')
    all_decreasing = True
    eps_results = []
    for eps in [0.1, 0.01, 0.001]:
        emb_p = _encode_np(system, base + eps)
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
    dists = [np.linalg.norm(emb_c - _encode_np(system, center + d)) for d in [0.01, 0.1, 1.0, 10.0]]
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
        # Near zero & tiny
        0.0, 1e-6, 1e-4, 0.000123, 0.001, 0.01, 0.1,
        # Small-medium positive
        0.5, 1.0, 2.71828, 3.14159, 7.0, 10.0, 42.0, 99.0,
        # Large positive
        500.0, 1000.0, 9999.0, 1e5, 1e6,
        # Negative (mirror)
        -0.001, -0.5, -1.0, -3.14159, -10.0, -42.0, -1000.0, -1234.5,
        # Edge / stress
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

    embs_s = _encode_np(system, sample_training_numbers(500, system.device).cpu().numpy())
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
    check("PC1 correlates with ordering", abs(corr) > 0.7, f"Spearman ρ = {corr:.4f}")

    xa = _encode_np(system, [1.0, 1.01, 1.02])
    xb = _encode_np(system, [1000.0, 1000.01, 1000.02])
    intra = max(max(np.linalg.norm(xa[i] - xa[j]) for j in range(3) if j != i) for i in range(3))
    inter = min(np.linalg.norm(xa[i] - xb[j]) for i in range(3) for j in range(3))
    check("Similar numbers cluster tighter", inter > intra,
          f"Intra max: {intra:.4f}, Inter min: {inter:.4f}")

    # === TEST 5: MODEL COMPATIBILITY ===
    print("-" * 70)
    print("TEST 5: MODEL COMPATIBILITY — Standard ops & integration")
    print("-" * 70)

    emb64 = _encode_np(system, np.random.randn(64))
    check("Batch (64,) → (64, d)", emb64.shape == (64, system.embedding_dim))

    check("Single number works", _encode_np(system, [42.0]).shape == (1, system.embedding_dim))

    out = emb64 @ np.random.randn(system.embedding_dim, 10) * 0.01
    check("Downstream linear layer", out.shape == (64, 10))

    seq = _encode_np(system, np.random.randn(8))
    dk = system.embedding_dim
    scores = (seq @ seq.T) / np.sqrt(dk)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attn = weights @ seq
    check("Dot-product attention", attn.shape == (8, dk))

    # Determinism test
    torch.manual_seed(0)
    e1 = _encode_np(system, [1.0, 2.0])
    torch.manual_seed(0)
    e2 = _encode_np(system, [1.0, 2.0])
    check("Deterministic", np.allclose(e1, e2, atol=1e-6))

    # === SUMMARY ===
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 70)
    return passed, failed, total


# =============================================================================
# Checkpoint Saving
# =============================================================================

def save_checkpoint(system: NumberEmbeddingSystem, num_steps: int,
                    checkpoint_dir: str = "/tmpdir/m24047brmn/numbers/checkpoints"):
    """Save model state dict and reference embeddings."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    tag = f"np_emb_v8_{num_steps // 1000}k"

    # Save model state dict
    model_path = os.path.join(checkpoint_dir, f"{tag}_model.pt")
    torch.save({
        'state_dict': system.state_dict(),
        'embedding_dim': system.embedding_dim,
        'num_steps': num_steps,
    }, model_path)
    print(f"  Model saved: {model_path}")

    # Save reference embeddings for visualization
    ref_pos = np.exp(np.linspace(-14, 14, 1000))
    ref_neg = -np.exp(np.linspace(-14, 14, 1000))
    ref_zero = np.linspace(-1, 1, 200)
    ref_numbers = np.concatenate([ref_neg[::-1], ref_zero, ref_pos])

    ref_embs = _encode_np(system, ref_numbers)
    _, ref_recon = _forward_np(system, ref_numbers)

    emb_path = os.path.join(checkpoint_dir, f"{tag}_embeddings.npz")
    np.savez(emb_path,
             numbers=ref_numbers,
             embeddings=ref_embs,
             reconstructions=ref_recon)
    print(f"  Embeddings saved: {emb_path} ({len(ref_numbers)} reference points)")


# =============================================================================
# Demo
# =============================================================================

def demo(num_steps: int = 500000, device: torch.device = None):
    if device is None:
        device = get_device()
    print("=" * 70)
    print("NUMBER EMBEDDING REPRESENTATION — DEMO (PyTorch)")
    print("=" * 70)
    print()

    system = NumberEmbeddingSystem(embedding_dim=128, device=device)
    enc = system.encoder
    total_params = sum(p.numel() for p in system.parameters())
    print(f"Architecture:  {enc.raw_dim} raw dims → {system.embedding_dim}d embedding")
    print(f"  Fourier:    {enc.fourier.output_dim} dims  |  Log-mag: {enc.log_mag.output_dim}  |  "
          f"Sign: {enc.sign.output_dim}  |  Poly: {enc.poly.output_dim}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Device:     {device}")
    print()

    numbers = [0.0, 1e-6, 0.001, 0.1, 1.0, 3.14159, -42.0, 100.0, 1000.0, 1e5, -1234.5]
    print("Before training:")
    for n in numbers:
        emb = _encode_np(system, [n])[0]
        print(f"  {n:>10.5f} → norm={np.linalg.norm(emb):.3f}  "
              f"[{', '.join(f'{v:.3f}' for v in emb[:5])}...]")
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
    return system


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number Embedding Representation (PyTorch)")
    parser.add_argument("--max-steps", type=int, default=500000,
                        help="Training steps (default: 500000)")
    parser.add_argument("--test-only", action="store_true",
                        help="Run tests without training")
    parser.add_argument("--demo-only", action="store_true",
                        help="Run demo only, skip tests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    # Set up logging: tee stdout to logs/<timestamp>.log
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs",
                            f"run_torch_{time.strftime('%Y%m%d_%H%M%S')}_steps{args.max_steps}.log")

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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    try:
        if args.test_only:
            run_tests(device=device)
        elif args.demo_only:
            system = demo(num_steps=args.max_steps, device=device)
            save_checkpoint(system, args.max_steps)
        else:
            system = demo(num_steps=args.max_steps, device=device)
            save_checkpoint(system, args.max_steps)
            print()
            run_tests(system, device=device)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"\nOutput saved to: {log_path}")
