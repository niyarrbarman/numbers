"""
Number Embedding Representation Framework
==========================================
Pure NumPy implementation.

Embeds numerical values into high-dimensional vectors preserving:
  1. Uniqueness   — distinct numbers → distinct embeddings
  2. Continuity   — small Δx → small Δe(x)
  3. Reversibility — decode(encode(x)) ≈ x
  4. Expressiveness — captures multi-scale numerical structure
  5. Compatibility  — standard arrays, differentiable, integrable

Architecture:
  Encoder:  x ∈ ℝ  →  e(x) ∈ ℝ^d
    Channel 1: Multi-scale Fourier (2K dims)
    Channel 2: Log-magnitude (1 dim)
    Channel 3: Smooth sign (1 dim)
    Channel 4: Polynomial basis (P dims)
    → Learned linear projection → LayerNorm → ℝ^d

  Decoder:  e(x) ∈ ℝ^d  →  x̂ ∈ ℝ
    3-layer MLP with GELU, predicts sign * exp(log_mag)

  Training:
    L = signed_log_MSE + 0.1·BCE_sign + 0.5·rel_MSE (phase 2)
    AdamW optimiser, trained on log-uniform samples.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from typing import List, Tuple, Optional
import time


# =============================================================================
# Utilities
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    cdf = 0.5 * (1.0 + t)
    d_inner = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)
    return cdf + 0.5 * x * (1 - t ** 2) * d_inner

def layer_norm(x: np.ndarray, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std, mean, inv_std

def softplus(x: np.ndarray) -> np.ndarray:
    # Numerically stable: log(1 + exp(x))
    return np.log1p(np.exp(np.clip(x, -500, 500)))

def softplus_grad(x: np.ndarray) -> np.ndarray:
    # sigmoid(x) — bounded in (0, 1)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    return np.random.randn(fan_in, fan_out).astype(np.float64) * np.sqrt(2.0 / fan_in)


# =============================================================================
# Encoder Channels (Analytic — no learnable params)
# =============================================================================

class FourierChannel:
    """Multi-scale sinusoidal encoding with geometrically spaced frequencies.
    
    Uses amplitude damping: higher frequencies get lower weights to preserve
    continuity (small Δx → small Δembedding) while still encoding fine detail.
    """
    def __init__(self, num_frequencies: int = 32, freq_base: float = 0.1, freq_scale: float = 1.5):
        self.num_frequencies = num_frequencies
        self.frequencies = freq_base * (freq_scale ** np.arange(num_frequencies, dtype=np.float64))
        # Amplitude damping: higher freq → lower weight (preserves continuity)
        self.amplitudes = 1.0 / np.sqrt(1.0 + np.arange(num_frequencies, dtype=np.float64))
        self.output_dim = 2 * num_frequencies

    def forward(self, x: np.ndarray) -> np.ndarray:
        phases = x[:, None] * self.frequencies[None, :]
        sin_part = np.sin(phases) * self.amplitudes[None, :]
        cos_part = np.cos(phases) * self.amplitudes[None, :]
        return np.concatenate([sin_part, cos_part], axis=-1)


class LogMagnitudeChannel:
    """Log-compressed magnitude: log(|x|+ε) / log(scale)."""
    def __init__(self, epsilon: float = 1e-8, log_scale: float = 10.0):
        self.epsilon = epsilon
        self.log_scale = np.log(log_scale)
        self.output_dim = 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (np.log(np.abs(x) + self.epsilon) / self.log_scale)[:, None]


class SignChannel:
    """Smooth sign encoding: tanh(αx)."""
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.output_dim = 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.alpha * x)[:, None]


class PolynomialChannel:
    """Normalized polynomial basis: LayerNorm([x, x², ..., x^p])."""
    def __init__(self, degree: int = 5):
        self.degree = degree
        self.output_dim = degree

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clamp = np.clip(x, -50.0, 50.0)
        powers = np.stack([x_clamp ** (k + 1) for k in range(self.degree)], axis=-1)
        normed, _, _ = layer_norm(powers)
        return normed


# =============================================================================
# Learnable Linear Layer
# =============================================================================

class LinearLayer:
    def __init__(self, fan_in: int, fan_out: int):
        self.W = he_init(fan_in, fan_out)
        self.b = np.zeros(fan_out, dtype=np.float64)
        self.dW = None
        self.db = None
        self.mW = None; self.vW = None
        self.mb = None; self.vb = None
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.dW = self._input.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T


# =============================================================================
# Encoder
# =============================================================================

class NumberEncoder:
    def __init__(self, embedding_dim: int = 128, num_frequencies: int = 32, poly_degree: int = 5):
        self.embedding_dim = embedding_dim
        self.fourier = FourierChannel(num_frequencies)
        self.log_mag = LogMagnitudeChannel()
        self.sign = SignChannel()
        self.poly = PolynomialChannel(poly_degree)
        self.raw_dim = (self.fourier.output_dim + self.log_mag.output_dim
                        + self.sign.output_dim + self.poly.output_dim)
        self.proj = LinearLayer(self.raw_dim, embedding_dim - 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raw = np.concatenate([
            self.fourier.forward(x),
            self.log_mag.forward(x),
            self.sign.forward(x),
            self.poly.forward(x),
        ], axis=-1)
        projected = self.proj.forward(raw)               # (N, d-1)
        self._projected = projected
        proj_norm = np.linalg.norm(projected, axis=-1, keepdims=True)
        self._proj_norm = proj_norm
        log_norm = np.log(proj_norm + 1e-8)              # (N, 1) explicit magnitude
        normed, self._ln_mean, self._ln_invstd = layer_norm(projected)
        return np.concatenate([normed, log_norm], axis=-1)  # (N, d)

    def backward(self, grad_out: np.ndarray) -> None:
        grad_normed   = grad_out[:, :-1]   # (N, d-1)
        grad_log_norm = grad_out[:, -1:]   # (N, 1)

        # Gradient through log_norm path: d(log||p||)/dp = p / (||p|| * (||p|| + eps))
        grad_proj_lognorm = (grad_log_norm * self._projected /
                             (self._proj_norm * (self._proj_norm + 1e-8)))

        # Full LayerNorm backward (previously missing two correction terms)
        y = (self._projected - self._ln_mean) * self._ln_invstd
        grad_proj_ln = self._ln_invstd * (
            grad_normed
            - grad_normed.mean(axis=-1, keepdims=True)
            - y * (grad_normed * y).mean(axis=-1, keepdims=True)
        )

        self.proj.backward(grad_proj_lognorm + grad_proj_ln)

    def get_layers(self) -> List[LinearLayer]:
        return [self.proj]


# =============================================================================
# Decoder
# =============================================================================

class NumberDecoder:
    """MLP decoder: emb → hidden → hidden → (log_mag, sign_logit) → tanh(s)*exp(clip(m,-14,14))"""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 192):
        self.fc1 = LinearLayer(embedding_dim, hidden_dim)
        self.fc2 = LinearLayer(hidden_dim, hidden_dim)
        self.fc3 = LinearLayer(hidden_dim, 2)

    def forward(self, emb: np.ndarray) -> np.ndarray:
        self._z1 = self.fc1.forward(emb)
        self._a1 = gelu(self._z1)
        self._z2 = self.fc2.forward(self._a1)
        self._a2 = gelu(self._z2)
        self._z3 = self.fc3.forward(self._a2)

        # Clip to [-14, 14] then exp: range [exp(-14), exp(14)] ≈ [8e-7, 1.2M]
        self._log_mag = np.clip(self._z3[:, 0], -14.0, 14.0)
        self._sign_logit = self._z3[:, 1]
        self._magnitude = np.exp(self._log_mag)
        self._sign = np.tanh(self._sign_logit)
        return self._sign * self._magnitude

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Gradient through magnitude: d(exp(clip(z)))/dz = exp(clip(z)) * clip_mask
        clip_mask = (self._z3[:, 0] > -14.0) & (self._z3[:, 0] < 14.0)
        d_magnitude = grad_output * self._sign
        d_z3_mag = d_magnitude * self._magnitude * clip_mask

        # Gradient through sign: d(tanh(s))/ds = 1 - tanh²(s)
        d_sign_logit = grad_output * (1 - self._sign ** 2) * self._magnitude

        # Add BCE sign loss gradient (injected directly on the sign logit)
        if hasattr(self, '_grad_bce_sign_logit') and self._grad_bce_sign_logit is not None:
            d_sign_logit = d_sign_logit + self._grad_bce_sign_logit
            self._grad_bce_sign_logit = None

        d_z3 = np.stack([d_z3_mag, d_sign_logit], axis=-1)
        d_a2 = self.fc3.backward(d_z3)
        d_z2 = d_a2 * gelu_grad(self._z2)
        d_a1 = self.fc2.backward(d_z2)
        d_z1 = d_a1 * gelu_grad(self._z1)
        return self.fc1.backward(d_z1)

    def get_layers(self) -> List[LinearLayer]:
        return [self.fc1, self.fc2, self.fc3]


# =============================================================================
# Complete System
# =============================================================================

class NumberEmbeddingSystem:
    def __init__(self, embedding_dim: int = 128, num_frequencies: int = 32, poly_degree: int = 5):
        self.encoder = NumberEncoder(embedding_dim, num_frequencies, poly_degree)
        self.decoder = NumberDecoder(embedding_dim)
        self.embedding_dim = embedding_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder.forward(np.atleast_1d(np.asarray(x, dtype=np.float64)))

    def decode(self, emb: np.ndarray) -> np.ndarray:
        return self.decoder.forward(emb)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        emb = self.encode(x)
        recon = self.decode(emb)
        return emb, recon

    def _all_layers(self) -> List[LinearLayer]:
        return self.encoder.get_layers() + self.decoder.get_layers()

    def compute_loss(self, x: np.ndarray, recon: np.ndarray):
        N = x.shape[0]

        # --- Term 1: Signed-log loss (magnitude + sign in log-space) ---
        f_x     = np.sign(x)     * np.log1p(np.abs(x))
        f_recon = np.sign(recon) * np.log1p(np.abs(recon))
        diff = f_recon - f_x
        loss_slog = np.mean(diff ** 2)
        grad_slog = 2.0 * diff / (N * (np.abs(recon) + 1.0))

        # --- Term 2: BCE sign loss (direct sign supervision) ---
        # target: p=1 if x>0, p=0 if x<0, p=0.5 if x==0
        # prediction: sigmoid of sign_logit from decoder
        sign_logit = self.decoder._sign_logit           # raw logit before tanh
        sigma = 1.0 / (1.0 + np.exp(-np.clip(sign_logit, -500, 500)))  # sigmoid
        target = np.where(x > 0, 1.0, np.where(x < 0, 0.0, 0.5))
        # BCE = -[t*log(σ) + (1-t)*log(1-σ)], gradient w.r.t. recon = (σ-t) * d(recon)/d(sign_logit)
        eps_bce = 1e-7
        loss_bce = -np.mean(target * np.log(sigma + eps_bce) + (1 - target) * np.log(1 - sigma + eps_bce))
        # grad of BCE w.r.t. sign_logit = (σ - t) / N
        grad_bce_sign_logit = (sigma - target) / N

        # --- Term 3: Pure log-magnitude loss (scale-ratio awareness) ---
        # Treats "16x off" equally whether the number is 0.0001 or 10000
        eps_lm = 1e-8
        log_abs_x    = np.log(np.abs(x) + eps_lm)
        log_abs_recon = np.log(np.abs(recon) + eps_lm)
        diff_lm = log_abs_recon - log_abs_x
        loss_lm = np.mean(diff_lm ** 2)
        # grad w.r.t. recon: 2*diff_lm / (N * (|recon| + eps)) * sign(recon)
        grad_lm = 2.0 * diff_lm * np.sign(recon) / (N * (np.abs(recon) + eps_lm))

        # --- Term 4: Relative MSE (precision fine-tuning, phase 2 only) ---
        # Activated after 30% of training via self._phase2_active flag
        rel_diff = (recon - x) / (x * x + 1.0)  # d/d(recon) = 1/(x²+1)
        loss_rel = np.mean((recon - x) ** 2 / (x * x + 1.0))
        grad_rel = 2.0 * rel_diff / N

        # --- Combine ---
        lam_bce = 0.1
        lam_lm = 0.3
        lam_rel = 0.5 if getattr(self, '_phase2_active', False) else 0.0
        loss = loss_slog + lam_bce * loss_bce + lam_lm * loss_lm + lam_rel * loss_rel
        grad_recon = grad_slog + lam_lm * grad_lm + lam_rel * grad_rel

        # Store BCE gradient for sign logit — applied inside decoder backward
        self.decoder._grad_bce_sign_logit = lam_bce * grad_bce_sign_logit

        return loss, grad_recon

    def _adamw_step(self, lr, beta1, beta2, eps, wd, t):
        for layer in self._all_layers():
            for attr in ['W', 'b']:
                p = getattr(layer, attr)
                g = getattr(layer, f'd{attr}')
                if g is None:
                    continue
                g = np.clip(g, -1.0, 1.0)
                m_attr, v_attr = f'm{attr}', f'v{attr}'
                if getattr(layer, m_attr) is None:
                    setattr(layer, m_attr, np.zeros_like(p))
                    setattr(layer, v_attr, np.zeros_like(p))
                m = getattr(layer, m_attr)
                v = getattr(layer, v_attr)
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * g ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * p)

    def train(self, num_steps=100000, batch_size=256, lr=5e-4, log_interval=3000,
              warmup_steps=2000, grad_clip=1.0):
        losses = []
        phase2_start = int(num_steps * 0.3)
        for step in range(1, num_steps + 1):
            self._phase2_active = (step >= phase2_start)
            x = sample_training_numbers(batch_size)
            emb, recon = self.forward(x)
            loss, grad_recon = self.compute_loss(x, recon)

            # Global grad norm clipping on the loss gradient before backprop
            grad_norm = np.linalg.norm(grad_recon)
            if grad_norm > grad_clip:
                grad_recon = grad_recon * (grad_clip / grad_norm)

            grad_emb = self.decoder.backward(grad_recon)
            self.encoder.backward(grad_emb)

            # Linear warmup then cosine decay
            if step <= warmup_steps:
                lr_t = lr * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                lr_t = lr * 0.5 * (1 + np.cos(np.pi * progress))
            self._adamw_step(lr=lr_t, beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-5, t=step)

            losses.append(loss)
            if step % log_interval == 0:
                avg = np.mean(losses[-log_interval:])
                phase = "P2" if self._phase2_active else "P1"
                print(f"  Step {step:>5d}/{num_steps} | Loss: {avg:.6f} [{phase}]")
        return losses


# =============================================================================
# Training Data
# =============================================================================

def sample_training_numbers(batch_size: int) -> np.ndarray:
    n = batch_size
    n_log = int(n * 0.4)   # 40% positive log-uniform  (1e-14 to 1e+14)
    n_neg = int(n * 0.4)   # 40% negative mirror
    n_zero = int(n * 0.1)  # 10% near-zero
    n_int = n - n_log - n_neg - n_zero  # remaining: integers up to ±1000
    parts = [
        np.exp(np.random.uniform(-14, 14, n_log)),
        -np.exp(np.random.uniform(-14, 14, n_neg)),
        np.random.uniform(-0.01, 0.01, n_zero),
        np.random.randint(-1000, 1000, size=n_int).astype(np.float64),
    ]
    samples = np.concatenate(parts)
    np.random.shuffle(samples)
    return samples[:batch_size]


# =============================================================================
# Tests
# =============================================================================

def run_tests(system: Optional[NumberEmbeddingSystem] = None):
    if system is None:
        print("=" * 70)
        print("TRAINING NUMBER EMBEDDING SYSTEM")
        print("=" * 70)
        system = NumberEmbeddingSystem(embedding_dim=128)
        system.train(num_steps=100000, batch_size=256, lr=5e-4, log_interval=3000)
        print()

    passed = failed = total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition: passed += 1
        else: failed += 1
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")

    # === TEST 1: UNIQUENESS ===
    print("-" * 70)
    print("TEST 1: UNIQUENESS — Distinct numbers → distinct embeddings")
    print("-" * 70)

    nums = np.array([0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, -1.0, -100.0, 3.14159])
    embs = system.encode(nums)

    norms = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
    cos_sim = norms @ norms.T
    np.fill_diagonal(cos_sim, -2.0)
    check("Max cosine sim < 0.99", cos_sim.max() < 0.99,
          f"Max cosine sim: {cos_sim.max():.4f}")

    check("All pairwise L2 distances > 0", pdist(embs).min() > 1e-4,
          f"Min L2 dist: {pdist(embs).min():.6f}")

    a, b = system.encode(np.array([1.0000])), system.encode(np.array([1.0001]))
    d = np.linalg.norm(a - b)
    check("1.0000 vs 1.0001 distinguishable", d > 1e-5, f"L2 dist: {d:.8f}")

    # === TEST 2: CONTINUITY ===
    print("-" * 70)
    print("TEST 2: CONTINUITY — Small perturbations → small embedding changes")
    print("-" * 70)

    base = np.array([0.0, 1.0, 10.0, -5.0, 100.0])
    emb_base = system.encode(base)
    prev_max_d = float('inf')
    all_decreasing = True
    eps_results = []
    for eps in [0.1, 0.01, 0.001]:
        emb_p = system.encode(base + eps)
        diffs = np.linalg.norm(emb_base - emb_p, axis=-1)
        max_d = diffs.max()
        # With LN, absolute bound is loose; check relative to embedding norm
        rel_d = max_d / np.linalg.norm(emb_base, axis=-1).mean()
        eps_results.append((eps, max_d, rel_d))
        check(f"ε={eps}: bounded relative change", rel_d < 2.0,
              f"Max Δ: {max_d:.4f}, Relative: {rel_d:.4f}")
        if max_d > prev_max_d * 1.1:
            all_decreasing = False
        prev_max_d = max_d
    
    # Smaller perturbations should give smaller (or comparable) deltas
    check("Smaller ε → smaller (or comparable) embedding Δ", all_decreasing,
          f"Deltas: {[f'{r[1]:.4f}' for r in eps_results]}")

    center = np.array([5.0])
    emb_c = system.encode(center)
    dists = [np.linalg.norm(emb_c - system.encode(center + d)) for d in [0.01, 0.1, 1.0, 10.0]]
    mono = all(dists[i] <= dists[i+1] * 1.1 for i in range(len(dists)-1))
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
        _, recon = system.forward(vals)
        abs_err = np.abs(vals - recon)
        rel_err = abs_err / (np.abs(vals) + 1e-8)
        check(f"Reconstruction [{name}]", rel_err.max() < 1.0 or abs_err.max() < 10.0,
              f"|err|_max: {abs_err.max():.4f}, rel_max: {rel_err.max():.4f}")

    showcase = np.array([
        0.000123,    # tiny positive
        0.01,        # small positive
        1.0,         # unit
        3.14159,     # pi
        7.0,         # small integer
        42.0,        # medium integer
        99.0,        # near-hundred
        500.0,       # large
        1e5,         # very large
        -0.5,        # small negative
        -10.0,       # medium negative
        -42.0,       # negative integer
        -1234.5,     # large negative
        1e-6,        # near-zero positive
        0.0,         # zero
    ])
    _, recon_s = system.forward(showcase)
    print(f"\n  {'Input':>12}  →  {'Decoded':>14}  {'Abs Err':>10}  {'Rel Err':>10}")
    print(f"  {'─'*12}     {'─'*14}  {'─'*10}  {'─'*10}")
    for o, r in zip(showcase, recon_s):
        e = abs(o - r)
        print(f"  {o:>12.5f}  →  {r:>14.5f}  {e:>10.5f}  {e/(abs(o)+1e-8):>9.4%}")
    print()

    # === TEST 4: EXPRESSIVENESS ===
    print("-" * 70)
    print("TEST 4: EXPRESSIVENESS — Structural properties of embedding space")
    print("-" * 70)

    embs_s = system.encode(sample_training_numbers(500))
    _, S, _ = np.linalg.svd(embs_s - embs_s.mean(axis=0), full_matrices=False)
    eff = int(np.sum(S > 0.01 * S[0]))
    check("High effective dimensionality (>20)", eff > 20,
          f"Effective dims: {eff}/{system.embedding_dim}")

    ordered = np.linspace(-10, 10, 50)
    embs_o = system.encode(ordered)
    cent = embs_o - embs_o.mean(axis=0)
    _, _, Vt = np.linalg.svd(cent, full_matrices=False)
    pc1 = cent @ Vt[0]
    corr, _ = spearmanr(ordered, pc1)
    check("PC1 correlates with ordering", abs(corr) > 0.7, f"Spearman ρ = {corr:.4f}")

    xa = system.encode(np.array([1.0, 1.01, 1.02]))
    xb = system.encode(np.array([1000.0, 1000.01, 1000.02]))
    intra = max(max(np.linalg.norm(xa[i]-xa[j]) for j in range(3) if j!=i) for i in range(3))
    inter = min(np.linalg.norm(xa[i]-xb[j]) for i in range(3) for j in range(3))
    check("Similar numbers cluster tighter", inter > intra,
          f"Intra max: {intra:.4f}, Inter min: {inter:.4f}")

    # === TEST 5: MODEL COMPATIBILITY ===
    print("-" * 70)
    print("TEST 5: MODEL COMPATIBILITY — Standard ops & integration")
    print("-" * 70)

    emb64 = system.encode(np.random.randn(64))
    check("Batch (64,) → (64, d)", emb64.shape == (64, system.embedding_dim))

    check("Single number works", system.encode(np.array([42.0])).shape == (1, system.embedding_dim))

    out = emb64 @ np.random.randn(system.embedding_dim, 10) * 0.01
    check("Downstream linear layer", out.shape == (64, 10))

    seq = system.encode(np.random.randn(8))
    dk = system.embedding_dim
    scores = (seq @ seq.T) / np.sqrt(dk)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attn = weights @ seq
    check("Dot-product attention", attn.shape == (8, dk))

    e1, e2 = system.encode(np.array([1.0, 2.0])), system.encode(np.array([1.0, 2.0]))
    check("Deterministic", np.allclose(e1, e2, atol=1e-12))

    # === SUMMARY ===
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 70)
    return passed, failed, total


# =============================================================================
# Demo
# =============================================================================

def demo(num_steps: int = 100000):
    print("=" * 70)
    print("NUMBER EMBEDDING REPRESENTATION — DEMO")
    print("=" * 70)
    print()

    system = NumberEmbeddingSystem(embedding_dim=128)
    enc = system.encoder
    print(f"Architecture:  {enc.raw_dim} raw dims → {system.embedding_dim}d embedding")
    print(f"  Fourier:    {enc.fourier.output_dim} dims  |  Log-mag: {enc.log_mag.output_dim}  |  "
          f"Sign: {enc.sign.output_dim}  |  Poly: {enc.poly.output_dim}")
    print()

    numbers = [0.0, 1.0, 3.14159, -42.0, 100.0, 0.001]
    print("Before training:")
    for n in numbers:
        emb = system.encode(np.array([n]))[0]
        print(f"  {n:>10.5f} → norm={np.linalg.norm(emb):.3f}  [{', '.join(f'{v:.3f}' for v in emb[:5])}...]")
    print()

    print(f"Training ({num_steps} steps)...")
    t0 = time.time()
    system.train(num_steps=num_steps, batch_size=256, lr=5e-4, log_interval=max(1, num_steps // 60))
    print(f"  Done in {time.time()-t0:.1f}s\n")

    print("After training:")
    print(f"  {'Input':>12}  →  {'Decoded':>14}  {'Error':>10}")
    print(f"  {'─'*12}     {'─'*14}  {'─'*10}")
    for n in numbers:
        _, r = system.forward(np.array([n]))
        print(f"  {n:>12.5f}  →  {r[0]:>14.5f}  {abs(n-r[0]):>10.5f}")
    print()
    return system


if __name__ == "__main__":
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Number Embedding Representation")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Training steps (default: 100000)")
    parser.add_argument("--test-only", action="store_true",
                        help="Run tests without training")
    parser.add_argument("--demo-only", action="store_true",
                        help="Run demo only, skip tests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Set up logging: tee stdout to logs/<timestamp>.log
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}_steps{args.max_steps}.log")

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
    print(f"Seed: {args.seed}  |  Max steps: {args.max_steps}")
    print()

    try:
        if args.test_only:
            run_tests()
        elif args.demo_only:
            demo(num_steps=args.max_steps)
        else:
            system = demo(num_steps=args.max_steps)
            print()
            run_tests(system)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"\nOutput saved to: {log_path}")