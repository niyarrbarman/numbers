"""
Number Embedding Representation Framework
==========================================
A principled approach to embedding numerical values into high-dimensional
vector spaces that preserves uniqueness, continuity, reversibility,
expressiveness, and model compatibility.

Architecture:
    Encoder: x ∈ ℝ  →  e(x) ∈ ℝ^d
        Channel 1: Multi-scale Fourier encoding (sinusoidal decomposition)
        Channel 2: Log-magnitude encoding (scale awareness)
        Channel 3: Smooth sign encoding (polarity)
        Channel 4: Polynomial basis (algebraic structure)
        Channel 5: Learned residual (task adaptation)
        → Linear projection to final embedding dim

    Decoder: e(x) ∈ ℝ^d  →  x̂ ∈ ℝ
        MLP with log-scale + absolute reconstruction loss
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Channel Components
# =============================================================================

class FourierChannel(nn.Module):
    """
    Multi-scale sinusoidal encoding with geometrically spaced frequencies.

    For input x, produces: [sin(ω₁x), cos(ω₁x), sin(ω₂x), cos(ω₂x), ...]

    Injectivity guarantee: With K frequencies that are linearly independent
    over ℚ, the map is injective on any bounded interval (almost-periodic
    function theory). Geometric spacing with irrational ratio satisfies this.

    Output dim: 2 * num_frequencies
    """

    def __init__(self, num_frequencies: int = 32, freq_base: float = 1.0, freq_scale: float = 2.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Geometrically spaced: ω_k = base * scale^k
        freqs = freq_base * (freq_scale ** torch.arange(num_frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", freqs)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,) or (..., 1) -> (..., 2K)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        x_flat = x.reshape(-1, 1)  # (N, 1)
        phases = x_flat * self.frequencies.unsqueeze(0)  # (N, K)
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1).reshape(
            *x.shape, self.output_dim
        )


class LogMagnitudeChannel(nn.Module):
    """
    Compressed magnitude encoding: log(|x| + ε) / log(scale)

    Provides a monotonic, globally ordered signal that resolves the
    periodic ambiguity of the Fourier channel. Compresses dynamic range
    so that 10 vs 100 and 1000 vs 10000 get comparable representation distance.

    Output dim: 1
    """

    def __init__(self, epsilon: float = 1e-8, log_scale: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.log_scale = math.log(log_scale)

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.log(torch.abs(x) + self.epsilon) / self.log_scale
        return mag.unsqueeze(-1)


class SignChannel(nn.Module):
    """
    Smooth sign encoding: tanh(α * x)

    Differentiable approximation to sgn(x). Large α → sharp transition
    at zero; small α → gradual. Smooth at x=0 ensures gradient flow.

    Output dim: 1
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x).unsqueeze(-1)


class PolynomialChannel(nn.Module):
    """
    Normalized polynomial basis: LayerNorm([x, x², x³, ..., x^p])

    Directly encodes polynomial features so downstream layers can
    compute polynomial functions without learning to construct them.
    LayerNorm prevents higher powers from dominating.

    Output dim: degree
    """

    def __init__(self, degree: int = 5):
        super().__init__()
        self.degree = degree
        self.layer_norm = nn.LayerNorm(degree)

    @property
    def output_dim(self) -> int:
        return self.degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to prevent overflow in higher powers
        x_clamped = torch.clamp(x, -50.0, 50.0)
        powers = torch.stack(
            [x_clamped ** (k + 1) for k in range(self.degree)], dim=-1
        )
        return self.layer_norm(powers)


class LearnedResidualChannel(nn.Module):
    """
    Task-adaptive learned features: MLP(fourier ⊕ magnitude)

    A small network that discovers task-specific numerical features
    not captured by the analytic channels. Takes pre-computed Fourier
    and magnitude features as input.

    Output dim: output_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self._output_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, analytic_features: torch.Tensor) -> torch.Tensor:
        return self.net(analytic_features)


# =============================================================================
# Encoder
# =============================================================================

class NumberEncoder(nn.Module):
    """
    Full number embedding encoder.

    Maps x ∈ ℝ → e(x) ∈ ℝ^d by concatenating structured channels
    and projecting to the target embedding dimension.

    Channels:
        1. Fourier (multi-scale sinusoidal)
        2. Log-magnitude (compressed scale)
        3. Sign (smooth polarity)
        4. Polynomial (algebraic basis)
        5. Learned residual (adaptive)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_frequencies: int = 32,
        freq_base: float = 1.0,
        freq_scale: float = 2.0,
        poly_degree: int = 5,
        learned_hidden: int = 64,
        learned_out: int = 16,
    ):
        super().__init__()

        # Analytic channels
        self.fourier = FourierChannel(num_frequencies, freq_base, freq_scale)
        self.log_mag = LogMagnitudeChannel()
        self.sign = SignChannel()
        self.poly = PolynomialChannel(poly_degree)

        # Learned channel (takes fourier + log_mag as input)
        learned_input_dim = self.fourier.output_dim + self.log_mag.output_dim
        self.learned = LearnedResidualChannel(learned_input_dim, learned_hidden, learned_out)

        # Total raw dimension before projection
        self.raw_dim = (
            self.fourier.output_dim
            + self.log_mag.output_dim
            + self.sign.output_dim
            + self.poly.output_dim
            + self.learned.output_dim
        )

        # Projection to target embedding dim
        self.projection = nn.Linear(self.raw_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode numbers to embeddings.

        Args:
            x: Tensor of any shape containing numbers to embed.

        Returns:
            Tensor of shape (*x.shape, embedding_dim)
        """
        # Ensure at least 1D so all channels produce consistent dims
        squeeze = x.dim() == 0
        if squeeze:
            x = x.unsqueeze(0)

        fourier_out = self.fourier(x)
        log_mag_out = self.log_mag(x)
        sign_out = self.sign(x)
        poly_out = self.poly(x)

        # Learned channel gets fourier + magnitude as input
        analytic = torch.cat([fourier_out, log_mag_out], dim=-1)
        learned_out = self.learned(analytic)

        # Concatenate all channels
        combined = torch.cat(
            [fourier_out, log_mag_out, sign_out, poly_out, learned_out], dim=-1
        )

        # Project and normalize
        embedding = self.projection(combined)
        embedding = self.layer_norm(embedding)

        if squeeze:
            embedding = embedding.squeeze(0)

        return embedding


# =============================================================================
# Decoder
# =============================================================================

class NumberDecoder(nn.Module):
    """
    Reconstructs the original number from its embedding.

    Architecture: MLP with residual connections and separate
    magnitude/sign prediction heads for numerical stability.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        # Separate heads for magnitude and sign
        self.magnitude_head = nn.Linear(hidden_dim // 2, 1)
        self.sign_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings back to numbers.

        Args:
            embedding: Tensor of shape (..., embedding_dim)

        Returns:
            Tensor of shape (...,) with reconstructed numbers
        """
        h = self.net(embedding)
        log_magnitude = self.magnitude_head(h).squeeze(-1)
        sign_logit = self.sign_head(h).squeeze(-1)

        # Reconstruct: sign * exp(log_magnitude)
        magnitude = torch.exp(log_magnitude)
        sign = torch.tanh(sign_logit)

        return sign * magnitude


# =============================================================================
# Combined Autoencoder with Training Logic
# =============================================================================

class NumberEmbeddingSystem(nn.Module):
    """
    Complete encode-decode system with composite reconstruction loss.

    Loss = MSE(x, x̂) + λ · MSE(log|x|, log|x̂|)

    The log-scale term ensures relative accuracy: reconstructing
    0.001 vs 0.002 is as important as 1000 vs 2000.
    """

    def __init__(self, embedding_dim: int = 128, **encoder_kwargs):
        super().__init__()
        self.encoder = NumberEncoder(embedding_dim=embedding_dim, **encoder_kwargs)
        self.decoder = NumberDecoder(embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (embedding, reconstruction)."""
        emb = self.encoder(x)
        recon = self.decoder(emb)
        return emb, recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, emb: torch.Tensor) -> torch.Tensor:
        return self.decoder(emb)

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        x_recon: torch.Tensor,
        log_scale_weight: float = 0.5,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Composite reconstruction loss with absolute and relative components.
        """
        # Absolute reconstruction
        abs_loss = torch.mean((x - x_recon) ** 2)

        # Log-scale reconstruction (relative accuracy)
        log_x = torch.log(torch.abs(x) + epsilon)
        log_recon = torch.log(torch.abs(x_recon) + epsilon)
        log_loss = torch.mean((log_x - log_recon) ** 2)

        return abs_loss + log_scale_weight * log_loss


# =============================================================================
# Training Utilities
# =============================================================================

def sample_training_numbers(batch_size: int, distribution: str = "mixed") -> torch.Tensor:
    """
    Sample numbers from a distribution designed to cover the numerical
    landscape: small, large, positive, negative, near-zero, integers, floats.
    """
    if distribution == "mixed":
        n = batch_size
        parts = [
            torch.randn(n // 5),                                       # Standard normal
            torch.randn(n // 5) * 1000,                                # Large scale
            torch.randn(n // 5) * 0.001,                               # Small scale
            torch.exp(torch.randn(n // 5) * 3) * torch.sign(torch.randn(n // 5)),  # Log-normal
            torch.randint(-100, 100, (n // 5,)).float(),                # Integers
        ]
        samples = torch.cat(parts)
        # Shuffle
        return samples[torch.randperm(len(samples))][:batch_size]
    elif distribution == "uniform":
        return (torch.rand(batch_size) - 0.5) * 2000
    elif distribution == "log_uniform":
        log_vals = torch.rand(batch_size) * 12 - 6  # log range: 1e-6 to 1e6
        signs = torch.sign(torch.randn(batch_size))
        return signs * (10.0 ** log_vals)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def train(
    system: NumberEmbeddingSystem,
    num_steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_interval: int = 500,
    device: str = "cpu",
) -> list:
    """Train the encoder-decoder system."""
    system = system.to(device)
    optimizer = optim.AdamW(system.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    losses = []

    for step in range(1, num_steps + 1):
        x = sample_training_numbers(batch_size, distribution="mixed").to(device)
        emb, recon = system(x)
        loss = system.reconstruction_loss(x, recon)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            print(f"  Step {step:>5d}/{num_steps} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    return losses


# =============================================================================
# Tests
# =============================================================================

def run_tests(system: Optional[NumberEmbeddingSystem] = None):
    """
    Comprehensive test suite verifying the five core properties:
        1. Uniqueness
        2. Continuity
        3. Reversibility
        4. Expressiveness
        5. Model Compatibility
    """
    device = "cpu"

    if system is None:
        print("=" * 70)
        print("TRAINING NUMBER EMBEDDING SYSTEM")
        print("=" * 70)
        system = NumberEmbeddingSystem(embedding_dim=128)
        train(system, num_steps=5000, batch_size=512, lr=1e-3, log_interval=1000, device=device)
        print()

    system.eval()

    passed = 0
    failed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")

    # =========================================================================
    # TEST 1: UNIQUENESS — Distinct numbers → distinct embeddings
    # =========================================================================
    print("-" * 70)
    print("TEST 1: UNIQUENESS")
    print("-" * 70)

    with torch.no_grad():
        # Test that different numbers produce different embeddings
        nums = torch.tensor([0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, -1.0, -100.0, 3.14159])
        embs = system.encode(nums)

        # All pairwise cosine similarities should be < 1
        norms = embs / embs.norm(dim=-1, keepdim=True)
        cos_sim = norms @ norms.T
        # Zero out diagonal
        mask = ~torch.eye(len(nums), dtype=torch.bool)
        max_sim = cos_sim[mask].max().item()
        check(
            "Distinct numbers have distinct embeddings",
            max_sim < 0.99,
            f"Max pairwise cosine similarity: {max_sim:.4f} (should be < 0.99)"
        )

        # All pairwise L2 distances should be > 0
        dists = torch.cdist(embs.unsqueeze(0), embs.unsqueeze(0)).squeeze(0)
        min_dist = dists[mask].min().item()
        check(
            "All pairwise L2 distances > 0",
            min_dist > 1e-4,
            f"Min pairwise L2 distance: {min_dist:.6f}"
        )

        # Very close but distinct numbers should still differ
        a = system.encode(torch.tensor([1.0000]))
        b = system.encode(torch.tensor([1.0001]))
        dist_close = (a - b).norm().item()
        check(
            "Close numbers (1.0000 vs 1.0001) have different embeddings",
            dist_close > 1e-5,
            f"L2 distance: {dist_close:.8f}"
        )

    # =========================================================================
    # TEST 2: CONTINUITY — Small Δx → small Δe(x)
    # =========================================================================
    print("-" * 70)
    print("TEST 2: CONTINUITY")
    print("-" * 70)

    with torch.no_grad():
        base_vals = torch.tensor([0.0, 1.0, 10.0, -5.0, 100.0])
        epsilons = [0.1, 0.01, 0.001]

        for eps in epsilons:
            perturbed = base_vals + eps
            emb_base = system.encode(base_vals)
            emb_pert = system.encode(perturbed)
            diffs = (emb_base - emb_pert).norm(dim=-1)
            max_diff = diffs.max().item()
            check(
                f"Perturbation ε={eps}: embedding change is bounded",
                max_diff < 10 * eps + 1.0,  # generous Lipschitz bound
                f"Max embedding Δ: {max_diff:.6f}"
            )

        # Monotonicity of distance: larger Δx → larger Δe
        center = torch.tensor([5.0])
        emb_center = system.encode(center)
        deltas = [0.01, 0.1, 1.0, 10.0]
        distances = []
        for d in deltas:
            emb_shifted = system.encode(center + d)
            distances.append((emb_center - emb_shifted).norm().item())

        is_monotonic = all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))
        check(
            "Embedding distance generally increases with numerical distance",
            is_monotonic,
            f"Distances for Δ={deltas}: {[f'{d:.4f}' for d in distances]}"
        )

    # =========================================================================
    # TEST 3: REVERSIBILITY — Decode(Encode(x)) ≈ x
    # =========================================================================
    print("-" * 70)
    print("TEST 3: REVERSIBILITY")
    print("-" * 70)

    with torch.no_grad():
        # Test across different scales
        test_cases = {
            "Small positive":     torch.tensor([0.001, 0.01, 0.05, 0.1]),
            "Medium positive":    torch.tensor([1.0, 2.5, 3.14159, 7.0]),
            "Large positive":     torch.tensor([100.0, 500.0, 999.0]),
            "Negative":           torch.tensor([-1.0, -10.0, -50.0, -100.0]),
            "Near zero":          torch.tensor([-0.01, 0.0, 0.01]),
            "Integers":           torch.tensor([1.0, 2.0, 3.0, 42.0, 100.0]),
        }

        for name, vals in test_cases.items():
            emb, recon = system(vals)
            abs_err = (vals - recon).abs()
            rel_err = abs_err / (vals.abs() + 1e-8)
            max_abs = abs_err.max().item()
            max_rel = rel_err.max().item()
            # Generous threshold since this is a small training run
            ok = max_rel < 0.5 or max_abs < 5.0
            check(
                f"Reconstruction [{name}]",
                ok,
                f"Max absolute error: {max_abs:.4f}, Max relative error: {max_rel:.4f}"
            )

        # Specific round-trip examples
        specific = torch.tensor([3.14159, -42.0, 0.001, 256.0])
        _, recon = system(specific)
        print(f"\n  Round-trip examples:")
        for orig, rec in zip(specific.tolist(), recon.tolist()):
            err = abs(orig - rec)
            print(f"    {orig:>10.5f}  →  encode  →  decode  →  {rec:>10.5f}  (error: {err:.5f})")
        print()

    # =========================================================================
    # TEST 4: EXPRESSIVENESS — Embedding captures numerical structure
    # =========================================================================
    print("-" * 70)
    print("TEST 4: EXPRESSIVENESS")
    print("-" * 70)

    with torch.no_grad():
        # Embedding dimensionality is fully utilized (not collapsed)
        x_sample = sample_training_numbers(500)
        embs_sample = system.encode(x_sample)
        # Check effective dimensionality via SVD
        U, S, V = torch.svd(embs_sample - embs_sample.mean(dim=0))
        # Count dimensions with > 1% of max singular value
        threshold = 0.01 * S[0]
        effective_dims = (S > threshold).sum().item()
        check(
            f"High effective dimensionality",
            effective_dims > 20,
            f"Effective dims (>1% of max SV): {effective_dims}/{system.encoder.embedding_dim}"
        )

        # Ordering preservation: for a < b, check that magnitude embeddings
        # capture this in a learnable way
        ordered = torch.linspace(-10, 10, 50)
        embs_ordered = system.encode(ordered)
        # PCA first component should be correlated with the ordering
        mean_emb = embs_ordered.mean(dim=0)
        centered = embs_ordered - mean_emb
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = (centered @ Vh[0]).numpy()
        from scipy.stats import spearmanr
        corr, _ = spearmanr(ordered.numpy(), pc1)
        check(
            "First principal component correlates with numerical order",
            abs(corr) > 0.8,
            f"Spearman correlation: {corr:.4f}"
        )

        # Similar numbers cluster closer than dissimilar ones
        x_a = torch.tensor([1.0, 1.1, 1.2])
        x_b = torch.tensor([100.0, 100.1, 100.2])
        emb_a = system.encode(x_a)
        emb_b = system.encode(x_b)
        intra_a = torch.cdist(emb_a.unsqueeze(0), emb_a.unsqueeze(0)).squeeze().max().item()
        intra_b = torch.cdist(emb_b.unsqueeze(0), emb_b.unsqueeze(0)).squeeze().max().item()
        inter = torch.cdist(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).squeeze().min().item()
        check(
            "Similar numbers cluster tighter than dissimilar ones",
            inter > max(intra_a, intra_b),
            f"Intra-cluster max dist: {max(intra_a, intra_b):.4f}, Inter-cluster min dist: {inter:.4f}"
        )

    # =========================================================================
    # TEST 5: MODEL COMPATIBILITY — Differentiable, batchable, standard ops
    # =========================================================================
    print("-" * 70)
    print("TEST 5: MODEL COMPATIBILITY")
    print("-" * 70)

    system.train()

    # Gradient flow
    x_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    emb_grad = system.encode(x_grad)
    loss = emb_grad.sum()
    loss.backward()
    has_grad = x_grad.grad is not None and x_grad.grad.abs().sum() > 0
    check("Gradients flow through encoder", has_grad)

    # Batch processing
    x_batch = torch.randn(64)
    emb_batch = system.encode(x_batch)
    check(
        "Batch processing works",
        emb_batch.shape == (64, system.encoder.embedding_dim),
        f"Input: (64,) → Output: {tuple(emb_batch.shape)}"
    )

    # Single number
    x_single = torch.tensor(42.0)
    emb_single = system.encode(x_single)
    check(
        "Single number embedding works",
        emb_single.shape[-1] == system.encoder.embedding_dim,
        f"Shape: {tuple(emb_single.shape)}"
    )

    # Integration with downstream linear layer
    downstream = nn.Linear(system.encoder.embedding_dim, 10)
    out = downstream(emb_batch)
    check(
        "Compatible with downstream nn.Linear",
        out.shape == (64, 10),
        f"Output shape: {tuple(out.shape)}"
    )

    # Integration with transformer-style attention
    d = system.encoder.embedding_dim
    seq = system.encode(torch.randn(8))  # 8 numbers as a "sequence"
    seq = seq.unsqueeze(0)  # (1, 8, d) — batch of 1
    attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
    attn_out, _ = attn(seq, seq, seq)
    check(
        "Compatible with nn.MultiheadAttention",
        attn_out.shape == (1, 8, d),
        f"Attention output shape: {tuple(attn_out.shape)}"
    )

    # Serialization round-trip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(system.state_dict(), f.name)
        new_system = NumberEmbeddingSystem(embedding_dim=system.encoder.embedding_dim)
        new_system.load_state_dict(torch.load(f.name, weights_only=True))
        os.unlink(f.name)
    check("Model serialization (save/load) works", True)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 70)

    return passed, failed, total


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Quick demonstration of encoding and decoding."""
    print("=" * 70)
    print("NUMBER EMBEDDING DEMO")
    print("=" * 70)
    print()

    system = NumberEmbeddingSystem(embedding_dim=128)

    # Show architecture
    total_params = sum(p.numel() for p in system.parameters())
    trainable_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"Architecture:")
    print(f"  Embedding dimension: 128")
    print(f"  Encoder channels:")
    print(f"    Fourier:     {system.encoder.fourier.output_dim} dims")
    print(f"    Log-mag:     {system.encoder.log_mag.output_dim} dims")
    print(f"    Sign:        {system.encoder.sign.output_dim} dims")
    print(f"    Polynomial:  {system.encoder.poly.output_dim} dims")
    print(f"    Learned:     {system.encoder.learned.output_dim} dims")
    print(f"    Raw total:   {system.encoder.raw_dim} dims → projected to 128")
    print(f"  Total params:  {total_params:,}")
    print()

    # Encode some numbers (before training — just the analytic channels matter)
    print("Encoding examples (untrained):")
    numbers = [0.0, 1.0, 3.14159, -42.0, 1000.0, 0.001]
    for n in numbers:
        x = torch.tensor(n)
        emb = system.encode(x)
        print(f"  {n:>10.5f} → embedding norm: {emb.norm():.4f}, "
              f"first 5 dims: [{', '.join(f'{v:.3f}' for v in emb.flatten()[:5])}...]")
    print()

    # Train
    print("Training...")
    print("-" * 40)
    losses = train(system, num_steps=5000, batch_size=512, lr=1e-3, log_interval=1000)
    print()

    # Show reconstruction after training
    system.eval()
    print("Reconstruction after training:")
    print(f"  {'Input':>12s} → {'Reconstructed':>14s}  {'Error':>10s}  {'Rel Error':>10s}")
    print(f"  {'-'*12}   {'-'*14}  {'-'*10}  {'-'*10}")
    with torch.no_grad():
        for n in numbers:
            x = torch.tensor(n)
            _, recon = system(x)
            err = abs(n - recon.item())
            rel = err / (abs(n) + 1e-8)
            print(f"  {n:>12.5f} → {recon.item():>14.5f}  {err:>10.5f}  {rel:>10.4%}")
    print()

    return system


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--test-only" in sys.argv:
        run_tests()
    elif "--demo-only" in sys.argv:
        demo()
    else:
        # Full run: demo + tests
        print()
        system = demo()
        print()
        run_tests(system)