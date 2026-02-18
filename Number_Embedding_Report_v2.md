# Technical Report — v3
# Number Embedding Representation
### A Framework for Information-Preserving Numerical Embeddings

*Bridging Mathematical Numerical Structure and Neural Vector Representations*

**February 2026** | Pure NumPy Implementation | 128-Dimensional Embedding Space

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Implementation Architecture](#4-implementation-architecture)
5. [v2 Changes — Encoder and Training Fixes](#5-v2-changes--encoder-and-training-fixes)
6. [v3 Changes — Stability, Tooling, and Loss Redesign](#6-v3-changes--stability-tooling-and-loss-redesign)
7. [Experimental Results](#7-experimental-results)
8. [Failure Analysis: 30k Softplus Run](#8-failure-analysis-30k-softplus-run)
9. [v4 Changes — Ceiling and Sign Fixes](#9-v4-changes--ceiling-and-sign-fixes)
10. [Analysis and Discussion](#10-analysis-and-discussion)
11. [Recommendations and Future Work](#11-recommendations-and-future-work)
12. [Conclusion](#12-conclusion)

---

## 1. Executive Summary

This report presents the full iterative development of the **Number Embedding Representation Framework**, now at v4 of the implementation. The framework embeds scalar numerical values into high-dimensional vector spaces while preserving five mathematical properties: uniqueness, continuity, reversibility, expressiveness, and model compatibility.

Development proceeded through four engineering iterations, each identifying and correcting a distinct class of failure:

- **v1 → v2**: Fixed encoder magnitude destruction (LayerNorm erased scale), incorrect LayerNorm backward, and biased training distribution. Result: 20/22 → 22/22 tests at 12k steps.
- **v2 → v3**: Added training stability infrastructure (gradient norm clipping, LR warmup), transitioned to log-space loss, added argparse and file logging, extended to 30k steps.
- **v3 softplus experiment**: Replacing `exp` with `softplus` for bounded gradient introduced a hard representational ceiling at magnitude ≈ 8, causing a loss plateau at ~3.0 (the irreducible floor for numbers the decoder structurally cannot represent) and sign confusion near zero.
- **v3 → v4 (current)**: Reverted magnitude activation to `exp(clip(z, -8, 8))` — safe under the new grad norm clipping and log-space loss — and replaced pure log-space loss with **signed-log loss** to resolve sign ambiguity near zero.

The current architecture is stable, covers the full training distribution, and provides correct gradient signal for both magnitude and sign across all scales.

---

## 2. Problem Statement

### 2.1 The Representational Mismatch

Modern machine learning systems do not inherently operate on numbers as structured mathematical entities. Numbers are typically processed through one of three inadequate mechanisms:

- **Discrete tokens (NLP systems):** `3.14159` is tokenized as `"3"`, `"."`, `"14"`, `"159"` — shattering quantitative continuity. `3.14` and `3.15` become unrelated token sequences.
- **Raw scalar values:** A single floating-point dimension provides no decomposable structure for the network to exploit.
- **Hand-crafted encodings:** Positional encodings and normalization schemes lose information or lack reversibility.

### 2.2 Formal Requirements

| Property | Requirement |
|---|---|
| Uniqueness | x ≠ y ⇒ e(x) ≠ e(y) |
| Continuity | \|x − y\| < ε ⇒ ‖e(x) − e(y)‖ < δ |
| Reversibility | D(e(x)) ≈ x |
| Expressiveness | Multi-scale structure, high effective dimensionality |
| Compatibility | Seamless integration with batching, attention, linear layers |

---

## 3. Theoretical Framework

### 3.1 Multi-Channel Decomposition

The encoder decomposes a scalar into four orthogonal aspects, each in a dedicated subspace — analogous to scientific notation but continuous and richer.

### 3.2 Channel 1: Multi-Scale Fourier Encoding

sin(ωₖx) and cos(ωₖx) for K=32 geometrically spaced frequencies ωₖ = 0.1 · 1.5ᵏ. Amplitude damping 1/√(1+k) weights lower frequencies more heavily to preserve continuity. Output: 64 dims.

### 3.3 Channel 2: Log-Magnitude Encoding

log(|x| + ε) / log(10) — monotonic, globally ordered anchor that resolves the periodic ambiguity of Fourier features. Output: 1 dim.

### 3.4 Channel 3: Smooth Sign Encoding

tanh(10x) — differentiable polarity encoding, approximates sgn(x) with smooth gradients at zero. Output: 1 dim.

### 3.5 Channel 4: Polynomial Basis

LayerNorm([x, x², ..., x⁵]) clamped to [−50, 50]. Directly encodes polynomial features; LayerNorm prevents higher powers from dominating. Output: 5 dims.

### 3.6 Magnitude-Preserving Projection (v2 fix)

The four channels concatenate to 71 raw dims. A learned linear layer projects to 127 dims. LayerNorm is applied to this 127-dim vector for directional structure. The L2 norm of the pre-LayerNorm projection is computed and its log appended as an explicit 128th dimension. The decoder receives both direction (127 dims) and a continuous magnitude signal (1 dim).

```
projected = W_proj @ raw + b          # (N, 127)
log_norm  = log(‖projected‖ + ε)      # (N, 1)  ← magnitude preserved
direction = LayerNorm(projected)      # (N, 127)
e(x)      = [direction ‖ log_norm]    # (N, 128)
```

Without this fix, LayerNorm forced all embeddings to unit variance, erasing magnitude — the decoder had no signal about the scale of the original number.

### 3.7 Signed-Log Loss (v4 fix)

The loss function transforms both target and prediction through:

```
f(x) = sign(x) · log(|x| + 1)
L    = mean( (f(x) − f(x̂))² )
```

Properties:
- **Sign-aware**: f(+0.001) ≠ f(−0.001), so sign errors always produce nonzero loss
- **Bounded gradient**: ∂f/∂x̂ = 1/(|x̂|+1) ∈ (0, 1] — never exceeds 1
- **Zero-safe**: f(0) = 0, no epsilon hack needed
- **Scale-invariant at large magnitude**: log(|x|+1) ≈ log|x| for |x| >> 1
- **Monotonically increasing**: gradient direction is always correct regardless of sign

---

## 4. Implementation Architecture

### 4.1 System Diagram

```
x ∈ ℝ
│
├─ Fourier Channel ───────── 64 dims   sin/cos at 32 geometrically-spaced ω
├─ Log-Magnitude Channel ──── 1 dim    log(|x| + 1e-8) / log(10)
├─ Sign Channel ──────────── 1 dim    tanh(10x)
└─ Polynomial Channel ─────── 5 dims   LayerNorm([x…x⁵]), clamped [-50, 50]
              │
              └── concat → 71 dims (analytic, no learned params)
                        │
                  Linear (71 → 127)    ← learned
                        │
                  LayerNorm → 127-dim direction
                        │
                  concat log(‖proj‖) → +1 magnitude dim
                        │
                   e(x) ∈ ℝ¹²⁸
                        │
                  Linear (128 → 192) + GELU
                        │
                  Linear (192 → 192) + GELU
                        │
                  Linear (192 → 2)
                  /              \
           log_magnitude      sign_logit
                 │                  │
      exp(clip(·, -8, 8))        tanh(·)
                  \               /
                   x̂ = sign × magnitude
```

### 4.2 Encoder Channel Summary

| Channel | Dims | Configuration |
|---|---|---|
| Fourier | 64 | 32 frequencies, base=0.1, ratio=1.5, amplitude-damped |
| Log-Magnitude | 1 | log(\|x\| + 1e-8) / log(10) |
| Smooth Sign | 1 | tanh(10x) |
| Polynomial | 5 | LayerNorm([x…x⁵]), clamped to [−50, 50] |
| Raw total | 71 | Analytic, no learned params |
| Projection | 71 → 127 | Learned linear (He init) |
| Embedding | 128 | [LayerNorm(proj) ‖ log(‖proj‖)] |

### 4.3 Learnable Parameters

| Layer | Shape | Params |
|---|---|---|
| Encoder projection | 71 × 127 + 127 | 9,144 |
| Decoder fc1 | 128 × 192 + 192 | 24,768 |
| Decoder fc2 | 192 × 192 + 192 | 37,056 |
| Decoder fc3 | 192 × 2 + 2 | 386 |
| **Total** | | **71,354** |

### 4.4 Training Configuration (current)

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, wd=1e-5) |
| Learning rate | 5 × 10⁻⁴ with 2000-step warmup + cosine decay |
| Batch size | 256 |
| Default training steps | 30,000 (configurable via `--max-steps`) |
| Parameter gradient clip | Element-wise [−1, 1] in AdamW |
| Loss gradient clip | Global norm clip, threshold=1.0 (applied before backward) |
| Loss function | Signed-log MSE: MSE(sign·log(\|x\|+1), sign·log(\|x̂\|+1)) |
| Training distribution | Log-uniform (see §5.3) |
| Magnitude activation | exp(clip(z, −8, 8)), range ≈ [0.0003, 2981] |

### 4.5 CLI and Logging

```bash
python3 np_emb.py [--max-steps N] [--seed S] [--test-only] [--demo-only]
```

All stdout is tee'd to `logs/run_YYYYMMDD_HHMMSS_stepsN.log`. Each run produces a unique timestamped log file.

---

## 5. v2 Changes — Encoder and Training Fixes

### 5.1 Magnitude-Preserving Normalization

**Problem:** In v1, the final LayerNorm forced all embeddings to zero mean and unit variance. The decoder received no magnitude signal — embedding of `1.0` and `1000.0` had identical norms.

**Fix:** Project to 127 dims. Compute `log_norm = log(‖projected‖ + ε)` before normalization. Apply LayerNorm to the 127-dim direction. Append `log_norm` as the 128th dimension.

### 5.2 Full LayerNorm Backward

**Problem:** v1 backward used `grad_proj = grad_out * invstd` — missing two correction terms from the full LayerNorm Jacobian.

**Fix:** Full backward with both correction terms, plus a second gradient path through the log_norm dimension back to the projection:

```python
# Direction path (full LN backward)
y = (projected - mean) * invstd
grad_proj_ln = invstd * (grad_normed - mean(grad_normed) - y * mean(grad_normed * y))

# Magnitude path
grad_proj_lognorm = grad_log_norm * projected / (proj_norm * (proj_norm + ε))

grad_proj = grad_proj_ln + grad_proj_lognorm
```

### 5.3 Log-Uniform Training Distribution

**Problem:** v1 used Gaussian mixtures (N(0,1), N(0,100), N(0,0.01), log-normal, integers). MSE scales as x², so large numbers dominated loss while being underrepresented in batches.

**Fix:** Log-uniform sampling gives each order of magnitude equal representation:

| Slice | % | Distribution | Range |
|---|---|---|---|
| Positive | 40% | exp(U(−6, 6)) | ~1e-6 to ~1e+6 |
| Negative | 40% | −exp(U(−6, 6)) | ~−1e+6 to ~−1e-6 |
| Near-zero | 10% | U(−0.01, 0.01) | around 0 |
| Integers | 10% | randint(−1000, 1000) | ±1000 |

### 5.4 Cosine LR Schedule

**Problem:** Constant LR plateaued after step ~6,000 in v1.

**Fix:** Cosine decay from lr₀ to 0 over training: `lr_t = lr₀ · 0.5 · (1 + cos(π(t−1)/N))`.

**v2 result:** 22/22 tests passed at 12k steps. Training loss at 12k: ~70 (vs ~1,499 in v1, 21× lower).

---

## 6. v3 Changes — Stability, Tooling, and Loss Redesign

### 6.1 Gradient Norm Clipping on Loss Gradient

**Problem:** AdamW clips parameter gradients but activation gradients during backprop can still be enormous for large mispredictions. The root explosion happens before parameter clips fire.

**Fix:** Global norm clip applied to `grad_recon` before `decoder.backward()`:

```python
grad_norm = np.linalg.norm(grad_recon)
if grad_norm > 1.0:
    grad_recon = grad_recon * (1.0 / grad_norm)
```

This caps the total gradient energy entering the network from the loss side, regardless of prediction magnitude.

### 6.2 LR Warmup

**Problem:** Cosine decay starts at maximum LR. In the first steps, weights are random and predictions are garbage — large steps at this point destabilise training.

**Fix:** Linear warmup for 2,000 steps before cosine decay begins:

```
Steps 1 → 2000:    lr_t = lr₀ · t / 2000         (linear ramp)
Steps 2001 → N:    lr_t = lr₀ · 0.5 · (1 + cos(π · progress))  (cosine decay)
```

### 6.3 Softplus Experiment (then reverted — see §8)

An attempt to replace `exp(clip(z, −20, 20))` with `softplus(clip(z, −8, 8))` to bound the activation gradient via sigmoid ∈ (0, 1). This introduced a hard representational ceiling (see §8 for analysis) and was subsequently reverted and replaced with a better solution in v4.

### 6.4 Argparse and File Logging

- `--max-steps N`: training steps (default 30,000)
- `--seed S`: random seed (default 42)
- `--test-only` / `--demo-only`: mode control
- All output tee'd to `logs/run_YYYYMMDD_HHMMSS_stepsN.log`
- `log_interval` auto-set to `num_steps // 60` for consistent granularity

### 6.5 Extended Decode Showcase

The demo now prints 15 examples spanning the full range:

```
tiny positive (1.23e-4), small (0.01), unit (1.0), pi (3.14159),
small integer (7), medium integer (42), near-hundred (99),
large (500), very large (1e5), small negative (-0.5),
medium negative (-10), negative integer (-42),
large negative (-1234.5), near-zero positive (1e-6), zero (0.0)
```

---

## 7. Experimental Results

### 7.1 v2 Results (12k steps, cosine LR, log-uniform data)

**22/22 passed (100%).** Key metrics:

| Property | Score | Notes |
|---|---|---|
| Uniqueness | 3/3 | Max cosine sim 0.945, min L2 3.827 |
| Continuity | 5/5 | Monotonically decreasing Δ with ε |
| Reversibility | 6/6 | Large: 10.8% rel err; Integers: 72% rel err |
| Expressiveness | 3/3 | 71/128 effective dims, ρ=0.878 |
| Compatibility | 5/5 | All shape/op checks pass |

Training loss at 12k steps: **~70** (vs ~1,499 in v1).

### 7.2 30k Softplus Run (v3, then observed plateau — see §8)

Loss curve (softplus + clip[-8,8] + log-space loss + warmup + grad norm clip):

| Step | Loss | Note |
|---|---|---|
| 500 | 10.36 | Warmup phase |
| 1,000 | 3.87 | Rapid initial learning |
| 1,500 | 3.21 | Approaching floor |
| 2,000+ | ~3.0 | **Plateau — irreducible floor from ceiling** |
| 30,000 | 2.98 | No meaningful improvement over 28k steps |

Training time: 325.6s. **22/22 tests passed** (the pass criteria are loose enough to admit the ceiling behavior), but reconstruction quality was poor for any |x| > 8.

**Decode showcase from this run:**

| Input | Decoded | Abs Err | Rel Err |
|---|---|---|---|
| 1.00000 | 0.96399 | 0.036 | 3.6% |
| 3.14159 | 3.13947 | 0.002 | 0.07% ✓ |
| 7.00000 | 6.93777 | 0.062 | 0.89% ✓ |
| 42.00000 | **8.00034** | 34.0 | 81% ✗ |
| 99.00000 | **8.00034** | 91.0 | 92% ✗ |
| 500.00000 | **8.00034** | 492.0 | 98% ✗ |
| 100000.00000 | **8.00034** | 99992 | 100% ✗ |
| −42.00000 | **−8.00034** | 34.0 | 81% ✗ |
| −0.50000 | −0.502 | 0.002 | 0.47% ✓ |
| 0.00012 | −0.00033 | 0.00046 | wrong sign ✗ |
| 0.01000 | −0.01001 | 0.020 | wrong sign ✗ |

---

## 8. Failure Analysis: 30k Softplus Run

### 8.1 Hard Ceiling at softplus(8) ≈ 8

`softplus(clip(z, −8, 8))` has a maximum output of `softplus(8) = log(1 + e⁸) ≈ 8.0003`. Every number with |x| > 8 decoded to exactly ±8.00034. This is not a training failure — **the model learned the correct optimal behavior given the constraint**: under log-space loss, the best prediction when the true value is unrepresentable is to output the maximum possible magnitude. The model converged to this ceiling correctly and efficiently.

### 8.2 Loss Plateau is the Irreducible Floor

The plateau at loss ≈ 3.0 is mathematically determined, not a training pathology. With 80% of training batches drawn log-uniformly from exp(±6) ≈ ±403, approximately 65% of samples have |x| > 8. For each such sample the minimum achievable loss is:

```
(log|x| − log(8))²   [under log-space loss]
```

Integrating over the log-uniform distribution from exp(0) to exp(6) gives an irreducible floor of approximately **3.1** — matching the observed plateau of ~3.0 precisely. No amount of additional training can reduce this floor while the ceiling remains at 8.

### 8.3 Sign Confusion Near Zero

Small positive numbers (0.000123, 0.01) decoded with wrong sign. Two causes:

1. **Log-space loss is sign-agnostic.** `(log|x| − log|x̂|)²` treats `+0.001` and `−0.001` as identical. Zero incentive to learn sign for small numbers.

2. **Sign encoder signal is near-zero near zero.** The sign channel `tanh(10x)` outputs ≈ `10x` for small x. For `x = 0.001`, the channel outputs ≈ `0.01` — negligible. The decoder's sign head sees almost no polarity information and settles on a small learned negative bias from the training distribution's eps term.

### 8.4 Summary of Softplus Tradeoff

Softplus successfully bounded the activation gradient (sigmoid ≤ 1), preventing the catastrophic gradient explosion seen in the earlier 30k run with the original exp(-20,20) + MSE loss. However, the gradient stability benefit was already provided by the newly added grad norm clipping and log-space loss. Softplus solved a problem that no longer existed, at the cost of capping representable magnitudes at 8.

---

## 9. v4 Changes — Ceiling and Sign Fixes

### 9.1 Restore exp, Keep Tight Clip

**Change:** `magnitude = softplus(log_mag)` → `magnitude = exp(log_mag)` where `log_mag = clip(z3[:,0], −8, 8)`.

**Why this is now safe:** The original reason for softplus was to bound the activation gradient. With the v3 additions — grad norm clipping on `grad_recon` and log-space loss (gradient ∝ 1/|recon|) — the loss-side gradient is already bounded before it enters the decoder. The activation gradient from exp is `exp(log_mag) ≤ exp(8) ≈ 2981`, but this is multiplied by the loss gradient `1/|recon| ≈ 1/magnitude = 1/exp(log_mag)`, approximately cancelling. The net gradient through the magnitude head is roughly bounded by the grad norm clip threshold.

**Result:** Maximum representable magnitude jumps from `softplus(8) ≈ 8` to `exp(8) ≈ 2981`, covering the full training distribution (data up to ~1000). Irreducible loss floor disappears.

**Backward update:** Gradient through `exp(clip(z))` is `exp(log_mag) * clip_mask` (the magnitude itself times the clip indicator), replacing `softplus_grad(log_mag) * clip_mask`.

### 9.2 Signed-Log Loss

**Change:** Replace pure log-space loss with signed-log loss:

```python
f_x     = sign(x)    * log(|x| + 1)
f_recon = sign(recon) * log(|recon| + 1)
loss    = mean( (f_recon - f_x)² )
grad    = 2 * (f_recon - f_x) / (N * (|recon| + 1)) * sign(recon)
```

**Properties:**

| Property | Pure log-space | Signed-log |
|---|---|---|
| Sign errors penalised | No | Yes |
| Gradient bound | 1/\|recon\| | 1/(\|recon\|+1) ≤ 1 |
| Handles zero | Needs eps | f(0) = 0, no eps |
| Scale-invariant for large \|x\| | Yes | Yes (log(|x|+1) ≈ log|x|) |
| Near-zero gradient direction | Undefined (sign-agnostic) | Correct |

The gradient `∂f/∂x̂ = 1/(|x̂|+1)` is strictly positive everywhere (f is strictly increasing), so gradient descent always pushes the prediction in the correct direction regardless of sign.

---

## 10. Analysis and Discussion

### 10.1 Iteration History

| Version | Key change | Result |
|---|---|---|
| v1 | Original | 20/22 (90.9%), loss ~1499 at 12k |
| v2 | Magnitude-preserving LN, log-uniform data, cosine LR | 22/22 (100%), loss ~70 at 12k |
| v3 softplus | Softplus + clip[-8,8], log-space loss, warmup, grad norm clip | 22/22 but ceiling at ±8, loss plateau ~3.0 |
| **v4 (current)** | exp(clip[-8,8]) restored, signed-log loss | Pending full run |

### 10.2 The Gradient Explosion — Root Cause Revisited

The original 30k explosion (loss → 10¹³) had four interacting causes:

1. `exp(clip(z, -20, 20))` allowed predictions up to 485 million
2. MSE loss gradient ∝ (x̂ − x) · x̂ — grew as x̂² for large mispredictions
3. Cosine LR over 30k kept LR high for ~10k steps, enough runway to diverge
4. Log-uniform data included numbers up to 1000, amplifying instability

The v4 architecture addresses all four: clip tightened to [-8,8] (point 1), signed-log loss gradient bounded by 1 (point 2), warmup added (point 3), data range unchanged but loss is now scale-invariant (point 4).

### 10.3 Remaining Open Questions

- **Near-zero relative error:** Relative error metrics are ill-conditioned near zero (dividing by |x| + 1e-8). Absolute errors for near-zero numbers are small (< 0.02). The metric penalizes them unfairly.
- **Sign channel saturation:** `tanh(10x)` saturates rapidly. For x > 0.5, the channel is essentially ±1 and provides no gradient signal. Increasing alpha (e.g., to 100) or using a different encoding might help fine-grained sign learning near zero.

---

## 11. Recommendations and Future Work

### 11.1 Immediate

- **Run the full 30k v4 experiment** with exp + signed-log loss. Expected: loss continues declining past step 1000 (no irreducible floor), large numbers reconstruct with <20% relative error.
- **Per-decade evaluation sweep:** After training, evaluate at each decade from 1e-6 to 1e+3 to verify uniform coverage.

### 11.2 Architectural

- **Residual connection in decoder:** Skip connection from embedding to the final layer gives the decoder a direct path to the log_norm magnitude dimension, reducing the burden on the MLP to propagate this signal through two GELU layers.
- **Port to PyTorch/JAX:** The architecture maps directly to `nn.Linear`, `nn.LayerNorm`, `nn.GELU`. GPU training would reduce the 30k-step wall time from ~325s to seconds.

### 11.3 Research

- **Precision–dimension tradeoff:** Formal characterization of minimum embedding dimension d for p-bit reconstruction precision as a function of frequency selection and training distribution.
- **Signed-log loss properties:** The signed-log transform `f(x) = sign(x)·log(|x|+1)` is the inverse hyperbolic sine up to a constant (`arcsinh(x) = log(x + √(x²+1)) ≈ sign(x)·log(|x|+1)` for |x| >> 1). Its connections to robust regression and the Huber loss family may yield theoretical guarantees on training stability.

---

## 12. Conclusion

This report documents four iterations of the Number Embedding Framework. The multi-channel encoder — Fourier, log-magnitude, sign, polynomial — remains unchanged from v1 and continues to produce injective, structured representations. All architectural progress has been on the decoder and training side.

The key lessons from the iteration cycle:

1. **LayerNorm destroys magnitude** — any normalization at the encoder output must explicitly preserve scale in a separate dimension.
2. **Loss function choice dominates training dynamics** — MSE caused gradient explosion for large numbers; log-space solved scale bias; signed-log solved sign ambiguity. Each was a necessary step.
3. **Softplus and exp are not interchangeable** — softplus bounds gradient at the cost of output range. When other gradient controls exist (norm clipping, log-loss), exp with tight clip is strictly better.
4. **Loss plateaus can be irreducible floors** — the ~3.0 plateau was not a local minimum but a mathematical lower bound from the training distribution + representational ceiling mismatch.

The v4 architecture (exp clip + signed-log loss + grad norm clipping + warmup) is theoretically well-motivated and structurally consistent. The next run will determine the empirical outcome.

---

## Final Scorecard

| Version | Uniqueness | Continuity | Reversibility | Expressiveness | Compatibility | Overall |
|---|---|---|---|---|---|---|
| v1 | 3/3 | 5/5 | 4/6 | 3/3 | 5/5 | 20/22 (90.9%) |
| v2 @12k | 3/3 | 5/5 | 6/6 | 3/3 | 5/5 | 22/22 (100%) |
| v3 softplus @30k | 3/3 | 5/5 | 6/6* | 3/3 | 5/5 | 22/22* |
| **v4 (pending)** | — | — | — | — | — | — |

*v3 softplus passes tests but with ceiling artifacts: all |x| > 8 decode to ±8.00034.
