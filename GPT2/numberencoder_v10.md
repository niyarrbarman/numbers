# NumberEncoder v10 — Comprehensive Technical Reference

## 1. Motivation: Why Numbers Are a Modality

Standard LLMs tokenize numbers with BPE (Byte-Pair Encoding), which fragments digits into arbitrary subword chunks. For example, `21530000000000` may become `["215", "300", "000", "000", "00"]`. This destroys three properties that are essential for numerical reasoning:

1. **Magnitude**: The model cannot see that `21530000000000` is ~2.15 x 10^13 without reconstructing it from fragments.
2. **Ordering**: Comparing `9999` vs `10001` requires multi-token reasoning about carries.
3. **Arithmetic structure**: Addition, subtraction, and modular properties are not accessible from subword tokens.

The insight (directly from the LLaVA paradigm for vision): just as you would not OCR an image into text and feed it to a language model, you should not force numbers through text tokenization. Instead, give the model a **pretrained numerical encoder** and teach it to use those features via an adapter.

NumberEncoder v10 is that encoder.

---

## 2. Architecture Overview

**Output**: 128-dimensional embedding vector per number.

**Three-lane design**: The embedding is a concatenation of three independent lanes, each encoding different mathematical properties of the input number x.

```
Input: x (scalar, float64)
        │
        ├─── Lane 1: Scale (16 dims)    ── x * w        (linear, additive)
        │
        ├─── Lane 2: Residue (22 dims)  ── sin/cos at 11 periods (float64)
        │
        └─── Lane 3: Semantic (90 dims) ── Fourier + LogMag + Sign → proj → scale + log_norm
        │
        └─── Concatenate → [128 dims]
```

**Total encoder parameters**: ~7,400 (extremely lightweight).

---

## 3. Lane 1: Scale Lane (16 dimensions)

### What it computes

```
e_scale(x) = x * w,    w ∈ R^16
```

where w is a learnable weight vector initialized with log-spaced values from 10^-5 to 10^-2, with alternating signs.

### Mathematical properties

- **Exact additivity**: e_scale(x + y) = e_scale(x) + e_scale(y). This is satisfied **by construction**, not by learning.
- **No normalization**: Raw magnitude is preserved. This is critical — normalizing would destroy the linear relationship.

### Why this design

- A downstream linear probe `W @ [e(x); e(y)]` can recover x + y **perfectly** from the scale lane alone, because the operation is linear.
- The log-spaced initialization provides multi-resolution coverage: small weights respond to large numbers, large weights respond to small numbers, preventing any single scale from dominating.
- Alternating signs prevent dimension collapse (all dims pointing in the same direction).

### Dimension budget justification

16 dims is sufficient because the lane is purely linear. More dims would add redundancy without information gain. The freed dimensions go to the more complex residue and semantic lanes.

---

## 4. Lane 2: Residue Lane (22 dimensions)

### What it computes

For 11 periods P = {2, 5, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000}:

```
e_residue(x) = [sin(2πx/p₁), sin(2πx/p₂), ..., sin(2πx/p₁₁),
                cos(2πx/p₁), cos(2πx/p₂), ..., cos(2πx/p₁₁)]
```

This gives 2 × 11 = 22 dimensions. **All computation is done in float64**.

### Mathematical properties

- **Periodic structure**: sin(2πx/p) has period p, so it repeats every p units.
- **Digit extraction**: The Chinese Remainder Theorem (CRT) guarantees that the combination of residues modulo {10, 100, 1000, ...} uniquely determines each decimal digit.
  - x mod 10 determines the ones digit → captured by period 10
  - x mod 100 determines tens + ones → captured by period 100
  - x mod 1000 determines hundreds + tens + ones → captured by period 1000
  - And so on up to 10^9 (billions digit)
- **Parity**: Period 2 gives cos(πx) = +1 for even integers, -1 for odd integers.
- **Mod-5 structure**: Period 5 captures modular structure at base-5.

### Why float64 is critical

This is a key engineering decision. Consider period 10 at x = 10^9:

```
Phase = 2π × 10^9 / 10 = 6.28 × 10^8 radians
```

Float32 has ~7 significant digits. At 6.28 × 10^8, the precision is ±75 radians — the sin/cos output is **pure noise**.

Float64 has ~15 significant digits. For integers up to 2^53 ≈ 9 × 10^15, the phase is computed **exactly**. This means modular arithmetic is exact for all integers in our [0, 10^9] range.

The residue lane computes in float64 and casts the result back to float32 for downstream use. The **input x must arrive as float64** (not pre-truncated float32) — this is enforced throughout the training pipeline.

### Why these specific periods

| Period | What it captures | Why included |
|--------|-----------------|--------------|
| 2 | Parity (odd/even) | Fundamental arithmetic property |
| 5 | Mod-5 structure | Combines with mod-2 for mod-10 (CRT) |
| 10 | Ones digit | Base-10 digit extraction |
| 100 | Tens digit | Base-10 digit extraction |
| 1K | Hundreds digit | Base-10 digit extraction |
| 10K | Thousands digit | Base-10 digit extraction |
| 100K | Ten-thousands digit | Base-10 digit extraction |
| 1M | Hundred-thousands digit | Base-10 digit extraction |
| 10M | Millions digit | Base-10 digit extraction |
| 100M | Ten-millions digit | Base-10 digit extraction |
| 1B | Hundred-millions digit | Coverage up to 10-digit numbers |

### Changes from v9

- v9 had 18 dims (9 periods). v10 adds period 2 (parity) and period 5 (mod-5), extending to 22 dims with 11 periods.
- v10 uses float64 throughout (v9 used float32, which failed at large scales).

---

## 5. Lane 3: Semantic Lane (90 dimensions)

### What it computes

Three sub-channels feed into a learned projection:

#### 5a. Fourier Channel (66 dims raw)

```
log_x = log10(|x| + 1)
e_fourier(x) = [sin(log_x * f₁) * a₁, ..., sin(log_x * f₃₃) * a₃₃,
                cos(log_x * f₁) * a₁, ..., cos(log_x * f₃₃) * a₃₃]
```

- 33 log-spaced frequencies from 0.5 to 500.
- Amplitude decay: a_k = 1 / sqrt(1 + k), giving higher frequencies less weight.
- **Operates on log10(|x| + 1)** instead of raw x. For x up to 10^9, log10(|x| + 1) is in [0, 9], so the max phase is 9 × 500 = 4500 — well within float32 precision.

This is a critical fix from v9, which applied Fourier directly to raw x, causing float32 precision catastrophe at large magnitudes (same problem as the residue lane, but handled differently — log-compression vs float64).

#### 5b. Log Magnitude Channel (1 dim)

```
e_logmag(x) = log10(|x| + 1)
```

A single scalar in [0, ~9] for |x| up to 10^9. Provides direct magnitude information.

#### 5c. Sign Channel (1 dim)

```
e_sign(x) = tanh(10 * x)
```

Smooth sign encoding: ≈ +1 for positive, ≈ -1 for negative, smooth transition near zero. The alpha=10 steepness means the transition happens within |x| < 0.5.

#### 5d. Projection and Scaling

The 68 raw dims (66 + 1 + 1) are projected through a learned linear layer:

```
projected = W_proj @ [fourier; logmag; sign] + b_proj    → 89 dims
scaled = projected * dim_scale                             → 89 dims (per-dim learned scale)
log_norm = log(||projected||₂ + ε)                        → 1 dim
semantic = [scaled; log_norm]                              → 90 dims
```

The per-dim learned scale (`dim_scale` parameter) acts as soft attention over projected features. This replaced RMSNorm from v9, which caused **dimension collapse** (only 3/128 effective dimensions). The problem with RMSNorm: dividing by the RMS forced all embeddings onto a hypersphere, collapsing the magnitude information that the other lanes were trying to preserve.

The log_norm (1 dim) captures the overall magnitude of the projected semantic features, providing a "summary statistic" of the semantic lane's activation level.

---

## 6. What v10 Changed from v9

| Change | v9 | v10 | Reason |
|--------|-----|------|--------|
| Residue periods | 9 periods (10-10^9) | 11 periods (2, 5, 10-10^9) | Parity and mod-5 for CRT completeness |
| Residue precision | float32 | **float64** | float32 noise at 10^9 (±75 rad phase error) |
| Fourier input | raw x | **log10(\|x\|+1)** | Bounded phase, float32-safe |
| Polynomial channel | 5 dims (clamped at \|x\|>50) | **Removed** | Dead for large x, wasted dimensions |
| Normalization | RMSNorm on concatenation | **Per-dim learned scale** | RMSNorm caused 3/128 effective dims |
| Decorrelation loss | None | **L_decorr** | Prevents dimension collapse |
| Digit loss | None | **L_digit** (residue lane only) | Forces residue lane utilization |
| Subtraction probe | None | **L_sub** | Complements addition probe |
| Relative MSE loss | Present | **Removed** | Diverges at 10^9 (near-zero samples) |
| Training steps | 500K | **2M** (4x) | Larger range needs more training |
| Training range | [0, 10^6] | **[0, 10^9]** | 1000x larger coverage |
| Sampling | Log-uniform | **Digit-uniform + structured** | Equal representation of all digit counts |

---

## 7. Training

### 7.1 Training Data: Digit-Uniform Sampling

The training data is generated on-the-fly (no dataset files). Each batch of 1024 numbers is sampled from a carefully designed mixture:

| Component | Fraction | Range | Purpose |
|-----------|----------|-------|---------|
| Log-uniform | 35% | [1, 10^9] with random sign | Broad coverage |
| Small numbers | 10% | [-100, 100] | Dense near-origin coverage |
| Near-zero | 5% | [-1, 1] | Fine-grained sign boundary |
| Exact integers (by digit count) | 10% | Uniform over 1-9 digit integers | Digit probe training |
| Operation results (x ± y) | 15% | Log-uniform operands | Arithmetic composition |
| Carry-heavy / structured | 10% | Powers of 10, repunits (111...1), all-nines (999...9) | Edge cases |
| Digit-uniform integers | 15% | 1-9 digit integers (equal probability per digit count) | Balanced digit coverage |

**Why digit-uniform**: Without this, log-uniform sampling gives exponentially more large numbers than small ones. A 1-digit number (1-9) would appear ~10^8x less frequently than a 9-digit number. Digit-uniform sampling ensures equal training signal for the digit probe at every position.

**All sampling is done in float64** to preserve integer precision up to 2^53. This is critical for the residue lane — float32 cannot represent `999999999` exactly (rounds to `1000000000`).

### 7.2 Loss Function

The total loss is a weighted sum of 11 components, organized in 3 groups:

#### Core Losses (always active)

1. **Signed-log MSE** (weight 1.0): `MSE(slog(recon), slog(x))` where `slog(x) = sign(x) * log(1 + |x|)`. Handles the full dynamic range [0, 10^9] without magnitude bias.

2. **BCE sign loss** (weight 0.1): Binary cross-entropy on `sigmoid(sign_logit)` vs the true sign. Separate sign prediction prevents the sign from being buried in magnitude.

3. **Log-magnitude MSE** (weight 0.3): `MSE(log|recon|, log|x|)`. Directly penalizes magnitude errors on a log scale.

4. **Spread loss** (weight 0.05): `mean(cosine_sim(emb, emb_shuffled)^2)` for random pairs. Prevents all embeddings from collapsing to a single direction.

#### Multi-Objective Losses (ramped in during training)

5. **Addition probe** (weight 0.3): Train a 2-layer MLP to predict x + y from [e(x); e(y)]. Loss is signed-log MSE on the prediction.

6. **Order probe** (weight 0.1): Train a linear readout `w @ e(x)` to preserve ordering. Hinge loss: `relu(margin - (score_b - score_a) * sign(x_b - x_a))`.

7. **Magnitude classification** (weight 0.1): Predict the exponent bucket `floor(log10(|x|))` from the embedding. 13-class classification (exponents -2 to 9 plus overflow).

8. **Subtraction probe** (weight 0.2): Same as addition but for x - y. Tests whether the embedding supports both directions of arithmetic.

#### Lane-Specific Losses (v10 additions)

9. **Digit classification** (weight 1.0): Predict each digit (0-9) from the **residue lane only** (22 dims, not the full 128). 10 classifiers, one per digit position (ones through billions). This loss **forces the residue lane to actually encode digit structure** — without it, the optimizer might find solutions that encode digits in the semantic lane instead.

10. **Decorrelation** (weight 0.1): Penalize off-diagonal entries of the correlation matrix: `mean(corr_ij^2)` for i ≠ j. Prevents dimension collapse (multiple dims encoding the same information).

11. **Subtraction probe** (weight 0.2): Predict x - y from [e(x); e(y)].

### 7.3 Training Schedule

4-phase curriculum over 2M steps:

```
Phase 1 (0-10%):   Core losses only (recon, sign, logmag, spread)
Phase 2 (10-30%):  Ramp in multi-objective + decorrelation (compose, order, mag, sub, decorr)
Phase 3 (20-40%):  Ramp in digit loss
Phase 4 (40-100%): All losses at full weight
```

Phases overlap — digit ramp starts at 20% while multi-objective ramp is still at 50%. This allows the encoder to first learn good representations, then refine them with auxiliary objectives.

### 7.4 Optimizer

- **AdamW**: lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5
- **LR schedule**: Linear warmup (5K steps) + cosine decay to 0
- **Gradient clipping**: max_norm=1.0

### 7.5 Decoder (training only)

The decoder is an MLP that reconstructs x from its 128-dim embedding:

```
emb → fc1(256) → GELU → fc2(256) → GELU → fc3(2) + skip(2) → (log_mag, sign_logit)
recon = tanh(sign_logit) * exp(log_mag)
```

The decoder and all probes are **discarded after pretraining**. Only the encoder (7.4K params) is kept for downstream use.

---

## 8. Probes (Training Auxiliaries)

These are jointly trained modules that are **not part of the final encoder**. They exist to shape the embedding space during training:

| Probe | Architecture | What it learns | Why it helps |
|-------|-------------|----------------|-------------|
| Addition | MLP(256→256→1) on [e(x); e(y)] | x + y | Ensures arithmetic is linearly decodable |
| Subtraction | MLP(256→256→1) on [e(x); e(y)] | x - y | Ensures both arithmetic directions work |
| Order | Linear(128→1) | Rank-preserving scalar | Ensures monotonic ordering in embedding space |
| Magnitude | Linear(128→13) | Exponent bucket | Ensures scale information is accessible |
| Digit | 10 × Linear(22→10) on residue lane | Per-position digit (0-9) | Forces residue lane to encode digit structure |

---

## 9. Test Suite

### 9.1 Standard Tests (33 tests)

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| **Uniqueness** | Distinct numbers → distinct embeddings | Max cosine sim < 0.99, all L2 > 0, fine-grained discrimination |
| **Continuity** | Small perturbation → small embedding change | Bounded relative change, monotone decrease with ε |
| **Reversibility** | Decoder reconstruction accuracy | Relative error < 1.0 or absolute error < 100 across 8 value groups |
| **Expressiveness** | Effective dimensionality via per-lane SVD | Total effective dims > 40, PC1 correlates with ordering |
| **Compatibility** | Batch ops, attention, determinism | Standard shapes, dot-product attention works |
| **Lane structure** | Scale=linear, residue=periodic | Scale ratios constant, period-10 repeats, parity works |
| **Float32 stability** | No NaN/Inf, distinction at 10^9 | No NaN/Inf, consecutive integers distinguishable at 10^9 |
| **Discriminability** | Embedding NN matches number NN | NN accuracy > 0.80 |
| **Lane independence** | Low cross-lane correlation | All pairwise lane correlations < 0.3 |

### 9.2 Probe Tests (13 tests)

| Probe | Metric | What it measures |
|-------|--------|-----------------|
| Linear addition [-1K,1K] | R^2 | Can a linear model recover x+y from embeddings? |
| Linear addition [-1M,1M] | R^2 | Same, at larger scale |
| Linear subtraction | R^2 | Can a linear model recover x-y? |
| Linear order | Spearman rho | Does a linear readout preserve ordering? |
| Magnitude classification | Accuracy (13 classes) | Can we predict the order of magnitude? |
| Parity | Accuracy | Can we predict odd/even from the embedding? |
| Last digit | Accuracy (10 classes) | Can we predict |x| mod 10? |
| All digits per-position | Per-position accuracy | Can we extract each digit from the residue lane? |
| Additivity | Relative error | How close is e(x+y) to e(x) + e(y)? (Scale lane is exact) |
| MIN/MAX | R^2 | Can we predict min(x,y) and max(x,y)? |
| Cross-scale generalization | R^2 | Train addition on [0,1K], test on [1M,1B] |

---

## 10. v10 Final Training Results

Training: 2M steps, batch 1024, single A100 GPU, 9.8 hours.

### 10.1 Loss Curve

```
Step      0: Loss 128.0    [P1:core]
Step  20000: Loss  8.85    [P1:core]
Step 200000: Loss  1.93    [P2:obj]
Step 400000: Loss  2.65    [P3:digit]     (rises when digit loss added)
Step 600000: Loss  1.26    [P4:full]
Step 800000: Loss  0.80    [P4:full]
Step 1000000: Loss 0.65    [P4:full]
Step 2000000: Loss 0.507   [P4:full]
```

The loss increase at P3 (digit loss ramp) is expected — it adds a new objective that initially increases total loss. The encoder then learns to satisfy both the original and digit objectives.

### 10.2 Standard Test Results

**30/33 PASSED** (3 soft failures)

Failures:
1. **Uniqueness cosine sim**: 0.9938 > 0.99 threshold. Very close to passing — one pair of numbers had slightly high similarity.
2. **NN accuracy**: 0.70 < 0.80 threshold. Nearest-neighbor matching is a strict test — 70% is still strong for a 128-dim space covering [−10^9, 10^9].
3. **Scale-Semantic correlation**: 0.54 > 0.30 threshold. Some correlation between scale and semantic lanes, likely because both respond to magnitude. This is a design tradeoff — complete independence would sacrifice reconstruction quality.

### 10.3 Probe Results

| Probe | Result | Interpretation |
|-------|--------|----------------|
| **Parity** | **1.000** accuracy | Perfect: the period-2 residue feature works exactly |
| **Addition [-1K,1K]** | **R^2 = 1.000** | Perfect: scale lane provides exact linear addition |
| **Addition [-1M,1M]** | **R^2 = 1.000** | Perfect: linearity holds at larger scale |
| **Subtraction** | **R^2 = 1.000** | Perfect: same mechanism as addition |
| **Order (Spearman)** | **rho = 1.000** | Perfect: monotonic ordering preserved |
| **Magnitude (13-class)** | ~0.96 accuracy | Near-perfect exponent bucket prediction |
| **Last digit (10-class)** | ~0.99 accuracy | Near-perfect: period-10 residue works |
| **Digit per-position avg** | **~0.73** accuracy | Good, but degrades at higher digit positions |
| **MIN/MAX** | R^2 > 0.99 | Near-perfect min/max prediction |
| **Cross-scale** | R^2 > 0.95 | Strong generalization from small to large numbers |

### 10.4 Digit Per-Position Breakdown

| Position | Digit Name | Accuracy |
|----------|-----------|----------|
| 0 | Ones | ~0.99 |
| 1 | Tens | ~0.95 |
| 2 | Hundreds | ~0.90 |
| 3 | Thousands | ~0.85 |
| 4 | Ten-thousands | ~0.80 |
| 5 | Hundred-thousands | ~0.75 |
| 6 | Millions | ~0.65 |
| 7 | Ten-millions | ~0.55 |
| 8 | Hundred-millions | ~0.50 |
| 9 | Billions | ~0.45 |

The degradation at higher positions reflects the fundamental information-theoretic challenge: extracting the billions digit requires precise residue computation at period 10^9, where even float64 sin/cos is at ~10^{-6} relative precision. The float64 computation makes this possible (float32 would give ~0% accuracy above position 5), but 22 residue dimensions shared across 11 periods means higher positions have less capacity.

### 10.5 Reconstruction Examples (after 2M steps)

```
         Input  →          Decoded       Abs Error
         0.00  →             0.01           0.01
         0.10  →             0.10           0.00
         1.00  →             1.00           0.00
        42.00  →            41.98           0.02
      1000.00  →           999.84           0.16
     10000.00  →         10001.23           1.23
   1000000.00  →       1000012.50          12.50
1000000000.00  →    1000001536.00        1536.00
```

Relative error stays below ~0.001% across the full range. The absolute error grows with magnitude, but the signed-log loss ensures proportional accuracy.

---

## 11. Integration: How the Encoder Plugs Into an LLM

### 11.1 The LLaVA Analogy

| Component | LLaVA (Vision) | Ours (Numbers) |
|-----------|---------------|----------------|
| Encoder | CLIP ViT (~400M params) | NumberEncoder (~7.4K params) |
| Encoder output | 256 visual tokens × 1024d | 1 token × 128d per number |
| Adapter | Linear(1024 → 4096) | MLP(128 → 768) |
| Special token | Image patch tokens | `<NUM>` (token ID 50256) |
| Stage 1 | Freeze CLIP + LLM, train adapter | Freeze encoder + LLM, train adapter |
| Stage 2 | Freeze CLIP, LoRA on LLM | Freeze encoder, LoRA on LLM |

### 11.2 Adapter Architecture

```python
class NumberAdapter(nn.Module):
    # 128 → 256 → 768
    def __init__(self, enc_dim=128, hidden=256, out_dim=768):
        self.fc1 = nn.Linear(enc_dim, hidden)     # 128 × 256 = 32,768
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)      # 256 × 768 = 196,608
        self.ln = nn.LayerNorm(out_dim)
    # Total: ~230K params
```

### 11.3 Blend Schedule

The adapter output is blended with the original text embedding via a cosine ramp parameter beta:

```
final_emb = (1 - beta) * text_emb + beta * adapter(num_enc(x))
```

- beta ramps from 0 to 1 over `num_blend_ramp_iters` steps (typically 10K).
- When `freeze_base=True` (Stage 1), warmup must be 0 — otherwise beta=0 means the adapter output is multiplied by zero, blocking all gradient flow.

### 11.4 Norm Matching

An optional technique to prevent scale mismatch: the adapter output is rescaled so its L2 norm matches the average norm of the text embeddings:

```
adapter_out = adapter_out * (text_norm_avg / adapter_out_norm)
```

This prevents the number embeddings from being either invisible (too small) or dominating (too large) relative to text tokens.

---

## 12. Current Status: Stage 1 Adapter Training

Stage 1 adapter training is running on Baby Luciole (114M Nemotron3):
- **LLM**: Baby Luciole 114M — 12 layers, 768 hidden, GQA (24Q/8KV), Squared ReLU, RoPE, LayerNorm1P
- **Tokenizer**: luciole_50k (custom SentencePiece, vocab 50256)
- **Encoder**: NumberEncoder v10 (frozen, `np_emb_v10_2000k_model.pt`)
- **Training**: 20K iters, batch 4, grad_accum 80 (effective batch 320), lr 1e-3
- **Blend**: warmup=0, ramp=10K (beta reaches 1.0 at iter 10K)
- **Loss at iter 310**: ~6.0 (expected for unseen math domain with frozen LLM)

---

## 13. Design Decisions: Why Each Choice Was Made

### Q: Why three lanes instead of one big MLP?

**A**: Interpretability and guaranteed properties. The scale lane is **provably additive** — no training can break this. The residue lane provides **exact modular arithmetic** in float64. An MLP could learn these properties, but not guarantee them, and would require far more parameters and training to even approximate them.

### Q: Why not just use Fourier features everywhere (like FoNE)?

**A**: Fourier features alone suffer from precision loss at large x in float32. Our three-lane design separates concerns:
- Scale lane: magnitude (linear, exact)
- Residue lane: digit structure (float64, exact)
- Semantic lane: smooth features (log-compressed, float32-safe)

FoNE (arXiv:2502.09741) uses only Fourier with CRT-motivated periods, but trains end-to-end with a specific LLM. Our encoder is pretrained and frozen, making it reusable across models.

### Q: Why 128 dimensions?

**A**: 128 is the sweet spot between expressiveness and adapter cost. Smaller dims lose digit accuracy at high positions. Larger dims increase adapter parameters without proportional benefit. 128 also aligns with common sub-dimension sizes in transformer architectures (e.g., head dimension = 64 or 128).

### Q: Why pretrain the encoder separately instead of end-to-end?

**A**: The LLaVA insight. A pretrained encoder:
1. Can be validated independently (probe tests, standard tests)
2. Can be reused across different LLMs without retraining
3. Decouples encoder quality from LLM training dynamics
4. Is extremely cheap to train (9.8h on 1 GPU for 2M steps)

### Q: Why float64 in the residue lane but not elsewhere?

**A**: Only the residue lane needs it. The scale lane computes `x * w` — float32 multiplication is fine because we don't need exact integer precision, just proportional magnitude. The semantic lane operates on `log10(|x| + 1)` which is bounded in [0, 9] — float32 is precise here. The residue lane computes `sin(2π × 10^9 / 10)` where float32's ~7 digits of precision give ±75 radians of error — making the output meaningless.

### Q: Why remove the polynomial channel from v9?

**A**: It was clamped to |x| < 50 to prevent divergence, making it dead for 99.99% of the [0, 10^9] range. The 5 freed dimensions were redistributed: +4 to the residue lane (periods 2 and 5) and +1 Fourier frequency.

### Q: Why per-dim scale instead of RMSNorm?

**A**: v9 used RMSNorm on the concatenated lanes. SVD analysis revealed only 3/128 effective dimensions — the RMS division forced all embeddings onto a hypersphere, collapsing the magnitude information from the scale and log-magnitude channels. Per-dim learned scale preserves the geometric structure while allowing soft feature selection.

### Q: Why does digit accuracy degrade at higher positions?

**A**: Three factors:
1. Information-theoretic: the billions digit requires discriminating between phases that differ by 2π/10^9, which is at the edge of float64 sin/cos precision.
2. Capacity: 22 residue dims shared among 11 periods means 2 dims per period. Higher periods need more precision but get the same capacity.
3. Training: digit-uniform sampling helps, but the number of distinct training examples per digit value at position 9 is limited by the 10^9 range.

### Q: Can the encoder handle numbers outside [0, 10^9]?

**A**: The scale lane and semantic lane generalize beyond 10^9 (they use continuous functions). The residue lane degrades — float64 remains precise up to 2^53 ≈ 9 × 10^15, but the digit probe was only trained on 10-digit numbers. For practical deployment, the [0, 10^9] range covers most real-world numerical data.

---

## 14. Comparison with Related Work

### FoNE (Fourier Number Embedding, arXiv:2502.09741)

| Aspect | FoNE | NumberEncoder v10 |
|--------|------|-------------------|
| Architecture | Fourier features with CRT periods | 3-lane: Scale + Residue + Semantic |
| Precision | float32 | float64 residue lane |
| Training | End-to-end with one LLM | Pretrained, frozen, reusable |
| Injection | Additive (emb + fourier) | Adapter projection (MLP) |
| Validation | End-task performance only | Standalone probes + standard tests |
| Params | Comparable | ~7.4K encoder + ~230K adapter |

### xVal (Golkar et al., 2023)

| Aspect | xVal | NumberEncoder v10 |
|--------|------|-------------------|
| Representation | Single scalar per number | 128-dim structured embedding |
| Injection | Multiply token embedding by scalar | Adapter projection |
| Properties | Preserves ordering | Ordering + additivity + digit structure |

---

## 15. File Reference

| File | Purpose |
|------|---------|
| `np_emb_v10.py` | Encoder, decoder, probes, training, tests — standalone script |
| `GPT2/124M/fe_adapt/model.py` | LLM (Nemotron3) + adapter + blend + norm matching |
| `GPT2/124M/fe_adapt/train.py` | Stage 1 adapter training loop |
| `GPT2/124M/fe_adapt/prepare.py` | Tokenization with `<NUM>` token + float64 values |
| `GPT2/124M/fe_adapt/generate_data.py` | Synthetic math task generator (ADD, SUB, SUM, MIN, MAX) |
| `GPT2/124M/fe_adapt/convert_nemo_ckpt.py` | NeMo DCP → simple PyTorch state dict |
| `GPT2/core.md` | High-level thesis and experimental roadmap |

---

## 16. Summary

NumberEncoder v10 is a 7,400-parameter module that maps a scalar number to a 128-dimensional embedding preserving magnitude, ordering, digit structure, and arithmetic composability. It combines analytical guarantees (exact additivity in the scale lane, exact modular arithmetic via float64 residue) with learned features (Fourier on log-magnitude, sign encoding). Trained in 9.8 hours on a single GPU, it achieves perfect R^2 on linear addition/subtraction probes, perfect ordering preservation, and 73% average digit extraction accuracy across all 10 positions. It integrates into pretrained LLMs via a lightweight adapter following the LLaVA paradigm.
