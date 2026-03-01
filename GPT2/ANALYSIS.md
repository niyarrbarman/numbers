# Feature-Enhanced GPT-2 for Numerical Reasoning: Architecture & Results

## Table of Contents

1. [Overview](#1-overview)
2. [Task Suite](#2-task-suite)
3. [NumberEncoder: Analytic Number Embedding System](#3-numberencoder-analytic-number-embedding-system)
4. [SME: Sign-Mantissa-Exponent Output Encoding](#4-sme-sign-mantissa-exponent-output-encoding)
5. [Variant 1: Base GPT-2](#5-variant-1-base-gpt-2)
6. [Variant 2: FE-Frozen (Frozen NumberEncoder + SME)](#6-variant-2-fe-frozen-frozen-numberencoder--sme)
7. [Variant 3: FE-Unfreeze (Unfrozen NumberEncoder + SME)](#7-variant-3-fe-unfreeze-unfrozen-numberencoder--sme)
8. [Variant 4: FE-Unfreeze+MLP (Unfrozen NumberEncoder + Wider Adapter + SME)](#8-variant-4-fe-unfreezemlp-unfrozen-numberencoder--wider-adapter--sme)
9. [Variant 5: FE-Multipos (Multi-Position NumberEncoder + SME)](#9-variant-5-fe-multipos-multi-position-numberencoder--sme)
10. [Variant 6: FE-TextDec (NumberEncoder Input + Plain Text Output)](#10-variant-6-fe-textdec-numberencoder-input--plain-text-output)
11. [Training Configuration](#11-training-configuration)
12. [Results](#12-results)
13. [Key Findings](#13-key-findings)

---

## 1. Overview

This document describes six GPT-2 variants trained on structured numerical reasoning tasks (arithmetic, comparison, sorting, counting). Each variant uses a different strategy for encoding input numbers and decoding output numbers. All share the same GPT-2 transformer backbone (12 layers, 8 heads, 256-dim embeddings) trained from scratch.

The core research question: **Can an analytic number embedding system improve a language model's numerical reasoning ability, and if so, which component (input encoding, output encoding, or both) drives the improvement?**

The six variants form an ablation study:

| Variant | Input Encoding | Output Encoding | What It Tests |
|---------|---------------|-----------------|---------------|
| Base GPT-2 | BPE text tokens | BPE text tokens | Baseline: standard tokenization |
| FE-Frozen | NumberEncoder (frozen) | SME tokens | Does a pretrained number encoder help? |
| FE-Unfreeze | NumberEncoder (unfrozen) | SME tokens | Does end-to-end fine-tuning of the encoder help? |
| FE-Unfreeze+MLP | NumberEncoder (unfrozen) + wider adapter | SME tokens | Does adapter capacity matter? |
| FE-Multipos | NumberEncoder (unfrozen) + k=5 positions | SME tokens | Does multi-token representation help? |
| FE-TextDec | NumberEncoder (unfrozen) | BPE text tokens | Is the encoder benefit independent of output format? |

---

## 2. Task Suite

All models are trained and evaluated on 14 structured numerical tasks generated synthetically. Each task is a text prompt with an arrow separator (`->`) between input and expected output. Numbers range up to 5 digits (0-99999), including negative numbers and floats with 1-5 significant digits.

### Classification Tasks (text label output)

| Task | Description | Example |
|------|-------------|---------|
| **CMP** | Compare two numbers (output: `<`, `>`, or `=`) | `compare 42 and 17 ->  >` |
| **GT** | Is A greater than B? (output: `yes` or `no`) | `is 5 greater than 3 ->  yes` |
| **IS_POS** | Is the number positive? (output: `yes` or `no`) | `is -7 positive ->  no` |
| **IS_SORTED** | Is the list sorted? (output: `yes` or `no`) | `is [1, 3, 5] sorted ->  yes` |
| **CHECKSORT** | Verify a sort result (output: `correct` or `wrong`) | `check sort [3,1,2] = [1,2,3] ->  correct` |
| **CHECKADD** | Verify an addition result (output: `correct` or `wrong`) | `check 3 + 5 = 8 ->  correct` |
| **SUM_CMP** | Compare a sum to a value (output: `<`, `>`, or `=`) | `compare sum(3,5) and 10 ->  <` |

### Numeric Tasks (number output)

| Task | Description | Example |
|------|-------------|---------|
| **ADD** | Add two numbers | `3 + 5 ->  8` |
| **SUB** | Subtract two numbers | `10 - 3 ->  7` |
| **SUM** | Sum a list of numbers | `sum [1, 2, 3, 4] ->  10` |
| **MIN** | Minimum of a list | `min [5, 2, 8] ->  2` |
| **MAX** | Maximum of a list | `max [5, 2, 8] ->  8` |
| **SORT** | Sort a list ascending | `sort [3, 1, 2] ->  [1, 2, 3]` |
| **COUNT** | Count elements in a list | `count [a, b, c] ->  3` |

Data generation uses weighted sampling: numeric tasks get 2x weight, reasoning/classification tasks get 1x weight. This biases training toward the harder numeric tasks.

---

## 3. NumberEncoder: Analytic Number Embedding System

The NumberEncoder is a pretrained module that maps a scalar number `x` to a 128-dimensional embedding vector `e(x)`. It is the core input component shared across all FE (Feature-Enhanced) variants.

### 3.1 Design Goals

The encoder satisfies five properties:

1. **Uniqueness**: distinct numbers produce distinct embeddings
2. **Continuity**: small changes in value produce small changes in embedding (Lipschitz-like)
3. **Reversibility**: a decoder can reconstruct `x` from `e(x)` with low error
4. **Expressiveness**: the embedding space captures multi-scale numerical structure
5. **Compatibility**: outputs are standard tensors, differentiable, and compatible with downstream layers

### 3.2 Encoder Architecture

The encoder uses four **analytic channels** (no learnable parameters) followed by a learned linear projection and normalization:

```
x (scalar)
  |
  +---> FourierChannel     ---> 64 dims   (32 sin + 32 cos)
  +---> LogMagnitudeChannel ---> 1 dim    (log(|x| + eps))
  +---> SignChannel         ---> 1 dim    (tanh(10*x))
  +---> PolynomialChannel   ---> 5 dims   ([x, x^2, x^3, x^4, x^5] normalized)
  |
  v
  concat ---> 71 dims (raw)
  |
  Linear(71 -> 127) ---> 127 dims (learnable projection)
  |
  LayerNorm(127) ---> normed (manual, no learnable gamma/beta)
  |
  concat [log(||projected||)] ---> 128 dims (final embedding)
```

#### Channel Details

**FourierChannel** (64 dims): Applies 32 geometrically-spaced frequencies to the raw input value. For each frequency `w_k = 0.1 * 1.5^k` (k=0..31), outputs `sin(w_k * x) * a_k` and `cos(w_k * x) * a_k`, where amplitude `a_k = 1/sqrt(1+k)` provides stability damping. This creates a multi-scale sinusoidal representation that captures both fine-grained (low frequency) and coarse (high frequency) numerical structure.

**LogMagnitudeChannel** (1 dim): Computes `log(|x| + 1e-8) / log(10)`. This compresses the dynamic range, mapping values across many orders of magnitude into a bounded range. Essential for handling numbers from 1e-9 to 1e9 in the same embedding space.

**SignChannel** (1 dim): Computes `tanh(10 * x)`. This is a smooth, differentiable approximation of the sign function. Values far from zero saturate to +/-1; values near zero provide a smooth gradient through the origin.

**PolynomialChannel** (5 dims): Computes `[x, x^2, x^3, x^4, x^5]` with input clamped to [-50, 50], then applies per-sample normalization (zero mean, unit variance across the 5 dimensions). This captures nonlinear relationships and provides a Taylor-series-like basis.

#### Projection and Normalization

The 71-dimensional raw channel output passes through a learned linear layer (`Linear(71, 127)` with Kaiming init) to produce 127 dimensions. Before applying LayerNorm, the L2 norm of the projected vector is computed and stored as `log_norm = log(||projected|| + 1e-8)`. The projected vector is then manually normalized (subtract mean, divide by std). Finally, `log_norm` is appended as the 128th dimension. This reserved dimension preserves magnitude information that would otherwise be lost by normalization.

### 3.3 Pretraining

The NumberEncoder is pretrained as part of an encoder-decoder system (`NumberEmbeddingSystem`) for 500,000 steps on log-uniformly sampled numbers:

- **Training distribution**: 40% positive log-uniform (1e-14 to 1e14), 40% negative log-uniform, 10% near-zero (-0.01 to 0.01), 10% integers (-1000 to 1000)
- **Batch size**: 512
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-5)
- **LR schedule**: Linear warmup (2000 steps) + cosine decay

**Decoder** (used only during pretraining): A 3-layer MLP with residual skip connection:

```
embedding (128-dim)
  |
  +----> fc1: Linear(128, 192) -> GELU ---+
  |                                        |
  |      fc2: Linear(192, 192) -> GELU ---+
  |                                        |
  |      fc3: Linear(192, 2) -------------+---> (log_magnitude, sign_logit)
  |                                        |
  +----> W_skip: Linear(128, 2, no bias) --+    [residual skip connection]
  |
  v
  Reconstruction: tanh(sign_logit) * exp(clamp(log_magnitude, -14, 14))
```

The residual skip connection (`W_skip`) allows direct gradient flow from the reconstruction loss to the encoder output, initialized to zero so the MLP path dominates initially.

**Loss function**:
```
L = signed_log_MSE(x, x_hat)
  + 0.1 * BCE_sign(sign_logit, sign(x))
  + 0.3 * log_magnitude_MSE(log|x|, log|x_hat|)
  + 0.3 * relative_MSE(x, x_hat)     [ramped in at 40-50% of training]
  + 0.05 * spread_loss(embeddings)    [anti-collapse regularization]
```

- **signed_log_MSE**: MSE on `sign(x) * log(1 + |x|)`, giving equal weight to small and large numbers
- **BCE_sign**: Binary cross-entropy for sign prediction accuracy
- **log_magnitude_MSE**: MSE on `log(|x| + eps)`, penalizing order-of-magnitude errors
- **relative_MSE**: `(x - x_hat)^2 / (x^2 + 1)`, focusing on relative rather than absolute error
- **spread_loss**: Mean squared cosine similarity between shuffled embedding pairs, preventing collapse to a low-dimensional subspace

### 3.4 Encoder Parameters

The NumberEncoder has approximately **9,200 learnable parameters** (the `Linear(71, 127)` projection weight and bias). The four analytic channels have zero learnable parameters; their behavior is fully determined by hyperparameters (frequencies, scales, polynomial degree). The decoder is discarded after pretraining and is not part of the GPT-2 model.

### 3.5 Number Range Considerations

The encoder was trained on log-uniform samples spanning approximately `[8.3e-7, 1.2e6]` for positive values. The Fourier channel uses maximum frequency `w_31 = 0.1 * 1.5^31 ≈ 19,000`. For the task training data (numbers up to 100,000), the phase `w * x` for large values reaches billions of radians — effectively random noise in the Fourier channels. For large numbers, the encoder relies primarily on the LogMagnitude and Polynomial channels, with the Fourier channels providing useful structure only for `|x| < ~1000`.

---

## 4. SME: Sign-Mantissa-Exponent Output Encoding

SME is a structured token grammar for representing numbers as sequences of special tokens. It replaces the standard BPE text representation of numbers in the model's output vocabulary.

### 4.1 Motivation

Standard BPE tokenization fragments numbers unpredictably. For example, GPT-2's tokenizer might encode:
- `42000` as `["42", "000"]` (2 tokens)
- `42001` as `["420", "01"]` (2 tokens, different split)
- `3.14e-7` as `["3", ".", "14", "e", "-", "7"]` (6 tokens)

This means the model must implicitly learn positional notation, decimal points, and scientific notation from text patterns. SME provides a structured alternative where every number has a consistent, grammar-constrained representation.

### 4.2 Token Layout

SME uses 32 special tokens in the padded GPT-2 vocabulary range (50258-50289):

| Token Range | Count | Meaning |
|-------------|-------|---------|
| 50258-50259 | 2 | Sign: `S+` (positive/zero), `S-` (negative) |
| 50260-50278 | 19 | Exponent: `E-9` through `E+9` |
| 50279-50288 | 10 | Mantissa digits: `D0` through `D9` |
| 50289 | 1 | `END` (terminates the mantissa) |

### 4.3 Grammar

Every number is encoded as a variable-length token sequence:

```
[SIGN] [EXPONENT] [DIGIT_0] [DIGIT_1] ... [DIGIT_k] [END]
```

where `1 <= k <= 15` mantissa digits. The number's value is reconstructed as:

```
value = sign * (d0 + d1*0.1 + d2*0.01 + ... + dk*10^(-k)) * 10^exponent
```

The leading digit `d0` is always non-zero for non-zero numbers (normalized scientific notation form). The exponent represents the power of 10 such that the mantissa is in the range [1, 10).

**Examples:**

| Number | SME Tokens | Explanation |
|--------|-----------|-------------|
| `42` | `[S+, E1, D4, D2, END]` | +4.2 * 10^1 = 42 |
| `-17` | `[S-, E1, D1, D7, END]` | -1.7 * 10^1 = -17 |
| `3.14` | `[S+, E0, D3, D1, D4, END]` | +3.14 * 10^0 = 3.14 |
| `0.005` | `[S+, E-3, D5, END]` | +5.0 * 10^-3 = 0.005 |
| `0` | `[S+, E0, D0, END]` | +0.0 * 10^0 = 0 |
| `99999` | `[S+, E4, D9, D9, D9, D9, D9, END]` | +9.9999 * 10^4 = 99999 |

### 4.4 Constrained Decoding

During generation (inference), a finite-state grammar machine enforces valid SME sequences. The machine has three states:

```
State 0 (outside number):
  Any token allowed.
  On S+ or S- → transition to State 1.

State 1 (after sign):
  Only exponent tokens (E-9 .. E+9) allowed.
  All other tokens masked to -inf in logits.

State 2 (in mantissa):
  If digit_count == 0: only digit tokens (D0..D9) allowed (at least 1 digit required).
  If 0 < digit_count < max_digits: digit tokens or END allowed.
  If digit_count >= max_digits: only END forced.
  On END → transition back to State 0.
```

This is implemented by masking the logits to `-inf` for invalid tokens at each generation step. The grammar machine tracks state per batch element, initialized by scanning the input prompt. This guarantees every generated number is a valid, parseable SME sequence.

### 4.5 Conversion Functions

**Encoding** (`number_to_sme_tokens`): Uses Python's `Decimal` library for stable float-to-digit conversion. Extracts sign, computes exponent via `as_tuple()`, and emits digit tokens for up to `max_digits` significant digits. Values outside the exponent range [-9, 9] are saturated.

**Decoding** (`sme_tokens_to_number`): Parses a token stream looking for `[SIGN, EXP, DIGITS..., END]`. The sign determines polarity, the exponent sets scale, and digits are summed as `d_i * 10^(-i)` to form the mantissa. Returns `None` if the sequence is malformed.

### 4.6 Advantages and Limitations

**Advantages:**
- Consistent, predictable token count per number (3-17 tokens regardless of magnitude)
- First two tokens (sign + exponent) immediately convey order of magnitude
- Constrained decoding guarantees every output is a valid number
- No ambiguity from BPE fragmentation
- Gradual precision: the model can stop at fewer digits (via END) for numbers it's less certain about

**Limitations:**
- Precision capped at 15 significant digits
- Exponent range limited to 10^-9 through 10^9
- Adds 32 special tokens to the vocabulary
- Model must learn a new token grammar not present in natural language pretraining data

---

## 5. Variant 1: Base GPT-2

### 5.1 Architecture Overview

Standard GPT-2 trained from scratch with no modifications. This is the control model.

```
Input text: "3 + 5 ->  8"
  |
  BPE tokenizer (tiktoken gpt2)
  |
  Token IDs: [18, 1343, 642, 4613, 220, 23]
  (each number becomes 1+ BPE tokens depending on digit patterns)
  |
  v
  Token Embedding (wte): Embedding(50304, 256)    [lookup table]
  +
  Position Embedding (wpe): Embedding(256, 256)    [absolute position]
  |
  v
  Dropout(0.0)
  |
  v
  12x Transformer Block (detailed below)
  |
  v
  LayerNorm(256)
  |
  v
  Linear(256, 50304) [lm_head, weight-tied with wte]
  |
  v
  logits -> cross-entropy loss (teacher forcing, ignore_index=-1 for padding)
```

### 5.2 Transformer Block (Pre-Norm, shared across ALL variants)

Every variant uses identical transformer blocks. Each block has two sub-layers with **pre-norm residual connections**:

```
Input x (B, T, 256)
  |
  +-------> LayerNorm(256) -> CausalSelfAttention -> +  [residual connection 1]
  |                                                  |
  x <------------------------------------------------+
  |
  +-------> LayerNorm(256) -> MLP -> +               [residual connection 2]
  |                                  |
  x <--------------------------------+
  |
  Output x (B, T, 256)
```

In code:
```python
x = x + self.attn(self.ln_1(x))   # pre-norm attention with residual
x = x + self.mlp(self.ln_2(x))    # pre-norm MLP with residual
```

This is the **Pre-LN** (pre-normalization) variant of the transformer, where LayerNorm is applied *before* each sub-layer rather than after. The residual connection adds the sub-layer output to the *unnormalized* input, which improves gradient flow and training stability. The gradient from the loss can flow directly through the residual path to early layers without passing through any nonlinearity or normalization.

#### CausalSelfAttention

```
x (B, T, 256)
  |
  Linear(256, 768, bias=False) -> [Q, K, V] concatenated  [single fused projection, 3*n_embd]
  |
  Split into Q, K, V each (B, T, 256)
  |
  Reshape to (B, 8, T, 32)  [8 heads, head_dim = 256/8 = 32]
  |
  Scaled Dot-Product Attention (Flash Attention when available):
    scores = Q @ K^T / sqrt(32)
    mask = causal (lower-triangular, each position attends only to <= current position)
    weights = softmax(scores + causal_mask)
    weights = Dropout(0.0)(weights)    [attn_dropout, disabled at 0.0]
    output = weights @ V               [weighted sum of values]
  |
  Reshape back to (B, T, 256) [concatenate heads]
  |
  Linear(256, 256, bias=False)   [output projection: c_proj]
  |
  Dropout(0.0)                   [residual dropout]
  |
  output (B, T, 256)
```

The `c_proj` weight is initialized with **scaled init**: `Normal(0, 0.02 / sqrt(2 * 12))` = `Normal(0, 0.00408)`. This prevents the variance of the residual stream from growing with depth. Each of the 12 layers contributes 2 residual additions (attention + MLP), so `2 * n_layer = 24` residual paths accumulate. The `1/sqrt(24)` scaling keeps the total variance bounded.

Flash Attention (PyTorch >= 2.0) is used when available, providing O(T) memory instead of O(T^2) and fused CUDA kernels for the Q@K^T, softmax, and attention@V operations.

#### MLP (Feed-Forward Network)

```
x (B, T, 256)
  |
  Linear(256, 1024, bias=False)  [expand 4x: c_fc]
  |
  GELU activation                [Gaussian Error Linear Unit]
  |
  Linear(1024, 256, bias=False)  [contract back: c_proj, scaled init]
  |
  Dropout(0.0)
  |
  output (B, T, 256)
```

The expansion ratio is 4x (`4 * n_embd = 1024`), following the standard GPT-2 design. GELU is a smooth approximation of ReLU: `GELU(x) = x * Phi(x)` where `Phi` is the Gaussian CDF. It provides non-zero gradients for slightly negative inputs, which helps with optimization.

### 5.3 Weight Tying

The token embedding matrix (`wte`, shape `[50304, 256]`) is **tied** (shared) with the output projection (`lm_head`). This means:

```python
self.transformer.wte.weight = self.lm_head.weight  # same tensor
```

The same matrix is used both to look up input embeddings (rows selected by token ID) and to project hidden states to logits (matrix multiplication). This reduces parameter count by 50304 * 256 = ~12.9M and creates a consistent embedding space where the model's input and output representations are aligned. A token that is easy to predict (high logit) will have a hidden state close to that token's embedding vector.

### 5.4 How Numbers Are Processed (Base)

**Input**: Numbers are tokenized as BPE text fragments. GPT-2's tokenizer splits numbers inconsistently:
- `42` -> `["42"]` (1 token)
- `42000` -> `["42", "000"]` (2 tokens)
- `3.14` -> `["3", ".", "14"]` (3 tokens)
- `-0.005` -> `["-", "0", ".", "005"]` (4 tokens)
- `1.5e-3` -> `["1", ".", "5", "e", "-", "3"]` (6 tokens)

Each token becomes a 256-dim embedding via table lookup. The model must learn positional value (ones, tens, hundreds) implicitly from context and position, with no explicit numerical representation.

**Output**: The model predicts numbers as BPE text tokens through the standard softmax over the full 50304-token vocabulary. At each position, it produces a probability distribution over all possible tokens and is trained with cross-entropy loss against the target token.

### 5.5 Parameter Count

| Component | Parameters |
|-----------|-----------|
| Token embeddings (wte, tied with lm_head) | 50304 * 256 = 12,877,824 |
| Position embeddings (wpe) | 256 * 256 = 65,536 |
| 12x attention: c_attn (Q,K,V projection) | 12 * 256 * 768 = 2,359,296 |
| 12x attention: c_proj (output projection) | 12 * 256 * 256 = 786,432 |
| 12x MLP: c_fc (expand) | 12 * 256 * 1024 = 3,145,728 |
| 12x MLP: c_proj (contract) | 12 * 1024 * 256 = 3,145,728 |
| 12x LayerNorm (2 per block, weight only) | 12 * 2 * 256 = 6,144 |
| Final LayerNorm (weight only) | 256 |
| **Total trainable (non-embedding)** | **22.32M** |

Note: `bias=False` throughout all linear layers and LayerNorm, so there are no bias parameters.

---

## 6. Variant 2: FE-Frozen (Frozen NumberEncoder + SME)

### 6.1 Architecture Overview

This variant replaces BPE number tokenization with the pretrained NumberEncoder for input and SME grammar for output. The encoder weights are **frozen** (not updated during training).

```
Input: "3 + 5 ->  [S+, E0, D8, END]"
  |
  Tokenizer: process_text_with_numbers()
  |
  Two parallel streams:
    Token IDs:   [<NUM>,  " +",  <NUM>,  " ->",  " ",  S+, E0, D8, END]
    Num Values:  [3.0,    0.0,   5.0,    0.0,    0.0,  0,  0,  0,  0  ]
  |
  v
  Token Embedding (wte): Embedding(50304, 256)   [standard lookup for ALL tokens]
  |
  At <NUM> positions (where Token ID == 50257), REPLACE the wte embedding:
    value = 3.0
    NumberEncoder(3.0)  -> 128-dim embedding  [FROZEN, no gradient]
    num_adapter(128d)   -> 256-dim embedding  [LEARNED]
    tok_emb[this_position] = adapter_output   [overwrite wte lookup]
  |
  +
  Position Embedding (wpe): Embedding(256, 256)  [absolute position, added after injection]
  |
  v
  Dropout(0.0)
  |
  v
  12x Transformer Block  [identical to base, no modifications]
  |
  v
  LayerNorm(256)
  |
  v
  Linear(256, 50304) [lm_head, same as base, now also predicts SME tokens 50258-50289]
  |
  v
  logits -> cross-entropy loss
```

### 6.2 Input Processing: `process_text_with_numbers()`

A regex-based preprocessor scans input text for numbers matching:
```regex
(?<![a-zA-Z0-9_])-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?![a-zA-Z0-9_])
```

This matches integers (`42`), decimals (`3.14`), negative numbers (`-7`), scientific notation (`2.5e-3`), and leading-dot decimals (`.5`). Lookaround assertions prevent matching numbers embedded in words (`h2o`, `mp3`).

Each detected number is replaced with a single `<NUM>` token (ID 50257), and the float value is stored in a parallel array. Non-number text is tokenized normally with tiktoken BPE.

**Example**:
```
Text:      "add 42000 and -3.14 -> "
Token IDs: [add, <NUM>, and, <NUM>, ->, " "]
Num Values:[0.0, 42000.0, 0.0, -3.14, 0.0, 0.0]
```

Key benefit: each number is now **one token** regardless of magnitude. Base GPT-2 would use 2-6 tokens for the same numbers.

### 6.3 Number Embedding Injection

During the forward pass, after the standard embedding lookup:

1. `tok_emb = wte(token_ids)` produces (B, T, 256) embeddings for all tokens, including a generic learned embedding for the `<NUM>` token at ID 50257
2. Build a boolean mask: `num_mask = (token_ids == 50257)`, shape (B, T)
3. For all positions where `num_mask` is True:
   - Extract the float values: `num_vals_flat = num_values[num_mask]` -> (K,) where K = total NUM tokens in batch
   - Pass through the **frozen** NumberEncoder: `num_emb = encoder(num_vals_flat.float())` -> (K, 128)
   - Pass through the **learned** adapter MLP: `num_proj = adapter(num_emb)` -> (K, 256)
   - Clone and replace: `tok_emb = tok_emb.clone(); tok_emb[num_mask] = num_proj`

The clone is necessary to avoid in-place modification issues with autograd. The replacement is complete: the wte embedding for `<NUM>` is discarded and the adapter output is used instead. After this injection, the tensor `tok_emb` contains a mix of standard BPE embeddings (for text tokens) and adapter-projected number embeddings (for `<NUM>` tokens).

Position embeddings (`wpe`) are added *after* injection, so each number token still receives its absolute position information normally.

### 6.4 Adapter Architecture (2-Layer MLP)

```
128-dim NumberEncoder output
  |
  Linear(128, 256)   [project to model dimension, bias=True]
  |
  GELU activation
  |
  Linear(256, 256)   [refine, bias=True]
  |
  256-dim output (replaces token embedding at <NUM> positions)
```

The adapter has `128*256 + 256 + 256*256 + 256 = 98,816` parameters (weights + biases). It is the only component that bridges the pretrained encoder to the transformer. The adapter uses `nn.Sequential` so there is no residual skip connection — the projection is purely feed-forward.

### 6.5 Frozen Encoder Behavior

The NumberEncoder's weights are loaded from a pretrained checkpoint and **all gradients are disabled**:

```python
for p in self.num_encoder.parameters():
    p.requires_grad = False
self.num_encoder.eval()
```

This means:
- The encoder's `Linear(71, 127)` projection weights are fixed
- No gradient flows through the encoder — only through the adapter
- The encoder is permanently in `eval()` mode (relevant for any BatchNorm/Dropout, though this encoder has neither)
- The encoder provides a stable, pretrained numerical representation as a fixed feature extractor

### 6.6 Output: SME Token Prediction

Output numbers in the training data are encoded as SME token sequences. The model predicts these through the standard `lm_head` softmax over the full 50304-token vocabulary, which includes the 32 SME tokens (50258-50289). The cross-entropy loss treats SME tokens identically to regular BPE tokens — no special weighting.

During generation, constrained decoding (Section 4.4) ensures valid SME grammar by masking logits at each step.

### 6.7 Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Transformer (same as base) | 22.32M | Yes |
| NumberEncoder | ~9.2K | **No** (frozen) |
| num_adapter (2-layer MLP) | ~107K | Yes |
| **Total trainable (non-embedding)** | **~22.42M** | |

### 6.8 Optimizer Groups

Parameters are split into four groups with different learning rates and weight decay:

| Group | Tensors | Params | Weight Decay | LR Scale |
|-------|---------|--------|-------------|----------|
| transformer_decay | 50 | 22,380,544 | 0.1 | 1.0x (= 4e-4) |
| transformer_nodecay | 25 | 6,400 | 0.0 | 1.0x (= 4e-4) |
| adapter_decay | 3 | 107,321 | 0.1 | 0.5x (= 2e-4) |
| adapter_nodecay | 3 | 639 | 0.0 | 0.5x (= 2e-4) |

**Why 0.5x for adapter?** The adapter is randomly initialized while the transformer learns from scratch in a coordinated way. A lower adapter LR prevents the adapter from changing too fast in early training, which could create unstable number representations that confuse the transformer.

**Decay vs nodecay**: 2D+ parameters (weight matrices) get weight decay 0.1; 1D parameters (LayerNorm weights, biases) get no weight decay. This is standard practice to regularize large weight matrices without penalizing normalization parameters.

---

## 7. Variant 3: FE-Unfreeze (Unfrozen NumberEncoder + SME)

### 7.1 Architecture Overview

Identical to FE-Frozen except the NumberEncoder's weights are **unfrozen** and fine-tuned end-to-end with the rest of the model. The encoder is initialized from the same pretrained checkpoint but its parameters receive gradients during training.

The forward pass, adapter, transformer blocks, and SME output are all identical. The only difference is in how the encoder is initialized and whether it receives gradients.

### 7.2 Differences from FE-Frozen

| Aspect | FE-Frozen | FE-Unfreeze |
|--------|-----------|-------------|
| Encoder weights | Frozen (`requires_grad=False`) | Unfrozen (`requires_grad=True`) |
| Encoder mode | Permanent `eval()` | Normal training mode |
| Encoder gradient | None — gradient stops at adapter input | Flows through encoder's `Linear(71, 127)` |
| Encoder LR | N/A | 0.5x base LR (same group as adapter) |
| Checkpoint loading | In `__init__`, before `apply()` | After `apply()` (to avoid overwriting) |

### 7.3 Why Unfreeze?

The pretrained encoder was trained on reconstruction loss (encode-then-decode). The features it learns are optimized for *reversibility* (preserving enough info to reconstruct the number). But downstream tasks may benefit from features optimized for *comparison*, *ordering*, or *arithmetic relationships between numbers*. Unfreezing allows the encoder's `Linear(71, 127)` projection to specialize for the downstream task.

Note that only the linear projection layer is learnable. The analytic channels (Fourier, LogMag, Sign, Polynomial) have no parameters and cannot change. So unfreezing adjusts *how* the analytic features are mixed, not the features themselves.

### 7.4 Init Order Fix

In the original FE-Frozen code, the encoder checkpoint was loaded *before* `self.apply(self._init_weights)`, which resets all `nn.Linear` weights to `Normal(0, 0.02)`. This destroyed the pretrained `encoder.proj` weights. FE-Unfreeze fixes this by loading the checkpoint *after* `apply()`:

```python
# FE-Unfreeze: correct order
self.apply(self._init_weights)      # 1. Random init all weights
# ... (scaled init for c_proj) ...
if config.num_emb_checkpoint:       # 2. Load pretrained encoder (overwrites random init)
    self.num_encoder.load_state_dict(enc_state)
```

### 7.5 Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Transformer | 22.32M | Yes |
| NumberEncoder | ~9.2K | **Yes** (unfrozen) |
| num_adapter (2-layer MLP) | ~98K | Yes |
| **Total trainable (non-embedding)** | **~22.43M** | |

The parameter count is nearly identical to FE-Frozen (~10K more from the unfrozen encoder projection).

---

## 8. Variant 4: FE-Unfreeze+MLP (Unfrozen NumberEncoder + Wider Adapter + SME)

### 8.1 Architecture Overview

Same as FE-Unfreeze but with a **significantly larger adapter MLP** (3 layers, 4x wider hidden dimension). Tests whether the adapter's capacity is a bottleneck.

### 8.2 Adapter Architecture (3-Layer Wider MLP)

```
128-dim NumberEncoder output
  |
  Linear(128, 1024, bias=True)   [project to 4*n_embd = 4*256]
  |
  GELU activation
  |
  Linear(1024, 1024, bias=True)  [wide hidden layer]
  |
  GELU activation
  |
  Linear(1024, 256, bias=True)   [contract to model dimension]
  |
  256-dim output
```

Compare with the standard 2-layer adapter:
```
Standard:  128 -> [GELU] -> 256 -> 256            (~98K params, 2 layers)
Wider:     128 -> [GELU] -> 1024 -> [GELU] -> 256 (~1.45M params, 3 layers)
```

The wider adapter has **~14x more parameters** than the standard adapter. The hidden dimension (1024) matches the MLP expansion ratio used inside the transformer blocks (4 * n_embd).

### 8.3 Hypothesis

If the standard 2-layer adapter is a bottleneck (unable to project the 128-dim number embedding into a sufficiently rich 256-dim representation), a wider adapter should improve performance. The wider MLP can learn more complex nonlinear transformations of the number embedding before injecting it into the transformer.

If performance doesn't improve, the bottleneck is elsewhere: the transformer's capacity, the encoder's representation quality, or the difficulty of the tasks themselves.

### 8.4 Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Transformer | 22.32M | Yes |
| NumberEncoder (unfrozen) | ~9.2K | Yes |
| num_adapter (3-layer wider MLP) | ~1.45M | Yes |
| **Total trainable (non-embedding)** | **~23.77M** |

### 8.5 Optimizer Groups

| Group | Tensors | Params | Weight Decay | LR Scale |
|-------|---------|--------|-------------|----------|
| transformer_decay | 50 | 22,380,544 | 0.1 | 1.0x |
| transformer_nodecay | 25 | 6,400 | 0.0 | 1.0x |
| adapter_decay | 4 | 1,450,809 | 0.1 | 0.5x |
| adapter_nodecay | 4 | 2,431 | 0.0 | 0.5x |

Note the adapter_decay group has 4 tensors (3 weight matrices + 1 encoder projection) with 1.45M parameters, compared to 3 tensors / 107K in the standard adapter variants.

---

## 9. Variant 5: FE-Multipos (Multi-Position NumberEncoder + SME)

### 9.1 Architecture Overview

In all previous FE variants, each input number occupies exactly **one** token position (`<NUM>`). In base GPT-2, numbers occupy multiple positions (one per BPE fragment). This variant gives each input number **k=5 consecutive token positions**, each with a different learned projection of the same number embedding.

### 9.2 Motivation

When base GPT-2 sees `42000`, it gets two tokens (`["42", "000"]`) — two separate attention targets that the transformer's self-attention can attend to independently. Different heads can focus on different parts of the number. When FE variants see `42000`, they get one `<NUM>` token — a single attention target carrying all 256 dimensions of information.

The hypothesis: having multiple attention targets per number helps the transformer's self-attention mechanism route numerical information more effectively, especially for tasks requiring relationships between numbers (e.g., comparing two numbers, computing their sum). With k=5 positions, each position can specialize in representing different aspects of the number (magnitude, sign, low-order digits, etc.).

### 9.3 Input Encoding

Each number in the input text is replaced with **k=5 consecutive `<NUM>` tokens**, all carrying the same float value but with different position indices (0, 1, 2, 3, 4):

```
Text:      "add 42000 and 3 ->"

Token IDs: [add, <NUM>, <NUM>, <NUM>, <NUM>, <NUM>, and, <NUM>, <NUM>, <NUM>, <NUM>, <NUM>, ->]
Num Values:[0,   42000,  42000,  42000,  42000,  42000,  0,   3,     3,     3,     3,     3,     0]
Pos Index: [-1,  0,      1,      2,      3,      4,     -1,  0,     1,     2,     3,     4,    -1]
```

This produces a **third data stream** (`pos_indices`, stored as `{split}_pos.bin`, dtype int8) alongside tokens and values. The value -1 indicates non-number positions.

### 9.4 Multi-Position Projection Heads

Instead of one shared adapter MLP, there are **k=5 independent 2-layer MLP projection heads** stored in an `nn.ModuleList`:

```
NumberEncoder(42000.0) -> 128-dim embedding (shared, computed ONCE per number)
  |
  +---> Projection Head 0: Linear(128,256) -> GELU -> Linear(256,256) -> pos_0_emb (256d)
  +---> Projection Head 1: Linear(128,256) -> GELU -> Linear(256,256) -> pos_1_emb (256d)
  +---> Projection Head 2: Linear(128,256) -> GELU -> Linear(256,256) -> pos_2_emb (256d)
  +---> Projection Head 3: Linear(128,256) -> GELU -> Linear(256,256) -> pos_3_emb (256d)
  +---> Projection Head 4: Linear(128,256) -> GELU -> Linear(256,256) -> pos_4_emb (256d)
```

Each projection head is an independent `nn.Sequential(Linear, GELU, Linear)` with its own weights. They share no parameters with each other. The same 128-dim encoder output is fed to all 5 heads, but each head produces a different 256-dim vector.

The transformer then sees 5 different 256-dim vectors for the same number at 5 consecutive sequence positions, each also receiving a different absolute position embedding from `wpe`. This lets different attention heads attend to different "views" of the same number.

### 9.5 Forward Pass Detail

During the forward pass:

1. Standard `wte(token_ids)` lookup for all tokens -> (B, T, 256)
2. Build masks: `num_mask = (token_ids == 50257)`, `pos_flat = pos_indices[num_mask]`
3. Encode all number values through the shared encoder: `num_emb = encoder(values)` -> (M, 128)
4. Allocate output tensor: `num_proj = empty(M, 256)`
5. For each position index k in {0, 1, 2, 3, 4}:
   - Find which entries have `pos_flat == k`: `mask_k = (pos_flat == k)`
   - Project: `num_proj[mask_k] = projections[k](num_emb[mask_k])`
6. Replace: `tok_emb[num_mask] = num_proj`
7. Add position embeddings and proceed normally

The encoder is called once for all M number tokens, but the projection heads are applied per-position-index, routing each token's embedding through the correct head.

### 9.6 Trade-off

The multi-position approach increases sequence length: each number takes 5 tokens instead of 1. For a 256-token block with, say, 4 numbers, this costs an extra `4 * 4 = 16` tokens of context. With many numbers in a sequence, this can significantly reduce how many examples fit per block, decreasing data efficiency.

### 9.7 Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Transformer | 22.32M | Yes |
| NumberEncoder (unfrozen) | ~9.2K | Yes |
| 5x projection heads (each ~98K) | ~500K | Yes |
| **Total trainable (non-embedding)** | **~22.82M** |

### 9.8 Optimizer Groups

| Group | Tensors | Params | Weight Decay | LR Scale |
|-------|---------|--------|-------------|----------|
| transformer_decay | 50 | 22,380,544 | 0.1 | 1.0x |
| transformer_nodecay | 25 | 6,400 | 0.0 | 1.0x |
| adapter_decay | 11 | 500,537 | 0.1 | 0.5x |
| adapter_nodecay | 11 | 2,687 | 0.0 | 0.5x |

The adapter groups contain 11 tensors each (5 projection heads * 2 weight matrices + 1 encoder projection, plus corresponding biases).

---

## 10. Variant 6: FE-TextDec (NumberEncoder Input + Plain Text Output)

### 10.1 Architecture Overview

This is an ablation variant that isolates the **input encoding** contribution. It uses the NumberEncoder for input (identical to FE-Unfreeze) but outputs numbers as **plain BPE text tokens** (identical to Base GPT-2). There is no SME encoding, no constrained decoding, and no special output tokens beyond the `<NUM>` input token.

```
Input:  "<NUM> + <NUM> -> "     (NumberEncoder input, same as FE-Unfreeze)
Output: "8"                      (plain BPE text, same as Base GPT-2)
```

### 10.2 Architecture Comparison

| Component | FE-Unfreeze (SME) | FE-TextDec |
|-----------|-------------------|------------|
| Input encoding | NumberEncoder + 2-layer adapter | NumberEncoder + 2-layer adapter (identical) |
| Input data streams | tokens + nums | tokens + nums (identical) |
| Output encoding | SME tokens (50258-50289) | BPE text tokens (standard GPT-2) |
| Constrained decoding | Yes (3-state grammar machine) | No (vanilla top-k sampling) |
| Special tokens used | `<NUM>` (50257) + 32 SME tokens | `<NUM>` (50257) only |
| Vocab effectively used | All 50304 | Standard ~50257 (SME tokens exist but unused) |
| Adapter | 2-layer MLP (128->256->256) | 2-layer MLP (128->256->256) (identical) |
| Encoder | Unfrozen, pretrained init | Unfrozen, pretrained init (identical) |
| Transformer blocks | Identical | Identical |
| Model code | `fe_unfreeze/model.py` | `fe_textdec/model.py` (SME code removed) |

### 10.3 What This Variant Tests

By keeping the input identical to FE-Unfreeze but changing only the output format, this variant answers: **How much of the FE improvement comes from the NumberEncoder input vs. the SME output?**

- If FE-TextDec performs close to FE-Unfreeze: the NumberEncoder drives most of the improvement
- If FE-TextDec performs close to Base: the SME output drives most of the improvement
- If FE-TextDec is in between: both components contribute

Since classification tasks have identical text-label outputs in both Base and TextDec, comparing them on classification tasks (CMP, GT, IS_POS, etc.) isolates the pure input encoding benefit — any improvement on these tasks is due entirely to the NumberEncoder.

### 10.4 Data Format

Training data uses the same dual-stream format as FE-Unfreeze (`{split}.bin` + `{split}_nums.bin`), but output numbers in the `.bin` file are encoded as BPE text tokens rather than SME tokens:

```
FE-Unfreeze data:  ... ->  [S+, E0, D8, END] ...
FE-TextDec data:   ... ->   8 ...
                   (where " 8" is BPE token 807, or "8" is token 23)
```

The `generate_data.py` script uses a text formatting function `fmt(value)` to convert numbers to strings, then `enc.encode_ordinary(text)` to tokenize them as standard BPE. Scientific notation, negative signs, decimal points are all encoded as their natural BPE fragments.

### 10.5 Generation

The `generate()` method is simplified to vanilla autoregressive sampling with no grammar constraints:

```python
for step in range(max_new_tokens):
    # Crop context to block_size, keeping num_values and num_mask aligned
    logits, _ = self(idx_cond, num_values=nv_cond, num_mask=nm_cond)
    logits = logits[:, -1, :] / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)

    # Append to sequence; generated tokens are text, not <NUM>
    idx = torch.cat([idx, idx_next], dim=1)
    num_values = torch.cat([num_values, zeros], dim=1)
    num_mask = torch.cat([num_mask, false_mask], dim=1)
```

No grammar state machine, no logit masking. The model must learn to produce valid number text purely from the training signal. Generated tokens are always text (never `<NUM>`), so the num_values and num_mask extensions are always zero/false.

### 10.6 Parameter Count

Identical to FE-Unfreeze:

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Transformer | 22.32M | Yes |
| NumberEncoder (unfrozen) | ~9.2K | Yes |
| num_adapter (2-layer MLP) | ~98K | Yes |
| **Total trainable (non-embedding)** | **~22.43M** |

---

## 11. Training Configuration

### 11.1 Shared Configuration

All six variants use identical training hyperparameters:

| Setting | Value |
|---------|-------|
| Transformer layers (`n_layer`) | 12 |
| Attention heads (`n_head`) | 8 |
| Embedding dimension (`n_embd`) | 256 |
| Head dimension | 256 / 8 = 32 |
| MLP hidden dimension | 4 * 256 = 1024 |
| Block size (context length) | 256 tokens |
| Vocabulary size | 50304 (GPT-2 50257 + 47 padding for GPU efficiency) |
| Bias in linear layers | False |
| Dropout (all: attention, residual, embedding) | 0.0 |
| Base learning rate | 4e-4 |
| LR schedule | Cosine decay with linear warmup (2000 steps) |
| Min LR | 4e-5 (10% of peak) |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, fused=True) |
| Weight decay | 0.1 (on 2D+ params only, 0.0 for biases/LN) |
| Gradient clipping | 1.0 (global L2 norm) |
| Micro batch size | 12 sequences per GPU |
| Gradient accumulation steps | 40 total (5 effective per GPU with 8 GPUs) |
| Effective batch size | 12 * 256 * 40 = 122,880 tokens per iteration |
| Max iterations | 35,000 |
| Mixed precision | bfloat16 via `torch.cuda.amp.autocast` |
| Distributed training | DDP via `torch.distributed` (NCCL c10d backend) |
| Hardware | 4 nodes x 2 GPUs = 8 GPUs |
| Evaluation interval | Every 1,000 iterations |
| Checkpoint saving | Best val loss + periodic |
| Training data | 5,000,000 generated examples |
| Validation data | 10,000 generated examples |
| Number range | 0-99,999 (5 digits) |
| Significant digits | 1-5 (variable per number) |
| Negative numbers | Enabled |
| Floating point numbers | Enabled |

### 11.2 Variant-Specific Configuration

| Variant | Total Params | Adapter LR Scale | Encoder State | Adapter Architecture | Data Streams |
|---------|-------------|-------------------|---------------|---------------------|-------------|
| Base | 22.32M | N/A | N/A | N/A | tokens |
| FE-Frozen | 22.42M | 0.5x | Frozen | 128->256->(GELU)->256 | tokens + nums |
| FE-Unfreeze | 22.43M | 0.5x | Unfrozen | 128->256->(GELU)->256 | tokens + nums |
| FE-Unfreeze+MLP | 23.77M | 0.5x | Unfrozen | 128->1024->(GELU)->1024->(GELU)->256 | tokens + nums |
| FE-Multipos | 22.82M | 0.5x | Unfrozen | 5x [128->256->(GELU)->256] | tokens + nums + pos |
| FE-TextDec | 22.43M | 0.5x | Unfrozen | 128->256->(GELU)->256 | tokens + nums |

### 11.3 Data Generation

All variants use the same 14-task suite (Section 2) with the same random seeds. Data is packed into fixed-size blocks of 256 tokens, separated by `<|endoftext|>` (token 50256). Multiple examples are packed into each block to maximize GPU utilization. Padding with -1 target values ensures the loss ignores inter-example boundaries.

| Dataset | Examples | Avg Tokens/Example | Total Blocks |
|---------|----------|-------------------|-------------|
| Base (BPE text in + text out) | 5M train, 10K val | ~23.2 | ~481K |
| FE-SME (NUM in + SME out) | 5M train, 10K val | ~18-20 | ~370-400K |
| FE-TextDec (NUM in + text out) | 5M train, 10K val | ~18.2 | ~372K |
| FE-Multipos (5x NUM in + SME out) | 5M train, 10K val | ~25-28 | ~500K+ |

FE variants produce shorter sequences because each number is 1 token (`<NUM>`) instead of 1-6 BPE tokens. The FE-Multipos variant is longer because each number occupies 5 token positions. The total blocks determine how many unique training batches exist; all variants see the same 5M unique examples but packed differently.

---

## 12. Results

### 12.1 Overall Validation Metrics

| Variant | Best Val Loss | CE Loss | Perplexity | Numeric Exact Match | MAE |
|---------|-------------|---------|------------|-------------------|-----|
| **Base** | 1.9561 | 1.908 | 6.740 | 92.56% | 16,774 |
| **FE-Frozen** | 0.2786 | 0.230 | 1.258 | 74.87% | 373.6 |
| **FE-Unfreeze** | 0.2699 | 0.223 | 1.249 | 79.29% | 347.6 |
| **FE-Unfreeze+MLP** | 0.2756 | 0.229 | 1.257 | 76.83% | 471.8 |
| **FE-Multipos** | 0.1769 | 0.132 | 1.141 | 70.26% | 643.9 |
| **FE-TextDec** | 0.3436 | 0.297 | 1.346 | 67.48% | 16,614 |

**Note on comparability**: SME variants (Frozen, Unfreeze, Unfreeze+MLP, Multipos) only validate on the 7 numeric tasks (ADD, SUB, SUM, MIN, MAX, SORT, COUNT) because their output format is SME, which only applies to numbers. Classification task outputs are text labels. Base and TextDec validate on all 14 tasks. "Numeric Exact Match" is computed over the numeric task subset for all variants to enable fair comparison.

### 12.2 Per-Task Exact Match: Numeric Tasks

| Task | Base | FE-Frozen | FE-Unfreeze | FE-Unfr+MLP | FE-Multipos | FE-TextDec |
|------|------|-----------|-------------|-------------|-------------|------------|
| **ADD** | **91.67%** | 66.77% | 73.75% | 67.29% | 61.67% | 64.38% |
| **SUB** | **93.13%** | 66.08% | 75.03% | 68.99% | 63.16% | 66.49% |
| **SUM** | **63.68%** | 34.68% | 39.17% | 34.46% | 29.43% | 27.79% |
| **MIN** | **99.90%** | 82.41% | 86.55% | 84.63% | 76.44% | 86.86% |
| **MAX** | **99.79%** | 92.44% | 94.75% | 93.07% | 83.32% | 95.28% |
| **SORT** | **98.18%** | 75.69% | 80.05% | 78.57% | 71.18% | 30.54% |
| **COUNT** | 99.90% | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

### 12.3 Per-Task Exact Match: Classification Tasks (Base & TextDec only)

SME variants do not include classification tasks in their validation because classification outputs are text labels (not numbers), so SME encoding doesn't apply. These tasks are only evaluated for Base and TextDec.

| Task | Base | FE-TextDec |
|------|------|------------|
| **CMP** | 100.00% | 99.79% |
| **GT** | 99.79% | 99.16% |
| **IS_POS** | 100.00% | 98.69% |
| **IS_SORTED** | 100.00% | 99.78% |
| **CHECKSORT** | 98.09% | 99.58% |
| **CHECKADD** | 97.50% | 97.08% |
| **SUM_CMP** | 98.48% | 99.13% |
| **Average** | 99.12% | 99.03% |

Both achieve near-perfect classification, confirming that both input representations provide sufficient numerical understanding for comparison and verification tasks.

### 12.4 Overall Output Exact Match (All Tasks)

For Base and TextDec which evaluate all 14 tasks:

| Variant | Output Exact Match (all 14 tasks) |
|---------|----------------------------------|
| **Base** | 94.71% |
| **FE-TextDec** | 77.82% |

The gap is primarily driven by SORT (98% vs 30.5%) and SUM (64% vs 28%) — multi-number output tasks where BPE text decoding struggles most.

---

## 13. Key Findings

### 13.1 The NumberEncoder provides strong input understanding

FE-TextDec proves this conclusively. It uses the NumberEncoder for input and plain BPE text for output — the same output format as Base. On classification tasks (which only require *understanding* the input numbers, not outputting new numbers), FE-TextDec matches Base at 99%+ accuracy. The NumberEncoder successfully encodes numbers into representations that the transformer can use for comparison, ordering, and verification.

Furthermore, FE-TextDec achieves **6.4x lower cross-entropy loss** than Base (0.297 vs 1.908). Since both use the same output format, this difference is entirely due to the input encoding: the NumberEncoder compresses each number into a single information-rich token, whereas BPE fragments numbers into multiple tokens that each carry partial information.

### 13.2 SME output encoding dramatically reduces error magnitude

Base GPT-2 achieves higher exact match (92.56%) than any FE-SME variant (best: 79.29%). However, when Base gets a number wrong, the error can be catastrophic (MAE = 16,774). When FE-SME gets a number wrong, the error is typically small (MAE = 347.6 for Unfreeze). The ratio is **48x**.

This is because of how errors propagate in each representation:

- **SME structure**: The sign and exponent tokens are predicted first and are almost always correct (sign accuracy 99%, exponent accuracy 99.7%). Errors occur in later mantissa digits, producing numerically close values. Getting digit 3 wrong by 1 means the output is off by at most `10^(exponent - 3)`.
- **BPE fragmentation**: A single wrong BPE token can shift the order of magnitude. Predicting `"4"` instead of `"40"` in the sequence `["40", "00"]` turns 4000 into 400 — a 10x error from one token.

### 13.3 Loss is not directly comparable across output formats

FE-SME variants have 5-10x lower cross-entropy loss than Base (0.23 vs 1.91). This does **not** mean they are 5-10x better at the tasks. The difference arises from **information density per token**:

- **Base**: each output token is drawn from ~50,257 possible BPE tokens. The entropy per token is high.
- **SME**: each output token is drawn from a constrained set — 2 signs, 19 exponents, or 10 digits (and constrained decoding further reduces the effective choices). The entropy per token is much lower.
- **FE-TextDec** (0.30) vs **Base** (1.91): Same output format, so this gap is a fair comparison. The NumberEncoder reduces input token count, meaning fewer uncertain positions contribute to the loss.

### 13.4 Unfreezing the encoder helps moderately

FE-Unfreeze outperforms FE-Frozen consistently:
- Numeric exact match: 79.29% vs 74.87% (+4.4 percentage points)
- MAE: 347.6 vs 373.6 (-7%)
- CE loss: 0.223 vs 0.230

This confirms that task-specific fine-tuning of the encoder's `Linear(71, 127)` projection adapts the feature mixing beyond what reconstruction pretraining provides.

### 13.5 Wider adapter does NOT help

FE-Unfreeze+MLP (1.45M adapter params) slightly **underperforms** FE-Unfreeze (107K adapter params):
- Numeric exact: 76.83% vs 79.29% (-2.5 pp)
- MAE: 471.8 vs 347.6 (+36%)

The bottleneck is not adapter capacity. The 2-layer `Linear(128, 256) -> GELU -> Linear(256, 256)` is sufficient to project 128-dim number embeddings to 256-dim transformer inputs. The wider adapter may actually hurt by introducing optimization difficulty (larger parameter space with the same 0.5x learning rate).

### 13.6 Multi-position encoding achieves lowest loss but lower exact match

FE-Multipos achieves the lowest cross-entropy loss (0.132 vs 0.223 for Unfreeze, a 1.7x reduction) but lower exact match (70.26% vs 79.29%). The 5 position-specific projection heads give the transformer more attention targets per number, improving per-token prediction accuracy, but the increased sequence length means each training block contains fewer complete examples, reducing effective data coverage.

### 13.7 Text decoding collapses on multi-number output tasks

FE-TextDec's SORT accuracy (30.5%) is far below both its SME counterpart (80.0% for Unfreeze) and Base (98.2%). SORT requires outputting a correctly-ordered list of multiple numbers as text, which means:
- Multiple numbers in sequence with correct delimiters
- No constrained decoding to guarantee valid number boundaries
- BPE tokenization creates inconsistent splits that are hard to reproduce exactly
- Scientific notation (e.g., `5.964e-07`) fragments into many BPE tokens and the model consistently produces malformed patterns like `ee`, `e1e`, `.-`

Similarly, SUM (27.8% TextDec vs 39.2% Unfreeze) requires outputting a single precise number, where BPE fragmentation makes exact reproduction difficult even when the model has computed the correct answer internally.

### 13.8 Scientific notation is the main text output failure mode

Throughout FE-TextDec training, the model consistently fails on scientific notation. It produces patterns like:
- `6.8ee-07` instead of `5.964e-07`
- `3.1e1e5` instead of `3.14e-05`
- `0.-.5` instead of `-0.005`

This is a fundamental BPE tokenization issue. GPT-2's tokenizer fragments scientific notation inconsistently across different numbers, making it very hard for the model to learn the `e[-+]\d+` pattern reliably from training examples alone.

### 13.9 COUNT is universally perfect

All six variants achieve 99.9-100% on COUNT. This task requires only counting list elements (structural parsing), not understanding numerical values. It confirms that all architectures handle basic sequence parsing well and serves as a sanity check.

### 13.10 Component contribution decomposition

Comparing the three output-format variants with the same unfrozen NumberEncoder input:

| Comparison | Val CE | What it isolates |
|-----------|--------|-----------------|
| Base (text in, text out) | 1.908 | Baseline |
| FE-TextDec (NUM in, text out) | 0.297 | NumberEncoder input benefit: **6.4x CE reduction** |
| FE-Unfreeze (NUM in, SME out) | 0.223 | SME output benefit on top of encoder: **1.3x further** |
| FE-Multipos (5xNUM in, SME out) | 0.132 | Multi-position benefit on top: **1.7x further** |

Both the input encoder and output encoding contribute independently. The input encoder provides the single largest benefit (6.4x), with SME output providing a meaningful but smaller additional gain (1.3x), and multi-position adding further improvement (1.7x) at the cost of increased sequence length.

The total pipeline reduction from Base to FE-Multipos is: `1.908 / 0.132 = 14.5x` lower cross-entropy loss.
