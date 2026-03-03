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
13. [Extended Evaluation](#13-extended-evaluation)
14. [Key Findings](#14-key-findings)
15. [Additive Embeddings: Theory, Experiments, and v9 Redesign](#15-additive-embeddings-theory-experiments-and-v9-redesign)

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

## 13. Extended Evaluation

The standard validation metrics (Section 12) compare overall exact match and MAE on each model's own validation set. This section presents three additional analyses designed to probe *how* and *when* models fail: conditional error magnitude, difficulty-controlled performance buckets, and out-of-distribution SUM length generalization. All three analyses use teacher-forced evaluation on the same 6,722 numeric examples per model.

Three models are compared: **Base** (text in, text out), **FE-Unfreeze** (NumberEncoder in, SME out), and **FE-TextDec** (NumberEncoder in, text out).

### 13.1 Conditional MAE: "When Wrong, How Wrong?"

Standard MAE averages over all examples, diluting errors from the small fraction of incorrect predictions. **Conditional MAE (CondMAE)** is computed only over examples where the model did *not* achieve exact match. This isolates the error magnitude of the model's failures.

|  | --- Base --- | | | --- FE-Unfreeze --- | | | --- FE-TextDec --- | | |
|---|---|---|---|---|---|---|---|---|---|
| **Task** | **Exact%** | **MAE** | **CondMAE** | **Exact%** | **MAE** | **CondMAE** | **Exact%** | **MAE** | **CondMAE** |
| ADD | 91.7% | 20,805 | 249,655 | 73.8% | 537 | 2,095 | 64.4% | 29,169 | 85,678 |
| COUNT | 99.9% | 0.00 | 1.00 | 100.0% | 0.00 | 0.00 | 100.0% | 0.00 | 0.00 |
| MAX | 99.8% | 1,888 | 899,500 | 94.8% | 27 | 545 | 95.3% | 1,374 | 37,014 |
| MIN | 99.9% | 0.53 | 525 | 86.6% | 16 | 130 | 86.9% | 495 | 4,703 |
| SORT | 98.2% | 49 | 2,223 | 32.5% | 8 | 11 | 30.5% | 1,094 | 1,675 |
| SUB | 93.1% | 4,548 | 66,219 | 75.0% | 1,094 | 4,508 | 66.5% | 13,458 | 41,746 |
| SUM | 63.7% | 182,959 | 504,659 | 39.2% | 2,637 | 4,428 | 27.8% | 121,420 | 168,362 |
| **OVERALL** | **92.6%** | **16,774** | **317,166** | **71.7%** | **358** | **776** | **67.5%** | **16,614** | **40,344** |

**Key observation**: FE-Unfreeze has a conditional MAE of **776** compared to Base's **317,166** — a **408x reduction**. When FE-Unfreeze gets a number wrong, it is typically off by hundreds; when Base gets a number wrong, it is typically off by hundreds of thousands.

FE-TextDec sits between the two (CondMAE 40,344), showing that the SME output format contributes substantially to the proximity effect — SME's structured sign-exponent-mantissa decomposition ensures that even incorrect predictions preserve order of magnitude.

### 13.2 Difficulty-Controlled Evaluation Buckets

Performance is stratified by three difficulty dimensions to reveal where models succeed and fail.

#### By Digit Count of Target

|  | --- Base --- | | | --- FE-Unfreeze --- | | | --- FE-TextDec --- | | |
|---|---|---|---|---|---|---|---|---|---|
| **Digits** | **N** | **Exact%** | **CondMAE** | **N** | **Exact%** | **CondMAE** | **N** | **Exact%** | **CondMAE** |
| 1-dig | 1690 | 94.9% | 0.05 | 1690 | 75.2% | 0.01 | 1690 | 74.6% | 0.13 |
| 2-dig | 275 | 96.4% | 994 | 275 | 78.5% | 28 | 275 | 91.6% | 0.81 |
| 3-dig | 194 | 93.8% | 246 | 194 | 75.8% | 256 | 194 | 80.4% | 118 |
| 4-dig | 395 | 92.9% | 12,597 | 395 | 78.7% | 297 | 395 | 68.9% | 3,947 |
| 5-dig | 2885 | 92.2% | 225,260 | 2885 | 72.7% | 1,016 | 2885 | 64.7% | 34,824 |
| 6+-dig | 1283 | 89.2% | 721,422 | 1283 | 60.8% | 704 | 1283 | 56.7% | 66,999 |

All models degrade with digit count, but FE-Unfreeze's conditional MAE stays remarkably flat (0.01 → 704 across 6 orders of magnitude of target values), while Base's CondMAE explodes (0.05 → 721,422).

#### By List Length (SORT/MIN/MAX/SUM/COUNT only)

|  | --- Base --- | | | --- FE-Unfreeze --- | | | --- FE-TextDec --- | | |
|---|---|---|---|---|---|---|---|---|---|
| **Length** | **N** | **Exact%** | **CondMAE** | **N** | **Exact%** | **CondMAE** | **N** | **Exact%** | **CondMAE** |
| 2 | 544 | 98.5% | 49,251 | 544 | 79.4% | 6,198 | 544 | 78.5% | 11,148 |
| 3 | 563 | 95.7% | 415,454 | 563 | 73.4% | 319 | 563 | 72.1% | 76,562 |
| 4-5 | 1114 | 91.7% | 393,663 | 1114 | 68.9% | 451 | 1114 | 65.0% | 38,272 |
| 6-8 | 1694 | 87.0% | 433,251 | 1694 | 64.6% | 538 | 1694 | 61.8% | 50,856 |
| 9-10 | 886 | 99.1% | 16,593 | 886 | 77.5% | 13 | 886 | 76.2% | 3,457 |

#### Per-Task Exact Match by Digit Count

This reveals a striking inversion pattern for comparison tasks:

| Digits | **MIN (Base)** | **MIN (FE-Unfreeze)** | **MAX (Base)** | **MAX (FE-Unfreeze)** |
|--------|-----------|------------------|-----------|------------------|
| 1-dig | 100.0% | 59.2% | 100.0% | 59.6% |
| 2-dig | 100.0% | 85.7% | 100.0% | 76.9% |
| 3-dig | 97.7% | 93.0% | 100.0% | 94.1% |
| 4-dig | 100.0% | 98.7% | 100.0% | 100.0% |
| 5-dig | 100.0% | 99.0% | 99.8% | 98.7% |
| 6+-dig | 100.0% | 100.0% | 99.6% | 100.0% |

FE-Unfreeze **struggles on small numbers** (1-digit: 59%) but **excels on large numbers** (6+-digit: 100%). This inversion occurs because:
- For small integers (0-9), each value has a dedicated BPE token embedding that the base model can perfectly memorize. The NumberEncoder must compress these into continuous representations, losing the discrete identity.
- For large numbers (5-6 digits), BPE fragments them into multiple tokens whose positional values must be learned implicitly. The NumberEncoder captures magnitude directly via its LogMagnitude and Fourier channels.

### 13.3 SUM Length Generalization

Models were trained on SUM tasks with 2-8 operands. This analysis tests all three models on generated SUM examples with list lengths [2, 3, 5, 8, 10, 15, 20, 30] using **integers 1-100** as operands, evaluated with teacher-forced forward pass (200 examples per length).

|  | --- Base --- | | | --- FE-Unfreeze --- | | | --- FE-TextDec --- | | |
|---|---|---|---|---|---|---|---|---|---|
| **Length** | **Exact%** | **MAE** | **CondMAE** | **Exact%** | **MAE** | **CondMAE** | **Exact%** | **MAE** | **CondMAE** |
| 2 | 35.0% | 57 | 90 | 32.0% | 121 | 197 | 27.5% | 33 | 45 |
| 3 | 15.0% | 91 | 111 | 6.0% | 78 | 85 | 2.5% | 56 | 57 |
| 5 | 0.0% | 143 | 143 | 0.5% | 55 | 56 | 0.0% | 375 | 375 |
| 8 | 0.0% | 44,898 | 44,898 | 0.0% | 69 | 69 | 0.5% | 18,337 | 18,430 |
| 10 * | 0.0% | 40,254 | 40,254 | 0.0% | 48 | 48 | 0.0% | 16,452 | 16,452 |
| 15 * | 0.0% | 798 | 798 | 0.0% | 71 | 71 | 0.0% | 12,512 | 12,512 |
| 20 * | 0.0% | 1,002 | 1,002 | 0.0% | 977 | 977 | 0.0% | 2,900 | 2,900 |
| 30 * | 0.0% | 1,501 | 1,501 | 0.0% | 1,400 | 1,400 | 0.0% | 9,903 | 9,903 |

\* = out-of-distribution (list length > 8, not seen during training)

**Key observations:**
1. **No model generalizes SUM to longer lists.** All three achieve 0% exact match at length 5+.
2. **FE-Unfreeze maintains remarkably low MAE even out-of-distribution.** At length 10, FE-Unfreeze's MAE is 48 while Base's is 40,254 — an **839x difference**. The NumberEncoder's continuous representation preserves a "sense of scale" even when the combinatorial structure is unfamiliar.
3. **Exact match drops dramatically even in-distribution.** The val set SUM accuracy is 63.7% (Base) and 39.2% (FE-Unfreeze), but on these generated examples it's 35% and 32% at length 2. This suggests distribution shift: the generated test examples (integers 1-100) have different number characteristics than the training data (which includes floats, negatives, and variable digit counts).
4. **FE-Unfreeze's MAE stays flat at 48-71 for lengths 5-15**, then rises only at 20+ where the sums exceed the training number range. This flat plateau suggests the model approximates a constant "expected sum" for unfamiliar lengths, which happens to be close to correct for small integers.

---

## 14. Key Findings

### 14.1 The NumberEncoder provides strong input understanding

FE-TextDec proves this conclusively. It uses the NumberEncoder for input and plain BPE text for output — the same output format as Base. On classification tasks (which only require *understanding* the input numbers, not outputting new numbers), FE-TextDec matches Base at 99%+ accuracy. The NumberEncoder successfully encodes numbers into representations that the transformer can use for comparison, ordering, and verification.

Furthermore, FE-TextDec achieves **6.4x lower cross-entropy loss** than Base (0.297 vs 1.908). Since both use the same output format, this difference is entirely due to the input encoding: the NumberEncoder compresses each number into a single information-rich token, whereas BPE fragments numbers into multiple tokens that each carry partial information.

### 14.2 SME output encoding dramatically reduces error magnitude

Base GPT-2 achieves higher exact match (92.56%) than any FE-SME variant (best: 79.29%). However, when Base gets a number wrong, the error can be catastrophic (MAE = 16,774). When FE-SME gets a number wrong, the error is typically small (MAE = 347.6 for Unfreeze). The ratio is **48x**.

This is because of how errors propagate in each representation:

- **SME structure**: The sign and exponent tokens are predicted first and are almost always correct (sign accuracy 99%, exponent accuracy 99.7%). Errors occur in later mantissa digits, producing numerically close values. Getting digit 3 wrong by 1 means the output is off by at most `10^(exponent - 3)`.
- **BPE fragmentation**: A single wrong BPE token can shift the order of magnitude. Predicting `"4"` instead of `"40"` in the sequence `["40", "00"]` turns 4000 into 400 — a 10x error from one token.

### 14.3 Loss is not directly comparable across output formats

FE-SME variants have 5-10x lower cross-entropy loss than Base (0.23 vs 1.91). This does **not** mean they are 5-10x better at the tasks. The difference arises from **information density per token**:

- **Base**: each output token is drawn from ~50,257 possible BPE tokens. The entropy per token is high.
- **SME**: each output token is drawn from a constrained set — 2 signs, 19 exponents, or 10 digits (and constrained decoding further reduces the effective choices). The entropy per token is much lower.
- **FE-TextDec** (0.30) vs **Base** (1.91): Same output format, so this gap is a fair comparison. The NumberEncoder reduces input token count, meaning fewer uncertain positions contribute to the loss.

### 14.4 Unfreezing the encoder helps moderately

FE-Unfreeze outperforms FE-Frozen consistently:
- Numeric exact match: 79.29% vs 74.87% (+4.4 percentage points)
- MAE: 347.6 vs 373.6 (-7%)
- CE loss: 0.223 vs 0.230

This confirms that task-specific fine-tuning of the encoder's `Linear(71, 127)` projection adapts the feature mixing beyond what reconstruction pretraining provides.

### 14.5 Wider adapter does NOT help

FE-Unfreeze+MLP (1.45M adapter params) slightly **underperforms** FE-Unfreeze (107K adapter params):
- Numeric exact: 76.83% vs 79.29% (-2.5 pp)
- MAE: 471.8 vs 347.6 (+36%)

The bottleneck is not adapter capacity. The 2-layer `Linear(128, 256) -> GELU -> Linear(256, 256)` is sufficient to project 128-dim number embeddings to 256-dim transformer inputs. The wider adapter may actually hurt by introducing optimization difficulty (larger parameter space with the same 0.5x learning rate).

### 14.6 Multi-position encoding achieves lowest loss but lower exact match

FE-Multipos achieves the lowest cross-entropy loss (0.132 vs 0.223 for Unfreeze, a 1.7x reduction) but lower exact match (70.26% vs 79.29%). The 5 position-specific projection heads give the transformer more attention targets per number, improving per-token prediction accuracy, but the increased sequence length means each training block contains fewer complete examples, reducing effective data coverage.

### 14.7 Text decoding collapses on multi-number output tasks

FE-TextDec's SORT accuracy (30.5%) is far below both its SME counterpart (80.0% for Unfreeze) and Base (98.2%). SORT requires outputting a correctly-ordered list of multiple numbers as text, which means:
- Multiple numbers in sequence with correct delimiters
- No constrained decoding to guarantee valid number boundaries
- BPE tokenization creates inconsistent splits that are hard to reproduce exactly
- Scientific notation (e.g., `5.964e-07`) fragments into many BPE tokens and the model consistently produces malformed patterns like `ee`, `e1e`, `.-`

Similarly, SUM (27.8% TextDec vs 39.2% Unfreeze) requires outputting a single precise number, where BPE fragmentation makes exact reproduction difficult even when the model has computed the correct answer internally.

### 14.8 Scientific notation is the main text output failure mode

Throughout FE-TextDec training, the model consistently fails on scientific notation. It produces patterns like:
- `6.8ee-07` instead of `5.964e-07`
- `3.1e1e5` instead of `3.14e-05`
- `0.-.5` instead of `-0.005`

This is a fundamental BPE tokenization issue. GPT-2's tokenizer fragments scientific notation inconsistently across different numbers, making it very hard for the model to learn the `e[-+]\d+` pattern reliably from training examples alone.

### 14.9 COUNT is universally perfect

All six variants achieve 99.9-100% on COUNT. This task requires only counting list elements (structural parsing), not understanding numerical values. It confirms that all architectures handle basic sequence parsing well and serves as a sanity check.

### 14.10 Component contribution decomposition

Comparing the three output-format variants with the same unfrozen NumberEncoder input:

| Comparison | Val CE | What it isolates |
|-----------|--------|-----------------|
| Base (text in, text out) | 1.908 | Baseline |
| FE-TextDec (NUM in, text out) | 0.297 | NumberEncoder input benefit: **6.4x CE reduction** |
| FE-Unfreeze (NUM in, SME out) | 0.223 | SME output benefit on top of encoder: **1.3x further** |
| FE-Multipos (5xNUM in, SME out) | 0.132 | Multi-position benefit on top: **1.7x further** |

Both the input encoder and output encoding contribute independently. The input encoder provides the single largest benefit (6.4x), with SME output providing a meaningful but smaller additional gain (1.3x), and multi-position adding further improvement (1.7x) at the cost of increased sequence length.

The total pipeline reduction from Base to FE-Multipos is: `1.908 / 0.132 = 14.5x` lower cross-entropy loss.

### 14.11 FE-Unfreeze errors are 400x closer to the correct answer

The conditional MAE analysis (Section 13.1) reveals the most striking difference between the models. When FE-Unfreeze produces an incorrect answer, it is off by an average of 776. When Base produces an incorrect answer, it is off by an average of 317,166 — a **408x ratio**. This is the strongest evidence that the NumberEncoder provides genuine numerical understanding beyond pattern matching: the continuous embedding preserves proximity, so even incorrect predictions land in the right neighborhood.

FE-TextDec's CondMAE (40,344) falls between the two, confirming that the SME output format contributes roughly half the proximity effect (on a log scale). SME's structured sign-exponent-mantissa decomposition means getting a mantissa digit wrong produces a small numerical error, whereas getting a BPE digit token wrong can shift the value by orders of magnitude.

### 14.12 FE models exhibit a small-number/large-number inversion

The difficulty-controlled analysis (Section 13.2) reveals a consistent pattern across comparison tasks (MIN, MAX): FE-Unfreeze underperforms Base on 1-digit numbers (~59% vs 100%) but matches or exceeds Base on 5-6+ digit numbers (99-100% vs 99-100%). This occurs because small integers (0-9) each have a dedicated BPE token embedding that Base memorizes perfectly, while the NumberEncoder must compress these into a continuous representation where nearby integers have similar embeddings. For large numbers, the situation reverses: BPE fragments them across multiple tokens requiring implicit positional reasoning, while the NumberEncoder captures magnitude directly.

### 14.13 No model generalizes SUM beyond training list lengths

All three models achieve 0% exact match on SUM with 5+ operands (Section 13.3), including in-distribution lengths (5 and 8 are within the training range of 2-8). However, FE-Unfreeze maintains remarkably low MAE even out-of-distribution (48 at length 10 vs Base's 40,254 — an 839x ratio), demonstrating that the encoder's continuous representation preserves scale awareness even when the model cannot compute exact sums.

---

## 15. Additive Embeddings: Theory, Experiments, and v9 Redesign

The extended evaluation (Section 13) showed that FE-Unfreeze's errors are numerically close to correct answers (CondMAE 776 vs Base's 317,166). This raised a natural question: **Can we make the embedding space directly support arithmetic?** Specifically, can we design an encoder where `e(x) + e(y) ≈ e(x + y)`, so that a downstream transformer could perform addition by simply summing embeddings?

### 15.1 Theoretical Foundation

**The impossibility result**: The only continuous functions `f: R → R^d` satisfying `f(x+y) = f(x) + f(y)` for all `x, y` are linear maps: `f(x) = x * v` for some fixed vector `v ∈ R^d`. But `f(x) = x * v` maps all of R to a 1-dimensional subspace (a line through the origin) — useless as an embedding since it collapses all structural information into a scalar multiple.

This means exact additivity across all 128 dimensions is fundamentally incompatible with the expressiveness needed for downstream tasks. Four relaxation strategies were considered:

| Approach | Strategy | Feasibility |
|----------|----------|-------------|
| 1. Log-space additivity | `e(x*y) ≈ e(x) + e(y)` (multiplication) | Already provided by LogMagnitude channel; wrong operation |
| 2. Additivity loss | Add `\|\|e(x+y) - e(x) - e(y)\|\|²` penalty to training | Implemented and tested |
| 3. Additive subspace | Reserve K dims for `x * w` (exact), rest unchanged | Implemented and tested |
| 4. RNS-inspired | `sin/cos(2πx/p)` at coprime periods — rotation = addition | Incorporated into v9 as Residue lane |

Approaches 2 and 3 were implemented as `np_emb_additive.py` and `np_emb_additive_subspace.py` and trained for 500K steps on GPU.

### 15.2 Approach 2: Additivity Loss (`np_emb_additive.py`)

**Architecture**: Identical to v8 (Section 3) — same NumberEncoder, same decoder, same channels. The only change is an additional loss term during training.

**Additivity loss**: For each batch, sample pairs `(x₁, x₂)` from the first and second halves, compute:

```
L_add = ||e(x₁ + x₂) - (e(x₁) + e(x₂))||²
```

weighted by `--add-weight` (default 0.1).

**Ramp schedule**: The additivity loss is off for the first 20% of training (pure reconstruction), linearly ramps from 20-30%, and reaches full weight at 30%+. This prevents the additivity objective from interfering with early feature learning.

**Key property**: The encoder state_dict is identical to v8, making this a **drop-in replacement** — any model code that loads a v8 checkpoint can load this one without modification.

**Training results** (SLURM job 78727, 500K steps):

| Metric | Value |
|--------|-------|
| Final loss | 0.118 (vs 0.005 for v8) |
| Reconstruction at 100K | 0.18% relative error |
| Additivity relative error | **1.03** (error as large as the signal) |
| Standard tests | 22/23 passed (additivity FAILED) |

**Failure analysis**: The additivity test measured `||e(x+y) - (e(x) + e(y))|| / ||e(x+y)||` and found it ≈ 1.0 — meaning the additive combination `e(x) + e(y)` is no closer to `e(x+y)` than a random vector would be. The root cause is **LayerNorm**:

```
LayerNorm normalizes all embeddings to approximately the same L2 norm (~11.4).
  e(x) has norm ~11.4
  e(y) has norm ~11.4
  e(x) + e(y) has norm ~16-23 (depends on alignment)
  e(x+y) has norm ~11.4

The vector sum e(x)+e(y) lives in a fundamentally different norm shell than e(x+y).
LayerNorm makes additivity structurally impossible.
```

The 24x higher training loss (0.118 vs 0.005) confirms the two objectives fought each other throughout training, with neither converging well.

### 15.3 Approach 3: Additive Subspace (`np_emb_additive_subspace.py`)

**Architecture**: Modified NumberEncoder with two explicit lanes:

```
Embedding layout (128 dims):
  [0 .. K-1]   : Additive subspace — x * learned_weight (K=32 dims, NO bias/nonlinearity)
  [K .. 127]   : Standard pipeline — 71 raw → Linear(71, 95) → LayerNorm → concat log_norm → 96 dims
```

The additive subspace satisfies `e_add(x+y) = e_add(x) + e_add(y)` **exactly** by construction, since it is a purely linear function of x with no bias or nonlinearity. The `additive_weight` parameter is initialized with `logspace(-5, -1, 32)` to cover multiple scales.

**Key property**: This is **NOT** a drop-in replacement — the state_dict has different keys (`additive_weight`, different projection dimensions) and requires model code changes to load.

**Training results** (SLURM job 78728, 500K steps):

| Metric | Value |
|--------|-------|
| Final loss | 0.0037 |
| Reconstruction at 100K | 2.05% relative error (vs 0.18% for v8) |
| Additive subspace error | **0.000000** (exactly additive, by construction) |
| Full embedding additivity error | 1.33 |
| Standard tests | 24/25 passed (full additivity FAILED, subspace PASSED) |
| Additive weight range | [-9.29e-06, 1.26e-04] |

**Failure analysis**: The additive subspace is mathematically perfect (`err = 0.000000`), but the **weights collapsed toward zero** (~1e-4), making the additive signal ~600x weaker than the standard lane signal. Three forces drove this collapse:

1. **Scale mismatch**: Pretraining numbers range up to 1e14. Even with `w = 1e-4`, the additive output for `x = 1e14` is `x * w = 1e10` — still very large. The optimizer pushed weights down to keep outputs bounded.
2. **Spread loss conflict**: The spread loss penalizes high cosine similarity between embeddings. Since all additive dims are `x * w_k` (same sign structure for same-sign numbers), they inherently have high cosine similarity — the spread loss actively suppresses them.
3. **Reconstruction doesn't need additivity**: The decoder can reconstruct x perfectly from the standard lane alone (Fourier + LogMag are sufficient). The additive lane provides no reconstruction benefit, so the optimizer has no incentive to keep its weights large.

**Result**: The additive subspace is provably correct but practically useless — a ~600x signal-to-noise ratio means the downstream transformer would need extreme precision to extract the additive information.

### 15.4 Summary of Failures

Both approaches failed for the same fundamental reason: **the v8 architecture is hostile to additivity**.

| Issue | Approach 2 Impact | Approach 3 Impact |
|-------|-------------------|-------------------|
| LayerNorm normalizes to constant norm | Prevents `e(x)+e(y) ≈ e(x+y)` structurally | Only affects standard lane (additive lane bypasses) |
| Spread loss penalizes same-sign structure | Competes with additivity loss | Suppresses additive weight magnitudes |
| Reconstruction-only pretraining | No incentive for arithmetic-useful features | No incentive to keep additive weights large |
| Training distribution (up to 1e14) | Additivity loss dominated by large-number pairs | Forces additive weights very small |

These results led to the v9 redesign, which addresses each failure mode directly.

### 15.5 NumberEncoder v9: Math-Aware Multi-Lane Architecture (`np_emb_v9.py`)

The v9 encoder is a ground-up redesign informed by the failure analysis of approaches 2 and 3, plus seven design recommendations:

1. Multi-objective pretraining (not just reconstruction)
2. Fix LayerNorm destroying scale (replace with RMSNorm)
3. Protect additive lane from normalization/rotation
4. Add digit/precision lane (modular arithmetic)
5. Align sampling distribution with math tasks
6. Probe-based evaluation
7. Prioritize invariances for generic math

#### 15.5.1 Three-Lane Architecture

The 128-dim embedding is partitioned into three specialized lanes:

```
x (scalar)
  |
  +──→ Lane 1: ScaleLane ──→ 16 dims    [x * w, exactly additive, NOT normalized]
  |
  +──→ Lane 2: ResidueLane ──→ 10 dims  [sin/cos at digit periods, NOT normalized]
  |
  +──→ Lane 3: Semantic ──→ 102 dims    [Fourier+LogMag+Sign+Poly → proj → RMSNorm + log_norm]
  |
  v
  concat ──→ 128 dims
```

**Lane 1 — Scale (16 dims)**: `output = x * weight` where `weight` is a learned parameter vector initialized with `logspace(-5, -2, 16)` with alternating signs. This lane satisfies `e_scale(x+y) = e_scale(x) + e_scale(y)` exactly by construction. Unlike Approach 3, the weights are initialized at larger magnitudes (1e-5 to 1e-2 vs 1e-5 to 1e-1) and the multi-objective training provides explicit incentive to keep them meaningful (see composition loss below).

**Lane 2 — Residue (10 dims)**: For each period `p ∈ {10, 100, 1000, 10000, 100000}`, computes `[sin(2πx/p), cos(2πx/p)]`. These are **analytic** (no learnable parameters) and capture digit-level structure:

| Period | What it captures | Example |
|--------|-----------------|---------|
| 10 | Last digit (ones place) | `sin(2π·42/10) = sin(2π·2/10)` — same for 2, 12, 22, ... |
| 100 | Last two digits | Distinguishes 142 from 242 |
| 1,000 | Last three digits | Carry detection across hundreds |
| 10,000 | Four-digit patterns | Useful for 5-digit arithmetic |
| 100,000 | Full 5-digit structure | Covers entire task number range |

These features enable parity detection (period 10 encodes even/odd via `sin(2πx/10)`), carry detection (the 9→0 transition in `sin(2πx/10)` creates a phase discontinuity), and last-digit reasoning.

This lane implements Approach 4 (RNS-inspired embedding) from the theoretical analysis. For integers, addition in the input maps to rotation in the `(sin, cos)` plane — the representation of the sum is computable from the representations of the parts via complex multiplication.

**Lane 3 — Semantic (102 dims)**: Same four analytic channels as v8 (Fourier 64 + LogMag 1 + Sign 1 + Poly 5 = 71 raw dims), projected through `Linear(71, 101)`, then normalized with **RMSNorm** (not LayerNorm), with `log_norm` appended as the 102nd dimension.

**RMSNorm vs LayerNorm**: This is the key architectural fix for the LayerNorm failure:

```
LayerNorm: normed = (x - mean(x)) / std(x) * γ + β
  → Subtracts mean: destroys relative position between dimensions
  → Divides by std: normalizes to constant norm (~11.4 for all inputs)

RMSNorm: normed = x / RMS(x) * γ    where RMS = sqrt(mean(x²) + ε)
  → Does NOT subtract mean: preserves relative structure
  → Divides by RMS: still normalizes scale but preserves direction
  → Learned per-dimension scale γ: different dims can have different magnitudes
```

RMSNorm normalizes the overall magnitude without subtracting the mean, so the direction of the projected vector is preserved. The learned per-dim scale `rms_scale` (initialized to 1.0) allows the network to assign different importance to different semantic dimensions.

#### 15.5.2 Multi-Objective Pretraining

The v9 training loss combines reconstruction (existing from v8) with three new probe-based objectives:

```
L = L_recon + L_compose + L_order + L_magnitude + L_spread
```

| Term | Weight | Description | Addresses |
|------|--------|-------------|-----------|
| L_slog | 1.0 | Signed-log MSE: `MSE(sign(x)·log(1+\|x\|), ...)` | Reconstruction |
| L_bce | 0.1 | BCE on sign prediction | Sign accuracy |
| L_lm | 0.3 | MSE on log-magnitude | Order-of-magnitude |
| L_rel | 0.3 (ramped) | Relative MSE: `(x-x̂)²/(x²+1)` | Precise reconstruction |
| L_spread | 0.05 | Cosine similarity penalty | Anti-collapse |
| **L_compose** | **0.3** (ramped) | Addition probe: predict `x+y` from `[e(x); e(y)]` | **Arithmetic utility** |
| **L_order** | **0.1** (ramped) | Hinge loss: `relu(0.1 - (score_b - score_a) · sign(x_b - x_a))` | **Ordering utility** |
| **L_magnitude** | **0.1** (ramped) | Cross-entropy for 13-class exponent bucket prediction | **Scale awareness** |

**Composition loss (L_compose)**: An `AdditionProbe` (2-layer MLP: `Linear(256, 128) → GELU → Linear(128, 1)`) is trained jointly with the encoder to predict `x₁ + x₂` from the concatenated embeddings `[e(x₁); e(x₂)]`. The prediction and target are compared in **signed-log space** (`sign(z) · log(1+|z|)`) for scale-invariant comparison. This directly rewards the encoder for producing embeddings from which addition is recoverable — the exact property that Approaches 2 and 3 failed to achieve.

**Order loss (L_order)**: An `OrderProbe` (linear: `Linear(128, 1, bias=False)`) maps embeddings to scalars. For pairs `(x_a, x_b)` where `x_a < x_b`, we want `score(e(x_b)) > score(e(x_a))` with a margin of 0.1. Violations are penalized with a hinge loss. This ensures the embedding space preserves numerical ordering via a simple linear readout.

**Magnitude loss (L_magnitude)**: A `MagnitudeProbe` (linear: `Linear(128, 13)`) classifies embeddings into 13 exponent buckets: `{< -6, [-6,-5), ..., [4,5), ≥ 5}`, covering the full range from sub-microscopic to 100K+. Trained with cross-entropy.

**Ramp schedule**: The three new objectives are off for the first 10% of training (pure reconstruction), linearly ramp from 10-20%, and reach full weight at 20%+. The relative MSE term ramps separately at 40-50% (same as v8). This two-phase approach ensures the encoder first learns a stable representation, then the probes refine it for arithmetic utility.

**Probes are discarded**: The AdditionProbe, OrderProbe, and MagnitudeProbe are only used during pretraining. The saved checkpoint contains only the encoder state_dict. These probes shape the encoder's representation but are not part of the downstream GPT-2 model.

#### 15.5.3 Operation-Aware Sampling

The v8 encoder was trained on log-uniform samples — good for covering the number line but poorly aligned with the arithmetic tasks the downstream model faces. The v9 sampling distribution includes:

| Fraction | Source | Purpose |
|----------|--------|---------|
| 30% | Positive log-uniform (1e-14 to 1e14) | Standard coverage |
| 30% | Negative log-uniform | Negative number coverage |
| 10% | Near-zero (-0.01 to 0.01) | Precision near origin |
| 10% | Integers [-1000, 1000] | Integer arithmetic alignment |
| 10% | Operation results (x+y, x-y for random integer pairs) | Direct arithmetic exposure |
| 10% | Carry-heavy/structured (999, 9999, powers of 10, etc.) | Boundary case exposure |

The operation-results fraction ensures the encoder sees (x, y, x+y, x-y) tuples in the same batch, directly benefiting the composition loss which samples pairs within the batch.

#### 15.5.4 Probe-Based Evaluation

After training, the encoder is evaluated with six probes that measure how useful the embeddings are for downstream math — not just reconstruction accuracy:

| Probe | Method | What it measures |
|-------|--------|-----------------|
| **Linear Addition** | Least-squares `W @ [e(x); e(y)] → x+y` | Can addition be recovered linearly? (R²) |
| **Linear Subtraction** | Least-squares `W @ [e(x); e(y)] → x-y` | Can subtraction be recovered linearly? (R²) |
| **Linear Order** | Least-squares `w @ e(x) → x`, then Spearman ρ | Does a linear readout preserve ordering? |
| **Magnitude** | Linear classifier → 13 exponent buckets | Can magnitude be read off the embedding? |
| **Parity** | Linear classifier → even/odd | Does the residue lane capture parity? |
| **Last Digit** | Linear classifier → 10 classes (\|x\| mod 10) | Does the residue lane capture digit structure? |

Additionally, the standard v8 tests (uniqueness, continuity, reversibility, expressiveness, compatibility) are preserved, plus a new lane structure test verifying that the scale lane dims scale linearly with x and the residue lane dims repeat with the correct period.

#### 15.5.5 Encoder Parameters

| Component | Parameters | Learnable |
|-----------|-----------|-----------|
| ScaleLane weight | 16 | Yes |
| ResidueLane | 0 | No (analytic) |
| FourierChannel | 0 | No (analytic) |
| LogMagnitudeChannel | 0 | No (analytic) |
| SignChannel | 0 | No (analytic) |
| PolynomialChannel | 0 | No (analytic) |
| Linear(71, 101) projection | 71 * 101 + 101 = 7,272 | Yes |
| RMSNorm scale | 101 | Yes |
| **Encoder total** | **~7,389** | |
| AdditionProbe (training only) | 256 * 128 + 128 + 128 * 1 + 1 = 33,025 | Discarded |
| OrderProbe (training only) | 128 | Discarded |
| MagnitudeProbe (training only) | 128 * 13 + 13 = 1,677 | Discarded |
| Decoder (training only) | ~38K | Discarded |

The encoder itself has ~7.4K learnable parameters (vs ~9.2K in v8). The probes add ~34.8K during pretraining but are discarded afterward. The checkpoint saves only the encoder state_dict.

#### 15.5.6 How v9 Addresses Each Failure

| Failure Mode (Approaches 2&3) | v9 Fix |
|-------------------------------|--------|
| LayerNorm normalizes to constant norm | RMSNorm preserves direction; scale lane bypasses normalization entirely |
| Spread loss suppresses additive weights | Scale lane is only 16 dims (vs 32); multi-objective losses provide positive incentive to keep them meaningful |
| No incentive for additive structure | Composition loss (L_compose) directly rewards arithmetic recoverability from embeddings |
| Reconstruction-only pretraining | Order loss + magnitude loss ensure embeddings encode comparison and scale information |
| Training distribution misaligned with tasks | Operation-aware sampling includes (x+y, x-y) tuples and carry-heavy numbers |
| No digit-level features | Residue lane provides exact digit-period structure for parity, carry, last-digit reasoning |
| Evaluation measures only reconstruction | Probe-based evaluation directly measures downstream math utility |

#### 15.5.7 Compatibility with GPT-2 Model

The v9 encoder is **NOT** a drop-in replacement for the v8 encoder. The state_dict has different keys:

| v8 Key | v9 Key | Notes |
|--------|--------|-------|
| `proj.weight` (71→127) | `proj.weight` (71→101) | Different output dimension |
| N/A | `scale_lane.weight` | New parameter (16 dims) |
| N/A | `residue_lane.periods` | Buffer, not parameter |
| N/A | `rms_scale` | New parameter (101 dims) |

Integrating v9 into the GPT-2 model (fe_unfreeze, fe_textdec) would require:
1. Updating `model.py` to import and instantiate the v9 `NumberEncoder`
2. Updating the adapter input dimension if needed (still 128→256, unchanged)
3. Loading the v9 checkpoint with the correct key mapping
4. No changes to the transformer blocks, attention, or loss computation


  ┌────────────────┬───────────┬────────────┐
  │     Probe      │ SLURM run │ --load run │
  ├────────────────┼───────────┼────────────┤
  │ Addition R²    │ 0.9997    │ 0.999994   │
  ├────────────────┼───────────┼────────────┤
  │ Addition MAE   │ 11.2      │ 1.1        │
  ├────────────────┼───────────┼────────────┤
  │ Subtraction R² │ 0.9991    │ 1.000000   │
  ├────────────────┼───────────┼────────────┤
  │ Order ρ        │ 0.9996    │ 1.000000   │
  └────────────────┴───────────┴────────────┘

#### 15.6 FE-v9 Integration Stabilization Update (March 3, 2026)

After integrating the v9 encoder into FE training, a key optimization issue was observed: gradients were often dominated by the `num_encoder`/`num_adapter` path while transformer gradients stayed very small. The architecture has now been updated to stabilize this interface.

**Change 1: Blended `<NUM>` injection (instead of hard replacement)**

The old FE-v9 path replaced the token embedding at `<NUM>` positions with adapter output directly:

```
tok_emb[num_mask] = num_proj
```

It now uses a training-time blend:

```
e_num = (1 - beta_t) * e_base + beta_t * delta
```

where:
- `e_base` = original GPT `<NUM>` token embedding
- `delta` = `num_adapter(num_encoder(value))`
- `beta_t` is scheduled from 0 to 1 over training

This keeps early training close to the native GPT embedding manifold and gradually hands control to numeric features.

**Change 2: Norm matching for adapter output**

Before blending, `delta` is norm-matched to `e_base` (per token):

```
delta <- delta * (||e_base|| / ||delta||)
```

This avoids magnitude mismatch between adapter output and token embeddings, reducing early instability.

**Change 3: More conservative adapter learning rate**

Default `adapter_lr_scale` was reduced from `0.5` to `0.2` for FE-v9 training to further reduce early adapter dominance and encourage transformer adaptation.

**Change 4: Expanded diagnostics**

Training diagnostics now report:
- preclip and postclip grad norms by group (`transformer`, `adapter`)
- preclip and postclip grad ratios
- blend/norm stats (`beta`, base norm, raw/effective delta norm, blend norm, norm scale)

Important interpretation note:
- These grad ratios are `||group|| / ||total||` in L2 norm space, so transformer and adapter ratios do **not** sum to 1. Their squared ratios approximately sum to 1.

**Expected behavior with current defaults**

Current schedule defaults:
- `num_blend_beta_start = 0.0`
- `num_blend_warmup_iters = 2000`
- `num_blend_ramp_iters = 18000`
- `num_blend_beta_end = 1.0`

So:
- Iter `< 2000`: `beta = 0.0` (adapter path intentionally off; adapter grads near 0)
- Iter `2000-20000`: smooth cosine ramp
- Iter `>= 20000`: `beta = 1.0` (full numeric injection)

This behavior was confirmed in logs:
- At iter 800: `beta=0.0000`, adapter grad ≈ 0 by design
- At iter 3200: `beta=0.0109`, adapter grads non-zero but transformer still dominant

#### 15.7 FE-v9 Run Update: `78772` / `78816` (March 3, 2026)

Logs reviewed:
- `slurm_logs/gpt2_sme_v9_78772.log` (training)
- `slurm_logs/validate_v9_78816.log` (standard + extended validation)
- Comparison baseline: `slurm_logs/validate_v9_78769.log`

##### 15.7.1 Training behavior (`gpt2_sme_v9_78772.log`)

The run completed successfully to 35k iters with steadily improving eval loss:

| Iter | Train loss | Val loss |
|------|------------|----------|
| 0 | 10.7343 | 10.7326 |
| 5000 | 0.5599 | 0.5640 |
| 10000 | 0.4927 | 0.4929 |
| 15000 | 0.3987 | 0.3988 |
| 20000 | 0.3638 | 0.3601 |
| 25000 | 0.3391 | 0.3381 |
| 30000 | 0.3254 | 0.3225 |
| 35000 | 0.3024 | 0.3007 |

Beta schedule behaved as expected:
- At iter 2000: `beta=0.0000` (adapter inactive by design)
- By iter 22000: `beta=1.0000` (full adapter path)

However, after beta approached 1.0, adapter gradients dominated preclip norm again:
- Iter 22000: preclip total `385.18`, transformer `0.51`, adapter `385.18`
- Iter 35000: preclip total `504.38`, transformer `0.46`, adapter `504.38`

This indicates the blend warmup stabilized early training, but full-handoff still re-enters the adapter-dominant regime.

##### 15.7.2 Standard validation vs previous v9

| Metric | v9 prev (`78769`) | v9 new (`78816`) | Delta |
|--------|-------------------|------------------|-------|
| CE loss | 0.241156 | 0.252861 | Worse |
| Perplexity | 1.272719 | 1.287705 | Worse |
| SME overall | 0.9072 | 0.8912 | Worse |
| SME digit | 0.8226 | 0.7890 | Worse |
| SME d4 | 0.7249 | 0.6199 | Worse |
| Invalid rate | 0.0070 | 0.0081 | Worse |
| Exact value rate | 0.7470 | 0.7076 | Worse |
| MAE | 644.997 | 252.593 | Better |
| RMSE | 20006.689 | 8700.013 | Better |

Interpretation:
- Exact/token quality dropped.
- Error magnitude improved substantially (fewer huge misses on average).

##### 15.7.3 Per-task comparison (ExactVal / MAE)

| Task | Prev Exact | New Exact | Prev MAE | New MAE | Net |
|------|------------|-----------|----------|---------|-----|
| ADD | 0.7427 | 0.6125 | 175.65 | 337.95 | Worse |
| COUNT | 0.9990 | 1.0000 | 0.001 | 0.000 | Slightly better |
| MAX | 0.9255 | 0.9161 | 2.85 | 29.41 | Worse |
| MIN | 0.8059 | 0.7937 | 10.36 | 29.05 | Worse |
| SORT | 0.7015 | 0.7063 | 13.69 | 23.17 | Mixed (exact up, MAE worse) |
| SUB | 0.7700 | 0.6296 | 113.82 | 149.84 | Worse |
| SUM | 0.5033 | 0.2812 | 7891.82 | 2570.67 | Mixed (exact down, MAE better) |

The largest change is SUM: exact correctness dropped heavily, but average numerical distance improved.

##### 15.7.4 Extended evaluation comparison

Overall extended metrics:
- Previous v9: `Exact 70.3%`, `MAE 658.23`, `CondMAE 1239.11`
- New v9: `Exact 63.2%`, `MAE 255.42`, `CondMAE 445.49`

So the new run is less often exactly right, but substantially closer when wrong.

SUM length generalization:
- New run has lower in-range MAE at short lengths (e.g., len2/len3), but still near-zero exact beyond trivial cases.
- OOD lengths (10/15/20/30) remain poor, with high MAE.
- The old suspicious `len=30 MAE=0.00` artifact is gone; new values are non-zero and more plausible.

##### 15.7.5 Practical conclusion for this run

This configuration shifted the model toward "closer numeric answers" at the cost of exact symbolic correctness.

If target objective is exact token/value accuracy, this run is a regression vs previous v9 and also behind FE-unfreeze runs.

If target objective is reducing catastrophic numeric error magnitude, this run is an improvement over previous v9.
