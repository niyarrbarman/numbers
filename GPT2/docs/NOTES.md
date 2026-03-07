# Development Notes: Bugs, Range Analysis, and Generalization

Supplementary notes to [ANALYSIS.md](ANALYSIS.md). Contains bug documentation, number range analysis, and the plug-and-play generalization discussion.

---

## 1. Bugs Found & Fixed

### 1.1 Block-Aligned Sampling (Critical — multipos only)

**File:** `fe_multipos/train.py:156`

**Bug:** `get_batch()` sampled random offsets into the flat data array, not block-aligned offsets. With k=5 tokens per number, this split ~37% of batches' first number group, giving the model partial representations (e.g., positions [2,3,4] without [0,1]) that never occur during generation.

**Evidence:** Sample outputs showed inconsistent repetitions — `<55590>` (1 token), `<-1.962e-06>` repeated 4 times instead of 5.

**Fix:**
```python
# Before (random offset):
ix = torch.randint(len(data) - block_size - 1, (batch_size,))

# After (block-aligned):
n_blocks = len(data) // block_size
block_ix = torch.randint(n_blocks, (batch_size,))
ix = block_ix * block_size
```

**Status:** Fixed. The v2 multipos run (validate_multipos_78641.log) uses the corrected code. `validate_checkpoint.py` already used block-aligned iteration and did not need fixing.

### 1.2 best_val_loss Tracking (Minor — multipos only)

**File:** `fe_multipos/train.py:560`

**Bug:** `best_val_loss = losses['val']` was inside the `always_save_checkpoint` block, so it always updated regardless of whether val loss actually improved. If val loss went up then came back down, `ckpt_best.pt` could contain a non-optimal checkpoint.

**Fix:** Moved `best_val_loss = losses['val']` inside the `if losses['val'] < best_val_loss:` block so it only updates when a genuine improvement occurs.

**Status:** Fixed in multipos. The same bug exists in all other `train.py` variants but hasn't caused issues because val loss has been monotonically decreasing in all other runs.

### 1.3 Init Order Bug (Historical — fixed in fe_unfreeze, fe_multipos)

**File:** `fe/model.py` (original frozen variant)

**Bug:** In the original FE-Frozen code, the NumberEncoder checkpoint was loaded in `__init__` *before* `self.apply(self._init_weights)` was called. `_init_weights` resets all `nn.Linear` modules to `Normal(0, 0.02)`, which overwrote the pretrained `num_encoder.proj` weights with random values. The encoder effectively started from random initialization despite having a pretrained checkpoint.

**Impact:** The FE-Frozen variant trained with a randomly-initialized encoder projection. The pretrained Fourier/LogMag/Sign/Poly channels still provided useful analytic features, but the learned mixing (the `Linear(71, 127)` projection) was random. This means FE-Frozen's results may understate the benefit of the pretrained encoder.

**Fix in FE-Unfreeze and later variants:**
```python
# Correct order:
self.apply(self._init_weights)       # 1. Random init everything
# ... scaled init for c_proj ...
if config.num_emb_checkpoint:        # 2. THEN load pretrained encoder
    self.num_encoder.load_state_dict(enc_state)  # overwrites random init
```

**Status:** Fixed in `fe_unfreeze/model.py`, `fe_multipos/model.py`, `fe_textdec/model.py`. The original `fe/model.py` still has the bug but is not used for any results reported in ANALYSIS.md (FE-Frozen was re-run with the fix).

### 1.4 Multipos dtype Bug (v1 only)

**File:** `fe_multipos/train.py` (v1)

**Bug:** `pos_indices` was loaded as float16 (inherited from the binary file dtype) instead of being cast to `torch.long` before being used as an index into `num_projections`. This caused silent failures in the position-head routing, with some projections receiving wrong subsets of embeddings.

**Fix:** Explicit cast: `PI = PI.to(torch.long)` after loading from the data file.

**Status:** Fixed in v2. The v1 multipos results (validate_multipos_78638.log) are affected and excluded from the main ANALYSIS.md results.

---

## 2. Number Range Mismatch

### 2.1 Encoder Training Range

The NumberEncoder was pretrained on `sample_training_numbers()` from `np_emb_torch.py`:

| Distribution | Range | Fraction |
|-------------|-------|----------|
| Log-uniform positive | `exp(uniform(-14, 14))` = ~[8.3e-7, 1.2e6] | 40% |
| Log-uniform negative | same range, negated | 40% |
| Near-zero | uniform(-0.01, 0.01) | 10% |
| Integers | randint(-1000, 1000) | 10% |

### 2.2 Task Data Range

The downstream numerical tasks use:
- `NUMBER_RANGE=100000` -> integers in [-100,000, 100,000]
- Float generation with 1-5 significant digits
- SME exponent range: E-9 to E+9 -> values as small as 1e-9

### 2.3 Mismatches

| Range | Encoder trained on | Task data contains | Potential Issue |
|-------|-------------------|-------------------|-----------------|
| Very small | down to ~8.3e-7 | down to 1e-9 | Encoder never saw values < 8.3e-7 during pretraining |
| Large | up to ~1.2e6 | up to 100,000 | OK — within the pretraining range |
| Near-zero | [-0.01, 0.01] | continuous | OK — well covered |
| Integers | [-1000, 1000] only | [-100,000, 100,000] | Large integers (>1000) only seen via log-uniform, not as exact integers |

### 2.4 Fourier Frequency Saturation

The Fourier channel uses 32 geometrically-spaced frequencies:
```
w_k = 0.1 * 1.5^k,  k = 0..31
w_0  = 0.1
w_10 = 5.77
w_20 = 332
w_31 = 19,172
```

For the Fourier features to be useful, the phase `w_k * x` should not wrap around too many times. When `w_k * x >> 2*pi`, the sin/cos values oscillate rapidly and become effectively random noise.

| Input value | Max meaningful frequency | Useful Fourier dims |
|-------------|------------------------|---------------------|
| x = 1 | All 32 (max phase = 19,172) | ~32 (some wrapping at high freq) |
| x = 10 | ~k=28 (phase = 191,720 at k=31) | ~28 |
| x = 100 | ~k=25 | ~25 |
| x = 1,000 | ~k=21 | ~21 |
| x = 10,000 | ~k=18 | ~18 |
| x = 100,000 | ~k=14 | ~14 |

For 5-digit numbers (x ~ 100,000), roughly half the Fourier dimensions (k=15..31) produce random noise. The encoder must rely on:
- **LogMagnitude** (1 dim): `log(100000) / log(10) = 5.0` — perfectly informative
- **Sign** (1 dim): saturated to +1 or -1 — informative for sign
- **Polynomial** (5 dims): `x^5 = 10^25` before clamping — clamped to 50^5, loses information for x > 50
- **Low Fourier** (14 useful dims): still informative for coarse structure

This means for large numbers, the encoder effectively operates with ~20 useful dimensions out of 71 raw dims, relying heavily on the learned `Linear(71, 127)` projection to extract signal from the non-noisy subset. This may explain why exact match drops on tasks involving large numbers (5-digit values) compared to earlier 3-digit experiments where the number range was within the encoder's full Fourier range.

### 2.5 Implications for Unfreezing

Unfreezing the encoder allows the `Linear(71, 127)` projection to be fine-tuned, which could help it learn to:
1. Down-weight the noisy high-frequency Fourier dimensions for large numbers
2. Up-weight the LogMagnitude and low-frequency Fourier dimensions that remain informative
3. Specialize the mixing for the specific number range used in the tasks

This may partially explain why FE-Unfreeze outperforms FE-Frozen (+4.4% exact match): the frozen encoder uses pretrained weights optimized for the full [-1e6, 1e6] range, while the unfrozen encoder can adapt to the specific [-100K, 100K] range of the task data.

---

## 3. Plug-and-Play Number Encoder: Generalization to Other LLMs

### 3.1 Core Idea

The NumberEncoder is architecture-agnostic. Every transformer-based LLM has the same structure at its input:

```
input_ids -> token_embedding_lookup -> tok_emb (B, T, d_model)
```

The number encoder injects at exactly this point: detect number tokens, replace their embeddings with learned continuous representations, and let the rest of the transformer proceed unchanged. This makes it a plug-and-play module that can augment any pretrained LLM.

### 3.2 Universal Injection Pattern

```python
# Works for any decoder-only transformer:
tok_emb = model.embed_tokens(input_ids)         # standard lookup
if num_mask.any():
    num_emb = number_encoder(num_values[num_mask])   # encode scalars -> 128d
    num_proj = adapter(num_emb)                       # project to d_model
    tok_emb[num_mask] = num_proj                      # replace in-place
# Everything downstream (attention, MLP, layernorm, RoPE) is unchanged
```

The only requirement is an adapter layer to match the encoder's output dimension (128) to the model's hidden dimension (`d_model`). This is a single linear layer or small 2-layer MLP — negligible parameter overhead relative to the base model.

### 3.3 Compatibility by Model Family

| Model | d_model | Tokenizer | Position Encoding | Injection Difficulty |
|-------|---------|-----------|-------------------|---------------------|
| GPT-2 | 768 | BPE (tiktoken) | Learned absolute | Easy (current implementation) |
| LLaMA / LLaMA 3 | 4096 | SentencePiece | RoPE | Easy — RoPE applied after embedding, no conflict |
| Mistral | 4096 | SentencePiece | RoPE | Same as LLaMA |
| Phi-3 | 3072 | tiktoken | RoPE | Easy |
| GPT-Neo / GPT-J | 2048-4096 | BPE | Learned / RoPE | Easy |
| Gemma | 2048-3072 | SentencePiece | RoPE | Easy |

The injection point is always pre-attention, so architectural differences downstream (grouped-query attention, sliding window attention, gated MLPs, RMSNorm vs LayerNorm, SwiGLU vs GELU) do not affect compatibility.

**Why RoPE is not a problem:** Rotary Position Embeddings are applied to Q and K vectors *inside* the attention module, not to the token embeddings. The number encoder replaces `tok_emb` before any positional information is added, so the RoPE rotation happens normally on the replaced embeddings just as it would on the original ones.

### 3.4 What Changes Per Model

1. **Adapter dimension:** `nn.Linear(128, d_model)` or `nn.Sequential(Linear(128, d_model), GELU, Linear(d_model, d_model))` — trivial change, ~100K-4M parameters depending on d_model

2. **Tokenizer integration:** The `process_text_with_numbers()` regex operates on raw text before tokenization, making it tokenizer-agnostic. The `NUM_TOKEN_ID` just needs to be set to an unused token ID in the target model's vocabulary. Most modern LLMs have padding in their vocab or allow adding special tokens.

3. **SME output tokens:** If using SME output encoding, the 32 token IDs (50258-50289) need to be mapped to unused IDs in the target vocabulary. Most models have padding tokens or expandable vocab that can accommodate this. Alternatively, for models where vocab expansion is difficult, the FE-TextDec approach (plain text output) avoids the need for any special output tokens.

4. **Training strategy at scale:** For small models (<1B params), full fine-tuning is practical. For large models (8B+), the recommended approach is:
   - Freeze or LoRA/QLoRA the transformer weights
   - Fully train the encoder + adapter (small parameter count)
   - Optionally unfreeze the encoder if LoRA rank is sufficient to co-adapt

### 3.5 Scaling Considerations

| Scale | Strategy | Estimated Cost |
|-------|----------|---------------|
| ~22M (our GPT-2 variant) | Full fine-tune from scratch | 8 GPUs, ~12 hours |
| ~124M (GPT-2 small) | Full fine-tune, optionally init from pretrained | 1-4 GPUs, hours |
| ~1B (LLaMA-like) | Full fine-tune or LoRA | 4-8 GPUs, hours-days |
| ~8B (LLaMA 3.1 8B) | LoRA for transformer, full train for encoder+adapter | 4-8 A100s, days |
| ~70B | QLoRA for transformer, full train for encoder+adapter | 8+ A100s, days-weeks |

### 3.6 Key Advantage Over Alternatives

Standard LLMs represent numbers as sequences of digit/subword tokens:
- `12345` -> `["123", "45"]` (2 tokens, BPE-dependent, no numerical meaning)
- `12346` -> `["123", "46"]` (2 tokens, nearly identical number but completely different second token)
- `12345.6` -> `["123", "45", ".", "6"]` (4 tokens, decimal splits unpredictably)

The NumberEncoder provides:
- **Continuity:** nearby numbers get similar embeddings (12345 ≈ 12346 in embedding space), so the model can generalize from seen numbers to unseen ones
- **Scale awareness:** the encoder captures magnitude, sign, and fine-grained structure in a single vector through its analytic channels
- **Tokenizer independence:** the same encoder works regardless of how the tokenizer would have split the number — `42000` gets the same embedding whether BPE would have produced `["42", "000"]` or `["420", "00"]`
- **Information density:** one token carries the full numerical value instead of spreading it across multiple tokens that must be composed

### 3.7 Pretrained Initialization Path

For maximum impact, combine the number encoder with pretrained transformer weights:

1. Load pretrained LLM (e.g., `LLaMA-3.1-8B`)
2. Add NumberEncoder (load from pretrained encoder checkpoint, ~9K params)
3. Add adapter layer (randomly initialized, ~100K-4M params depending on d_model)
4. Add `<NUM>` as a special token in the tokenizer
5. Fine-tune on numerical tasks:
   - Lower LR for pretrained transformer weights (or use LoRA)
   - Higher LR for adapter (adapter_lr_scale may need to be >1.0 in this setting, since the pretrained transformer is already well-trained and the adapter needs to catch up)
6. The pretrained transformer already understands language; it just needs to learn to use the continuous number representations

This is the most promising direction — the pretrained model brings linguistic competence, and the number encoder brings numerical competence. Neither has to learn the other's domain from scratch.

### 3.8 Open Questions for Generalization

1. **Does the encoder benefit scale with model size?** Larger models may already have better implicit number understanding from pretraining on more data. The marginal benefit of the encoder may decrease.

2. **Does SME output scale?** For pretrained models, the SME tokens are completely novel — the model has never seen them during pretraining. It may take significant fine-tuning to learn the SME grammar. The FE-TextDec approach (keeping text output) may be more practical for pretrained models since it doesn't require learning new output tokens.

3. **Multi-position at scale?** With larger context windows (4K-128K tokens), the sequence length cost of k=5 positions per number becomes less significant. Multi-position may be more effective at larger scale.

4. **Encoder pretraining range:** The current encoder was trained on numbers up to ~1e6. Real-world numerical data spans much wider ranges (financial data: 1e-4 to 1e12; scientific data: 1e-30 to 1e30). The encoder's Fourier frequencies may need to be adjusted or the pretraining range expanded.
