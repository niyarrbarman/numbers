# FE-SME GPT-2 Analysis

## Results Summary (5-digit, 5M data, 35K iters)

### Exact Match & MAE (Full Validation Set)

| Variant | Exact Value | MAE | Perplexity |
|---|---|---|---|
| Base GPT-2 | task-level* | 16,774 | 6.74 |
| FE Frozen | 74.87% | 373.6 | 1.258 |
| FE Unfreeze | **79.29%** | **347.6** | 1.249 |
| FE Unfreeze+MLP | 76.83% | 471.8 | 1.257 |
| FE Multipos (buggy) | 69.68% | 728.0 | 1.140 |

*Base uses text output, not SME — exact match measured per-task below.

### SME Token Accuracy (Validation)

| Metric | Frozen | Unfreeze | Unf+MLP | Multipos (buggy) |
|---|---|---|---|---|
| Overall | 0.919 | **0.929** | 0.921 | 0.905 |
| Sign | 0.986 | **0.990** | 0.991 | 0.991 |
| Exp | **0.995** | **0.997** | 0.996 | 0.995 |
| Digit | 0.845 | **0.863** | 0.847 | 0.815 |
| End | 0.993 | 0.990 | 0.992 | 0.991 |
| d0 | 0.970 | **0.979** | 0.970 | 0.961 |
| d1 | 0.846 | **0.879** | 0.859 | 0.789 |
| d2 | 0.780 | **0.783** | 0.775 | 0.760 |
| d3 | 0.725 | **0.748** | 0.727 | 0.688 |
| d4 | 0.700 | **0.746** | 0.701 | 0.653 |

### Per-Task Exact Match (Validation)

| Task | Base | Frozen | Unfreeze | Unf+MLP | Multipos (buggy) |
|---|---|---|---|---|---|
| ADD | **91.7%** | 66.8% | 73.8% | 67.3% | 61.0% |
| SUB | **93.1%** | 66.1% | 75.0% | 69.0% | 63.5% |
| SORT | **98.2%** | 75.6% | 80.0% | 78.6% | 69.8% |
| MAX | **99.8%** | 92.3% | 94.8% | 93.0% | 84.4% |
| MIN | **99.9%** | 82.4% | 86.6% | 84.6% | 75.7% |
| SUM | 63.7% | 34.7% | 39.2% | 34.5% | 30.6% |
| COUNT | 99.9% | **100%** | **100%** | **100%** | **100%** |

### Per-Task MAE (Validation)

| Task | Base | Frozen | Unfreeze | Unf+MLP | Multipos (buggy) |
|---|---|---|---|---|---|
| ADD | 20,805 | 1,452 | **537** | 1,614 | 1,509 |
| SUB | 4,548 | 357 | 1,094 | 1,338 | **613** |
| SORT | 49 | 94 | **7** | 82 | 303 |
| MAX | 1,888 | 147 | **27** | 178 | 417 |
| MIN | 1 | 141 | **16** | 110 | 339 |
| SUM | 182,959 | **1,961** | 2,637 | 2,098 | 4,369 |
| COUNT | 0 | 0 | 0 | 0 | 0 |

---

## Bugs Found & Fixed

### 1. Block-Aligned Sampling (Critical — multipos only)

**File:** `fe_multipos/train.py:156`

**Bug:** `get_batch` sampled random offsets into the flat data array, not block-aligned offsets. With k=5 tokens per number, this split ~37% of batches' first number group, giving the model partial representations (e.g., positions [2,3,4] without [0,1]) that never occur during generation.

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

**Status:** Fixed. v2 run submitted with fix. `validate_checkpoint.py` already used block-aligned iteration — no fix needed there.

### 2. best_val_loss Tracking (Minor — multipos only)

**File:** `fe_multipos/train.py:560`

**Bug:** `best_val_loss = losses['val']` was inside the `always_save_checkpoint` block, so it always updated regardless of whether val loss improved. If val loss went up then came back down, `ckpt_best.pt` could contain a non-optimal checkpoint.

**Fix:** Moved `best_val_loss = losses['val']` inside the `if losses['val'] < best_val_loss:` block.

**Status:** Fixed. Note: same bug exists in all other train.py variants but hasn't caused issues since val loss has been monotonically decreasing in all runs so far.

### 3. Init Order Bug (Historical — fixed in fe_unfreeze, fe_multipos)

**Bug:** `self.apply(self._init_weights)` ran AFTER loading the encoder checkpoint, destroying pretrained `num_encoder.proj` weights. Fixed in fe_unfreeze and fe_multipos by loading checkpoint AFTER apply. Still unfixed in original `fe/model.py`.

---

## Number Range Mismatch

### Encoder Training Range (np_emb_torch.py)

The NumberEncoder was trained on `sample_training_numbers()`:
- **Log-uniform positive:** `exp(uniform(-14, 14))` → ~[8.3e-7, 1.2e6]
- **Log-uniform negative:** same range, negated
- **Near-zero:** uniform(-0.01, 0.01)
- **Integers:** randint(-1000, 1000)

### FE Task Training Range

Data generated with:
- `NUMBER_RANGE=100000` → numbers in [-100,000, 100,000]
- SME exponent range: E-9 to E9 → values as small as 1e-9

### Mismatches

| Range | Encoder trained on | Task data contains | Issue |
|---|---|---|---|
| Very small | down to ~8.3e-7 | down to 1e-9 | Encoder never saw values < 8.3e-7 |
| Large | up to ~1.2e6 | up to 100,000 | OK — within range |
| Fourier freqs | max ω ≈ 19,000 | values up to 100,000 | sin/cos oscillate wildly for large values |

**Impact:** The Fourier channel uses `sin(ω·x)` with max frequency ~19,000. For x=100,000, the phase is ~1.9 billion radians — effectively random noise. The encoder's useful representation may degrade significantly for |x| > ~1000, relying only on log-magnitude, sign, and polynomial channels for large numbers.

This could explain why exact match drops on tasks involving large numbers (5-digit values) compared to earlier 3-digit experiments where the number range was within the encoder's effective Fourier range.

---

## Architecture Variants

### Base GPT-2 (22M params)
- Standard nanoGPT, n_embd=256, 12 layers, 8 heads
- Numbers tokenized as individual digit tokens by BPE
- Each 5-digit number → ~5-6 tokens
- Wins on exact match, loses badly on MAE (45x worse than FE)

### FE Frozen (22M + 9K encoder params)
- NumberEncoder (128d) frozen, loaded from pretrained checkpoint
- Single adapter: Linear(128→256) projects to transformer dim
- Each number → 1 token position
- Baseline FE variant

### FE Unfreeze (22M + 9K encoder params)
- Same as frozen but encoder is trainable (unfrozen)
- Init order bug fixed (checkpoint loaded AFTER apply)
- +4.4% exact match over frozen
- Best FE variant so far

### FE Unfreeze+MLP (22M + 525K adapter params)
- Unfrozen encoder + wider 2-layer MLP adapter (128→512→256)
- Worse than plain unfreeze — bigger adapter made optimization harder
- Transformer gradient magnitude dropped (0.0771 vs 0.2391)

### FE Multipos k=5 (22M + 500K projection params)
- 5 separate projection heads, each Linear(128→256, GELU, 256→256)
- Each number occupies k=5 consecutive <NUM> token positions
- Position indices stored in separate pos.bin (int8)
- Designed to address single-token bottleneck
- Results inconclusive due to sampling bug; v2 run pending

---

## Key Observations

### The Single-Token Bottleneck
FE compresses each number into one 256-dim vector, while base GPT-2 gets ~5-6 separate tokens per number. The left-to-right digit accuracy cascade (d0: 97% > d1: 85% > d2: 78% > d3: 72% > d4: 70%) is consistent with information loss through a single-vector bottleneck.

### FE's Trade-off
Base GPT-2 wins on exact match (91-99% on most tasks) but has 45x worse MAE than FE unfreeze (16,774 vs 348). FE-SME trades some exact match for dramatically better approximate predictions — when FE is wrong, it's wrong by much less.

### Unfreezing Helps, Bigger Adapters Don't
Simply unfreezing the encoder (+4.4% exact match) is more effective than adding a larger adapter (-2.5% exact match). The bottleneck is in the representation quality, not the projection capacity.

### SUM is Hard for Everyone
SUM accuracy is the lowest task across all variants (30-64%). This is expected — SUM requires precise multi-number aggregation, and errors compound with more operands.

---

## Plug-and-Play Number Encoder: Generalization to Other LLMs

### Core Idea

The NumberEncoder is architecture-agnostic. Every transformer-based LLM has the same structure at its input:

```
input_ids → token_embedding_lookup → tok_emb (B, T, d_model)
```

The number encoder injects at exactly this point: detect number tokens, replace their embeddings with learned continuous representations, and let the rest of the transformer proceed unchanged. This makes it a plug-and-play module that can augment any pretrained LLM.

### Injection Point

```python
# Universal injection pattern (works for any decoder-only transformer):
tok_emb = model.embed_tokens(input_ids)        # standard lookup
if num_mask.any():
    num_emb = number_encoder(num_values[num_mask])  # encode scalars
    num_proj = adapter(num_emb)                      # project to d_model
    tok_emb[num_mask] = num_proj                     # replace in-place
# Everything downstream (attention, MLP, layernorm) is unchanged
```

The only requirement is an adapter layer to match the encoder's output dimension (128) to the model's hidden dimension (d_model). This is a single linear layer or small MLP — negligible parameter overhead.

### Compatibility by Model Family

| Model | d_model | Tokenizer | Position Encoding | Injection Difficulty |
|---|---|---|---|---|
| GPT-2 | 768 | BPE (tiktoken) | Learned absolute | Easy (current impl) |
| LLaMA / LLaMA 3 | 4096 | SentencePiece | RoPE | Easy — RoPE applied after embedding, no conflict |
| Mistral | 4096 | SentencePiece | RoPE | Same as LLaMA |
| Phi-3 | 3072 | tiktoken | RoPE | Easy |
| GPT-Neo / GPT-J | 2048-4096 | BPE | Learned / RoPE | Easy |
| Gemma | 2048-3072 | SentencePiece | RoPE | Easy |

The injection point is always pre-attention, so architectural differences downstream (GQA, sliding window attention, gated MLPs, RMSNorm vs LayerNorm) do not affect compatibility.

### What Changes Per Model

1. **Adapter dimension:** `nn.Linear(128, d_model)` — trivial change
2. **Tokenizer integration:** The `process_text_with_numbers()` regex is tokenizer-agnostic (it operates on raw text before tokenization). The NUM_TOKEN_ID just needs to be set to an unused token ID in the target model's vocabulary
3. **SME output tokens:** Token IDs for the sign/exponent/digit/end tokens need to be mapped to unused IDs in the target vocabulary. Most models have padding in their vocab that can be repurposed
4. **Training strategy:** At scale (8B+ params), full fine-tuning is impractical. Use LoRA/QLoRA for the transformer weights and only fully train the encoder + adapter

### Scaling Considerations

| Scale | Strategy | Estimated Cost |
|---|---|---|
| ~124M (GPT-2 small) | Full fine-tune, optionally init from pretrained | Single GPU, hours |
| ~1B (LLaMA-like) | Full fine-tune or LoRA | 4-8 GPUs, hours-days |
| ~8B (LLaMA 3.1 8B) | LoRA for transformer, full train for encoder+adapter | 4-8 A100s, days |
| ~70B | QLoRA for transformer, full train for encoder+adapter | 8+ A100s, days-weeks |

### Key Advantage Over Alternatives

Standard LLMs represent numbers as sequences of digit/subword tokens:
- `12345` → `["123", "45"]` (2 tokens, BPE-dependent, no numerical meaning)
- `12346` → `["123", "46"]` (2 tokens, nearly identical number but different token sequence)

The NumberEncoder provides:
- **Continuity:** nearby numbers get similar embeddings (12345 ≈ 12346 in embedding space)
- **Scale awareness:** the encoder captures magnitude, sign, and fine-grained structure in a single vector
- **Tokenizer independence:** the same encoder works regardless of how the tokenizer would have split the number

This is especially valuable for tasks requiring numerical reasoning (arithmetic, comparison, sorting, scientific computation) where the arbitrary tokenization of digits actively hurts LLM performance.

### Pretrained Initialization Path

For maximum impact, combine the number encoder with pretrained transformer weights:

1. Load pretrained LLM (e.g., `LLaMA-3.1-8B`)
2. Add NumberEncoder (load from pretrained encoder checkpoint)
3. Add adapter layer (randomly initialized, small)
4. Fine-tune on numerical tasks with lower LR for pretrained weights, higher LR for adapter
5. The pretrained transformer already understands language; it just needs to learn to use the continuous number representations

This is the most promising direction — the pretrained model brings linguistic competence, and the number encoder brings numerical competence. Neither has to learn the other's domain from scratch.
