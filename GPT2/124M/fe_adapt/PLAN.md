# fe_adapt: LLaVA-Style NumberEncoder Integration

## Core Idea

Treat numbers as a **new modality** (like images in LLaVA) and plug the pretrained NumberEncoder into a pretrained base GPT-2 124M model. The encoder is already trained to produce rich numerical representations — we just need to teach the transformer to use them.

## Architecture

```
Input text: "ADD: 1234, 5678 →"

                  ┌─────────────────────────────┐
  BPE tokens:     │ ADD  :  <NUM>  ,  <NUM>  →  │
                  └──────────┬──────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
         wte(token)   NumberEncoder(val)        │
              │         128-dim                 │
              │              │                  │
              │        Adapter MLP              │
              │     128 → 768 → 768             │
              │              │                  │
              │    norm_match + blend(β)        │
              │              │                  │
              └──────┬───────┘                  │
                     │                          │
              blended embedding                 │
                     │                          │
              ┌──────▼──────────────────────────┘
              │   Pretrained GPT-2 124M          │
              │   (12 layers, 768 dim)           │
              └──────────────┬──────────────────┘
                             │
                        text output
```

## Three-Stage Training

### Stage 1: Adapter Alignment (Projection Only)

**Goal**: Learn the adapter MLP that maps NumberEncoder output into the transformer's embedding space.

**What's frozen**: Everything except the adapter MLP
- Frozen: GPT-2 transformer (all 124M params)
- Frozen: NumberEncoder (7.4K params, pretrained)
- **Trainable: Adapter MLP only** (~590K params for 128→768→768)

**Training details**:
- Data: Same numerical tasks (ADD, SUB, MIN, MAX, SORT, etc.)
- Input: `<NUM>` tokens with encoder injection
- Output: Plain text BPE
- Blend schedule: β ramps 0→1 over ~5K iters (faster ramp since transformer is stable)
- LR: ~1e-3 for adapter (higher since only training projection)
- Duration: ~5-10K iters (adapter converges fast)

**Rationale**: Like LLaVA Stage 1, this teaches the projection layer to map numerical representations into the "language" the frozen transformer already understands. The transformer's existing text processing stays intact.

**Success criteria**: Val loss starts decreasing, adapter output norms roughly match base `<NUM>` embedding norms.

### Stage 2: LoRA Fine-Tune

**Goal**: Allow the transformer to learn to *use* the numerical representations, without catastrophic forgetting.

**What's frozen/trained**:
- Frozen: NumberEncoder (always frozen — it's the pretrained "sensor")
- **Trainable: LoRA adapters on transformer** (rank 8-16, on Q/V projections)
- **Trainable: Adapter MLP** (continue training)

**Training details**:
- LoRA rank: 8 or 16 (adds ~600K-1.2M params)
- LR: ~2e-4 (lower than Stage 1)
- Blend: β=1.0 (fixed, adapter already aligned)
- Duration: ~20-30K iters
- Total trainable params: ~1.2-1.8M (vs 124M total)

**Rationale**: Like LLaVA Stage 2, this lets the transformer learn task-specific reasoning with numerical inputs while preserving its pretrained text capabilities. LoRA constrains the update to a low-rank subspace, preventing catastrophic forgetting.

**Success criteria**: Significant improvement on numerical tasks (especially SORT, multi-number tasks) while maintaining text coherence.

### Stage 3: Full Fine-Tune (Optional)

**Goal**: Squeeze out maximum performance by unfreezing everything.

**What's trained**: All parameters
- NumberEncoder (7.4K) — optional, may keep frozen
- Adapter MLP (~590K)
- Full transformer (124M)

**Training details**:
- LR: ~1e-5 (very low to avoid catastrophic forgetting)
- Adapter LR scale: 0.2x (adapter is already well-tuned)
- Encoder LR scale: 0.05x (encoder is pretrained, minimal updates)
- Duration: ~5-10K iters (short, just refinement)

**Rationale**: Final polish. Only do this if Stage 2 shows the approach works but needs more capacity.

## Key Design Decisions

### Blend Schedule
- Stage 1: Cosine ramp β: 0→1 over warmup+ramp iters
- Stage 2+: Fixed β=1.0

### Norm Matching
- Always enabled — ensures adapter outputs are on the same scale as base embeddings
- Critical for stability when blending with pretrained weights

### LoRA Configuration (Stage 2)
```python
lora_config = {
    'r': 8,              # rank
    'alpha': 16,         # scaling factor
    'target_modules': ['c_attn'],  # Q, K, V projections
    'dropout': 0.05,
}
```

### Data Format
- Same as 124M/fe: `<NUM>` input tokens + text BPE output
- Can reuse the same generate_data.py and prepare.py

## Implementation Checklist

- [ ] `model.py` — Add LoRA wrapper around GPT with NumberEncoder adapter
  - Load pretrained 124M/base checkpoint
  - Attach NumberEncoder + adapter (from 124M/fe or fresh)
  - Add LoRA layers to transformer attention
  - Freeze/unfreeze control per stage
- [ ] `train.py` — Multi-stage training script
  - Stage selector (1/2/3) via config
  - Freeze masks per stage
  - Stage 1→2 checkpoint handoff
  - Same diagnostics as 124M/fe (blend stats, grad norms, output accuracy)
- [ ] `prepare.py` — Reuse from 124M/fe (identical format)
- [ ] `generate_data.py` — Reuse from 124M/fe (identical format)
- [ ] `validate_checkpoint.py` — Reuse from 124M/fe

## Experiment Plan

1. **Train 124M/base** on numerical tasks → baseline checkpoint
2. **Stage 1**: Load base checkpoint, attach encoder+adapter, train adapter only
3. **Evaluate Stage 1**: Should show some improvement on numerical tasks
4. **Stage 2**: Load Stage 1 checkpoint, add LoRA, train adapter+LoRA
5. **Evaluate Stage 2**: Compare against 124M/base and 124M/fe (trained from scratch)
6. **Stage 3** (if needed): Full fine-tune from Stage 2 checkpoint

## Key Question: fe_adapt vs fe (from scratch)

The central experiment is comparing:
- **124M/fe**: Train NumberEncoder-equipped GPT-2 from scratch (all params from random init)
- **124M/fe_adapt**: Plug NumberEncoder into pretrained base, fine-tune with LoRA

If fe_adapt matches or beats fe with far less training, it validates the "numbers as a modality" thesis and shows the approach can scale to any pretrained LLM.

## Dependencies

- Pretrained 124M/base checkpoint (from training 124M/base)
- Pretrained NumberEncoder v9 checkpoint (`np_emb_v9_500k_model.pt`)
- LoRA implementation (can use simple custom implementation or `peft` library)
