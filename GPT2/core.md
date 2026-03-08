# Core Thesis

Numbers are a modality.

Just as images are a modality that BPE handles poorly (you would not OCR an image into text and feed that to a language model), numbers are also a modality that BPE handles poorly.

Example: `21530000000000` can be fragmented into arbitrary chunks like `"215"`, `"300"`, `"000"`, `"000"`, `"00"`, which destroys magnitude, ordering, and arithmetic structure.

LLaVA's key insight was: do not force images through text tokenization. Give the model a pretrained visual encoder and teach it to use those features.

This work applies the same principle to numbers.

## LLaVA Architecture (Reference)

LLaVA training has two stages.

### Stage 1: Feature Alignment

- Freeze CLIP.
- Freeze the LLM.
- Train only the adapter.
- Data: ~600K image-caption pairs.
- Goal: project CLIP features into the LLM embedding manifold.
- Result: the LLM does not change; the adapter learns to "speak its language."
- Cost: relatively cheap (roughly a day on 8x A100s in reported setups).

### Stage 2: Visual Instruction Tuning

- Freeze CLIP.
- Apply LoRA to the LLM.
- Train adapter + LoRA parameters.
- Data: ~150K visual Q&A and instruction pairs.
- Goal: teach the LLM to reason over visual tokens.
- Benefit: LoRA avoids full finetuning and reduces catastrophic forgetting.

The key pattern: the encoder remains frozen; adaptation happens in the projection and lightweight LLM updates.

## Your Architecture: NumericalLLM

Direct translation of the LLaVA recipe.

### Stage 1: Number Alignment

- Freeze `NumberEncoder`.
- Freeze the LLM.
- Train only an adapter.
- Data: synthetic numerical tasks (existing generator).
- Goal: project 128-dim number embeddings into the LLM embedding manifold.
- Interpretation: this is the blend schedule done with a frozen LLM.
- Cost: very low (`NumberEncoder` ~7.4K params; adapter ~690K).

### Stage 2: Math Instruction Tuning

- Freeze `NumberEncoder`.
- Apply LoRA to the LLM.
- Train adapter + LoRA parameters.
- Data: math reasoning + mixed language data.
- Goal: teach the LLM to use numerical features for reasoning.
- Practical setting: LoRA rank ~16-64 on attention projections and MLP.
- Stability: mix general text to reduce forgetting.

## Why Numbers Are Different from Vision

This is where the research case gets stronger.

### 1) Numbers compress, images expand

- LLaVA commonly injects many visual tokens per image.
- Number encoding can reduce token count.
- Example: one numeric span that becomes 1 numeric token instead of 3-5 BPE tokens.
- Effect: context savings (e.g., many numbers in long reports).

### 2) The encoder is tiny and analytically grounded

- CLIP-scale encoders are huge and data-hungry.
- `NumberEncoder` is tiny (~7.4K params).
- Built from mathematical priors (Fourier features, log-magnitude, residue system).
- Pretraining can be done quickly and reproducibly.

### 3) Numbers appear inline, not as a single prefix block

Unlike images (often encoded as a block of tokens), numbers are scattered throughout language:

`The population of France is 67390000 and Germany is 83200000, so the difference is 15810000.`

The model must handle mixed text-number sequences with number tokens anywhere.

### 4) Encoder properties can be formally validated

You can probe and verify properties such as:

- ordering preservation (Spearman rho ~1.0),
- addition recoverability (high R^2),
- magnitude separability.

This gives a cleaner scientific narrative than purely empirical feature quality.

## Sequence Position Problem

Inline number replacement changes positions.

When 3 BPE tokens are replaced by 1 `<NUM>` token, subsequent positions shift. For RoPE-based models, this alters attention geometry.

### Example

```text
Original:     [The, GDP, was, 215, 300, 000, in, 2023]
With encoder: [The, GDP, was, <NUM>, in, 2023]
```

Here, `in` moves earlier in position index.

### Solutions

#### A) Multi-token Number Slots (cleanest for plug-and-play)

Use fixed-width number slots with `K` tokens (e.g., `K=4`):

```text
[The, GDP, was, <NUM>, <NUM_CONT>, <NUM_CONT>, <NUM_CONT>, in, 2023]
```

- First token: adapter-projected `NumberEncoder` embedding.
- Remaining `K-1`: learned continuation embeddings (or tied/zero variants).
- Position behavior becomes predictable.
- Choose `K` from number-token statistics in data.

This mirrors the fixed-token visual pattern in multimodal systems.

#### B) Finetune Through Position Shift

- Keep `K=1` and rely on LoRA finetuning.
- RoPE is often robust enough for moderate shifts.
- Simpler implementation, weaker theoretical cleanliness.

#### C) Position-aware Adapter

- Adapter predicts both embedding and virtual positional adjustment.
- Most complex option.

### Practical Recommendation

- For the 124M prototype: **B** is acceptable.
- For a strong paper-grade system: **A** is cleaner.

## Output Asymmetry and Aux Value Loss

Input-side numeric encoding improves understanding, but output is still text tokens (BPE). This is normal in multimodal setups.

You can add an auxiliary numeric regression objective during training:

```text
Output path 1 (main): LM head -> text tokens (e.g., "15810000")
Output path 2 (aux):  regression head -> scalar (e.g., 15810000.0)
Loss: L_aux = |slog(pred) - slog(target)|
```

The auxiliary head is training-only. Inference remains standard text generation.

Effect: it pressures hidden states to carry numerically precise representations, improving downstream token prediction for numbers.

## Related Work: FoNE (Fourier Number Embedding)

Closest prior: FoNE (`arXiv:2502.09741`).

### FoNE Design (summary)

- **Encoding**: Fourier features with CRT-motivated periods (notably bases 2 and 5 for base-10 digit structure), scaled across powers of ten.
- **Injection**: additive at numeric positions (`regular_embeddings + fourier_embeddings`), minimal projection.
- **Decoding**: fixed Fourier digit fingerprints (0-9) used to compute per-digit logits.
- **Training**: end-to-end with a specific LLM (no separately pretrained frozen encoder).

### FoNE Results (as reported)

- Very strong long-digit arithmetic performance.
- Large data-efficiency gains vs standard baselines.

### How Your Approach Differs

1. **Structured multi-lane encoder vs raw Fourier only**
- Your lanes: Scale + Residue + Semantic.
- Enables lane-specific probing and richer behavior.

2. **Pretrained frozen encoder vs joint coupling to one LLM**
- Train once, reuse across models.
- CLIP/LLaVA-style modularity.

3. **Adapter projection vs pure additive injection**
- More expressive mapping into LLM manifold.
- Decouples encoder dimension from model embedding size.

4. **Range/generalization design choices**
- Your setup targets broader numeric coverage with explicit residue period planning.

### Required Baseline Grid (2x2)

- FoNE additive (their encoder + their injection)
- FoNE + adapter (their encoder + your injection)
- Ours additive (your encoder + additive injection)
- Ours + adapter (your full method)

This separates gains from encoding vs gains from injection strategy.

## Experimental Roadmap

### Experiment 1: Controlled Comparison at 124M (from scratch)

Common setup:

- Same model scale.
- Same synthetic data (all tasks, same numeric range).
- Same output format.

Variants:

- **Base 124M**: BPE input -> BPE output (baseline).
- **FE-TextDec 124M**: number-encoded input -> BPE output.
- **FE-TextDec 124M + aux**: number-encoded input -> BPE output + value aux loss.

What it proves:

- Isolate benefit of numeric input encoding.
- Measure additional precision benefit from aux objective.

### Experiment 2: Plug-and-play at 124M (pretrain then adapt)

1. Train Base 124M.
2. Freeze it; plug in `NumberEncoder`; train adapter only (Stage 1).
3. LoRA finetune on math data (Stage 2).
4. Compare against Base 124M and from-scratch FE model.

What it proves: transferability to pretrained checkpoints.

### Experiment 3: Plug Into Llama/Qwen

1. Select practical open model size (e.g., 1B-3B class).
2. Stage 1 adapter alignment on synthetic math data.
3. Stage 2 LoRA on benchmark math tasks.
4. Evaluate both math and general-language benchmarks.

What it proves: real-world applicability and limited language regression.

### Experiment 4: Ablations

- Freeze vs unfreeze encoder.
- With vs without value aux loss.
- Slot width `K=1,2,4` vs single-token replacement.
- Encoder width `64,128,256`.
- LoRA rank `8,16,32,64`.

## Paper Shape

### Working Title

`Numerical Modality Adaptation for Large Language Models`

### Core Claim

A small analytically constructed number encoder (~7.4K params), aligned to pretrained LLMs via adapter + LoRA, improves mathematical reasoning at low adaptation cost while preserving language capability.

### Why This Is Publishable

- Novel framing: numbers as a modality, not only tokenization.
- Principled encoder design with probeable properties.
- Practical recipe: low compute, reusable across LLMs.
- Benchmark-ready evaluation path.

### Narrative Arc

- 124M experiments establish the controlled proof of concept.
- Larger pretrained-model adaptation is the headline result.
