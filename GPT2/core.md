 The Core Thesis

  Numbers are a modality. Just as images are a modality that BPE handles poorly (you wouldn't
  OCR an image into text and feed that to a language model), numbers are a modality that BPE
  handles poorly. BPE fragments "21530000000000" into arbitrary chunks like ["215", "300",
  "000", "000", "00"] — destroying the magnitude, ordering, and arithmetic structure that
  matters.

  LLaVA's insight was: don't force images through text tokenization. Give the model a
  pretrained visual encoder and teach it to use those features. Your insight is the same for
  numbers.

  LLaVA Architecture (for reference)

  LLaVA training has two stages:

  Stage 1: Feature Alignment (freeze CLIP, freeze LLM, train ONLY the adapter)
    - Data: 600K image-caption pairs
    - Goal: adapter learns to project CLIP features into the LLM's
      embedding manifold
    - The LLM doesn't change — the adapter learns to "speak its language"
    - Cheap: ~24 hours on 8× A100s

  Stage 2: Visual Instruction Tuning (freeze CLIP, LoRA on LLM, train adapter)
    - Data: 150K visual Q&A + instruction pairs
    - Goal: LLM learns to reason about visual tokens
    - LoRA keeps most weights frozen, prevents catastrophic forgetting
    - The LLM adapts to the new modality without losing language ability

  The key insight: the encoder never changes. CLIP was pretrained separately and stays frozen.
  Only the adapter and (via LoRA) the LLM adapt.

  Your Architecture: NumericalLLM

  Direct translation:

  Stage 1: Number Alignment (freeze NumberEncoder, freeze LLM, train ONLY adapter)
    - Data: synthetic numerical tasks (your existing generator)
    - Goal: adapter learns to project 128-dim number embeddings into
      the LLM's embedding manifold
    - This IS your blend schedule — but done with the LLM frozen
    - Very cheap: the NumberEncoder is 7.4K params, adapter is ~690K

  Stage 2: Math Instruction Tuning (freeze encoder, LoRA on LLM, train adapter)
    - Data: math problems, numerical reasoning, mixed with language
    - Goal: LLM learns to USE the numerical features for reasoning
    - LoRA (rank 16-64) on attention projections + MLP
    - Mix in general text to prevent forgetting

  What Makes Numbers Different from Vision

  This is where the research gets interesting. Numbers are a better modality to adapt than
  images:

  1. Numbers compress, images expand.

  LLaVA injects 576 visual tokens per image — massively expanding the sequence. Your encoder
  does the opposite: "21530000000000" goes from 5 BPE tokens to 1 token. You're saving context
  length. For a financial report with 50 numbers, you might save 100+ positions. This is a
  feature, not a cost.

  2. The encoder is tiny and analytically grounded.

  CLIP needs 400M params and billions of image-text pairs. Your NumberEncoder needs 7.4K params
   and is constructed from mathematical principles (Fourier features, log-magnitude, residue
  system). It can be pretrained in hours on a single GPU. This means anyone can reproduce it.

  3. Numbers appear inline, not as a prefix.

  LLaVA puts visual tokens at the start of the sequence, then text follows. Numbers appear
  scattered throughout: "The population of France is 67390000 and Germany is 83200000, so the
  difference is 15810000." This is architecturally more interesting — the model must handle
  mixed text-and-number sequences where numbers can appear anywhere.

  4. The encoder provably captures the right properties.

  You can formally verify that the NumberEncoder preserves ordering (Spearman ρ = 1.0), enables
   addition recovery (R² = 0.9999), and distinguishes magnitudes. CLIP's properties are
  empirical. This is a cleaner scientific story.

  The Sequence Position Problem (and the solution)

  There's a technical challenge unique to inline number injection. When you replace 3 BPE
  tokens with 1 <NUM> token, all subsequent positions shift:

  Original:  [The, GDP, was, 215, 300, 000, in, 2023]
                                ↑positions 3-5↑   ↑pos 6↑

  With encoder: [The, GDP, was, <NUM>, in, 2023]
                                        ↑pos 4↑

  " in" moved from position 6 to position 4. For a pretrained model using RoPE, this changes
  the attention geometry.

  Three solutions (pick one):

  A. Multi-token number slots (cleanest for plug-and-play):

  Always represent each number as K tokens (e.g., K=4):
  [The, GDP, was, <NUM>, <NUM_CONT>, <NUM_CONT>, <NUM_CONT>, in, 2023]
  - First token: NumberEncoder embedding via adapter
  - Remaining K-1 tokens: learned "continuation" embeddings (or zeros, or copies)
  - Positions stay exactly where a typical BPE tokenization would put them
  - K chosen to match the median BPE token count for numbers in the training data

  This is exactly what LLaVA does — it uses a fixed number of visual tokens per image
  regardless of content.

  B. Just fine-tune through it:

  RoPE is relatively robust to moderate position shifts. With LoRA fine-tuning, the model
  adapts in a few thousand steps. Simpler to implement but less clean theoretically.

  C. Position-aware adapter:

  The adapter produces the embedding AND a "virtual position offset" that adjusts the RoPE
  rotation for subsequent tokens. Clever but complex.

  For the 124M experiment, B is fine. For the Llama paper, A is the right choice.

  The Output Problem

  The encoder helps the model understand numbers in the input. But the model still outputs
  numbers as BPE tokens. This asymmetry is fine — it's the same as LLaVA, which understands
  images but outputs only text.

  But you can push further. The value-regression aux loss I described earlier fits naturally
  here:

  Input: "what is the population difference between France (67390000)
          and Germany (83200000)?"

  Encoder handles: 67390000 → dense embedding, 83200000 → dense embedding

  Transformer processes everything, produces hidden states.

  Output pathway 1 (main): LM head → text tokens → "15810000"
  Output pathway 2 (aux):  Regression head → scalar → 15810000.0
                            L_aux = |slog(pred) - slog(15810000)|

  The aux loss is training-time only. At inference, normal text output.

  The aux loss forces the transformer's internal representation to be numerically precise,
  which makes the BPE token prediction better downstream.

  Experimental Roadmap for a Paper

  Experiment 1: Controlled comparison at 124M (train from scratch)

  Model: Base 124M
  Input: BPE
  Output: BPE
  What it proves: Baseline
  ────────────────────────────────────────
  Model: FE-TextDec 124M
  Input: NumberEncoder
  Output: BPE
  What it proves: Encoder helps with text output
  ────────────────────────────────────────
  Model: FE-TextDec 124M + aux
  Input: NumberEncoder
  Output: BPE + value aux
  What it proves: Aux loss improves precision

  Data: your synthetic tasks, range 10^6, all 14 tasks. Same data for all three.

  This is the cleanest experiment. Same model size, same data, same output format. Only the
  input encoding differs.

  Experiment 2: Plug-and-play at 124M (pretrain then adapt)

  1. Train Base 124M to convergence
  2. Freeze it, plug in NumberEncoder, train adapter only (Stage 1)
  3. LoRA fine-tune on same math data (Stage 2)
  4. Compare to Base 124M and FE-from-scratch

  This proves the approach transfers to pretrained models.

  Experiment 3: Plug into Llama/Qwen (the real test)

  1. Take Llama 3.2 1B or 3B (small enough to fine-tune on limited compute)
  2. Stage 1: adapter alignment on synthetic math data
  3. Stage 2: LoRA fine-tune on math benchmarks (GSM8K, MATH, etc.)
  4. Evaluate on standard math benchmarks AND general language benchmarks (to show no
  regression)

  This is the paper's main result.

  Experiment 4: Ablations

  - Freeze vs unfreeze encoder
  - With vs without value aux loss
  - Multi-token slots (K=1,2,4) vs single token
  - Encoder size (64, 128, 256 dims)
  - LoRA rank (8, 16, 32, 64)

  What the Paper Looks Like

  Title: something like "Numerical Modality Adaptation for Large Language Models"

  Claim: A small, analytically-constructed number encoder (7.4K params) plugged into any
  pretrained LLM via adapter alignment + LoRA fine-tuning improves mathematical reasoning by X%
   on standard benchmarks while maintaining language performance — at < 1% of pretraining cost.

  Why it's publishable:
  - Novel framing (numbers as a modality, not just a tokenization problem)
  - Principled encoder design (three lanes with provable properties)
  - Practical recipe (works on any LLM, cheap to train)
  - Strong empirical results on standard benchmarks

  The 124M experiments are your proof of concept. The Llama experiment is the headline result.
