# Surface Version Changes

This document summarizes the changes made for the surface-oriented numeric
model/training path. The goal of this version is to align numeric training and
numeric generation more closely with exact rendered surface text, instead of
only learning an exponent-style numeric representation under teacher forcing.

## Motivation

The earlier analytic path was good at teacher-forced numeric decoding but was
misaligned with exact autoregressive generation:

- the old decoder was optimized around `sign + exponent + digits`
- the benchmark ultimately scores rendered output strings
- generated `<NUM>` tokens were rendered back to text later, so surface-form
  fidelity mattered a lot
- exact match stayed poor even when teacher-forced numeric losses looked good

The surface version keeps the old exponent path available for comparison, but
adds a new surface-oriented path that models numbers closer to the way they are
rendered and evaluated.

## High-Level Summary

The surface version adds:

- a new `numeric_output_mode='surface'`
- canonical numeric targets represented as `sign + scale + length + digits`
- structured numeric decoding, rendering, and feedback embeddings
- a small shared numeric trunk before the surface heads
- same-number consistency loss across repeated mentions
- short rollout training with structured generated-number feedback
- rollout self-consistency loss on later repeated mentions
- a synth-only zero-shot benchmark for the adapted surface path
- launcher scripts that train, resume, merge, and benchmark the surface model

The legacy exponent-style decoder remains intact and can still be selected via
config.

## Files Added

The following new files were added for the surface path:

- `model_analytic_surface.py`
- `numeric_surface.py`
- `generate_data_analytic_surface.py`
- `generate_synth_math_surface.py`
- `train_analytic_surface.py`
- `train_tulu_lora_surface.py`
- `benchmark_synth_surface.py`
- `merge_tulu_lora_surface_checkpoint.py`
- `run_surface_pipeline.sh`
- `run_surface_resume_from_s1_best.sh`

## 1. Surface Numeric Decoder

Implemented in `model_analytic_surface.py`.

### New config fields

The surface model adds config fields such as:

- `numeric_output_mode`
- `surface_max_digits`
- `surface_scale_min`
- `surface_scale_max`
- `surface_embed_dim`
- `numeric_trunk_hidden`
- `same_number_consistency_lambda`
- `same_number_digit_consistency_weight`

### Surface representation

Numbers are represented canonically as:

- `sign`
- `scale`
- `length`
- `digits`

Examples:

- `85355 -> sign=+, scale=0, length=5, digits=85355`
- `123.45 -> sign=+, scale=2, length=5, digits=12345`
- `0.0072 -> sign=+, scale=4, length=2, digits=72`

This representation is closer to exact rendered text than the old
`sign + exponent + fixed mantissa digits` representation.

### Decoder heads

The surface path predicts:

- `num_decoder_sign`
- `num_decoder_scale`
- `num_decoder_len`
- `num_decoder_digits`

The old exponent head is still present for legacy mode.

## 2. Structured Numeric Helpers

Implemented across `model_analytic_surface.py` and `numeric_surface.py`.

### Canonical rendering helpers

`numeric_surface.py` provides shared helpers for:

- canonical decimal string normalization
- converting values into surface components
- converting surface components back into canonical text
- converting surface components into row/tensor form
- decoding stored rows back into surface components

This makes data generation, training, generation, and benchmarking all use the
same numeric convention.

### Structured decode/render split

The model no longer treats number decoding as "hidden state -> final string"
only. The surface version adds:

- structured decode from hidden states
- canonical rendering for display/evaluation
- structured numeric feedback rows for generation

The wrapper that returns strings remains for compatibility, but internally the
generation path now preserves structured numeric state much longer.

## 3. Structured Numeric Feedback Embedding

Implemented in `model_analytic_surface.py`.

Generated or provided surface numbers are embedded through:

- sign embedding
- scale embedding
- length embedding
- digit embeddings
- a small projection MLP (`num_surface_feedback`)

This is used when feeding numeric state back into the model at `<NUM>` input
positions. The old scalar-value numeric adapter path still exists for legacy
analytic values.

This means generated numbers are no longer forced to collapse immediately to a
single float before being fed back into the model.

## 4. Shared Numeric Trunk

Implemented in `model_analytic_surface.py`.

To improve exact digit prediction, a small shared numeric trunk was added before
the surface heads:

- `hidden_num`
- `num_decoder_trunk(hidden_num)`
- residual-style addition before prediction heads

This was added because the shallow direct heads were learning sign/scale well,
but exact digits remained the clear bottleneck.

## 5. Surface Numeric Loss

Implemented in `model_analytic_surface.py`.

The surface numeric loss replaces the old exponent/mantissa objective with:

- sign cross-entropy
- scale cross-entropy
- length cross-entropy
- masked digit cross-entropy on active digit positions only

Earlier digits receive slightly higher weight through `digit_decay`.

This loss is better aligned with exact rendered numbers than the previous
numeric closeness objective.

## 6. Same-Number Consistency Loss

Implemented in `model_analytic_surface.py`.

The surface path adds a same-number consistency term over repeated mentions of
the same canonical target number within one sample.

Mechanically:

- repeated identical surface target rows are grouped per sample
- the predicted sign/scale/length/digit distributions for those mentions are
  compared
- disagreement is penalized

This is intended to reduce the failure mode where the model emits plausible
numbers locally but fails to preserve exact numeric identity when the same
quantity reappears later.

The main knobs are:

- `same_number_consistency_lambda`
- `same_number_digit_consistency_weight`

## 7. Surface Data Pipeline

Implemented in:

- `generate_data_analytic_surface.py`
- `generate_synth_math_surface.py`

### New supervision tensors

The surface data pipeline writes surface-aligned supervision tensors:

- `train_surface.bin`
- `val_surface.bin`
- `test_surface.bin`

These contain the canonical surface row representation aligned to `<NUM>`
positions.

### Analytic adapted synthetic data

The synthetic arithmetic generator writes surface supervision for the adapted
split as well, so stage 2 can train on assistant-side `<NUM>` targets with the
new decoder.

## 8. Stage 1 Surface Training

Implemented in `train_analytic_surface.py`.

Stage 1 now:

- loads surface supervision
- trains the surface decoder/adapter path
- logs `sign`, `scale`, `len`, `digit`, and `consistency`
- prints the real canonical target text in sample eval

The old broken sample-eval debug print that showed `target_val=0` for many
surface targets was fixed. It now renders the actual target surface row.

## 9. Stage 2 Surface LoRA Training

Implemented in `train_tulu_lora_surface.py`.

Stage 2 starts from a Stage 1 surface checkpoint and applies LoRA to attention
layers while keeping the surface numeric path active.

### Parameter groups

Training is explicitly separated into:

- LoRA parameters
- numeric adapter parameters
- numeric decoder / surface-feedback parameters

Each group has its own learning-rate scale.

### Decoder warmup

Stage 2 includes a warmup period where LoRA is frozen and only the numeric
adapter/decoder path learns first. This gives the surface decoder a chance to
stabilize before the transformer hidden states start moving through LoRA.

### Surface diagnostics

Stage 2 logs:

- `text_loss`
- `num_loss`
- `sign`
- `scale`
- `len`
- `digit`
- `consistency`
- per-group gradient norms
- per-group learning rates
- `<NUM>` token prediction

## 10. Structured Rollout Training

Implemented in `train_tulu_lora_surface.py`.

The surface path adds short rollout training:

- take a prefix
- generate a few greedy steps
- feed generated structured numbers back into the model
- append a short future suffix
- compute future loss under self-generated numeric history

This is intended to reduce the train/inference mismatch of pure teacher
forcing.

## 11. Rollout Self-Consistency Loss

Implemented in `train_tulu_lora_surface.py`.

Rollout training also includes a second consistency mechanism:

- if a number is generated during rollout
- and the same gold quantity appears later in the future suffix
- the later position is trained to stay consistent with the generated numeric
  state

This is logged through:

- `rollout: loss=...`
- `gen_steps`
- `future_steps`
- `gen_nums`
- `consistency`
- `consistency_targets`

`consistency_targets` counts how many later numeric positions in the future
suffix can be tied back to an earlier generated number during rollout.

## 12. Surface Benchmark

Implemented in `benchmark_synth_surface.py`.

This benchmark is:

- synth-only
- zero-shot
- base vs adapted

It avoids the TULU/few-shot benchmarking path and focuses directly on the
surface numeric behavior you actually care about.

### Logged generation metrics

The surface benchmark records:

- exact match
- full exact match
- numeric accuracy
- MAE
- digit accuracy
- scale accuracy
- length accuracy
- copied-number consistency
- first wrong digit position

It also stores detailed per-example generations for comparison.

## 13. Merge Script

Implemented in `merge_tulu_lora_surface_checkpoint.py`.

This merges a surface LoRA checkpoint into a standalone checkpoint for
benchmarking and deployment.

## 14. Launcher Scripts

### `run_surface_pipeline.sh`

Full pipeline launcher for:

- Stage 1 surface data
- Stage 1 surface training
- Stage 2 synth surface data
- Stage 2 surface LoRA training
- merge best checkpoint
- synth zero-shot benchmark

Important behavior:

- Stage 1 runs for 15k steps
- Stage 2 uses `ckpt_best.pt` from Stage 1
- Stage 2 is currently configured for 8k steps
- Stage 2 SLURM time limit is 24 hours

### `run_surface_resume_from_s1_best.sh`

Resume launcher that skips Stage 1 and starts from the current Stage 1 best
checkpoint. It submits:

- Stage 2 surface training
- benchmark after Stage 2 succeeds

This is useful when Stage 1 is already trained or when you stop Stage 1 early
and want to continue from the current `ckpt_best.pt`.

## 15. Current Known Behavior

From the current training runs:

- Stage 1 learns number routing and surface structure well
- Stage 2 improves teacher-forced numeric losses substantially
- `<NUM>` token prediction is now near-perfect in the good S2 regime
- exact digit prediction improved a lot relative to the earlier analytic path

Current caveats:

- rollout self-consistency is still sparse in diagnostics because
  `consistency_targets` is often zero
- the main signal is still coming from the teacher-forced objective
- final exact-match quality still needs to be verified by the synth zero-shot
  benchmark, not inferred only from training loss

## 16. Intended Outcome

The surface path is meant to test a cleaner hypothesis:

- exact match should emerge from a better-aligned numeric representation
- numbers should be modeled closer to their actual rendered surface form
- generated numbers should be fed back structurally, not as a lossy scalar only

This version does not "hack" exact match with special-case formatting logic.
Instead, it changes the representation, loss, and generation loop so exact
surface generation is a more natural consequence of the model's training
objective.
