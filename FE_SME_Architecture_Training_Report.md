# FE-SME vs Base GPT-2: Architecture, Training, and Data Report

## 1. Scope
This report documents the current project setup for:
- the FE-SME model (`np_emb_torch.py` + `GPT2/fe/model.py` + `GPT2/fe/train.py`)
- the base model (`GPT2/base/base.py` + `GPT2/base/train.py`)
- the synthetic numerical training data pipelines (`GPT2/generate_data.py`, `GPT2/generate_data_base.py`)

It also anchors key points with concrete run artifacts from:
- `GPT2/slurm_logs/generate_data_77759.log`
- `GPT2/slurm_logs/generate_data_base_77769.log`
- `GPT2/slurm_logs/gpt2_sme_77765.log`
- `GPT2/slurm_logs/gpt2_base_77770.log`
- `GPT2/slurm_logs/validate_sme_77768.log`
- `GPT2/slurm_logs/validate_base_77778.log`

---

## 2. FE-SME System Overview

### 2.1 Core idea
FE-SME combines:
1. a pretrained continuous number encoder (`np_emb_torch.py`) used at **input** number positions
2. a GPT-2 style language model (`GPT2/fe/model.py`)
3. an SME tokenization scheme for **output** numbers (Sign + Exponent + Digits), so decoding is done through standard next-token prediction

The FE path is therefore:
- Input number in prompt text -> `<NUM>` placeholder token + numeric side-channel value
- `<NUM>` embedding replaced by adapter(NumberEncoder(value))
- Output number predicted as normal text tokens, but restricted to SME token vocabulary for numeric tasks

There is no separate numeric regression head in FE training; loss is plain token cross-entropy.

---

## 3. Number Embedding Subsystem (`np_emb_torch.py`)

### 3.1 Encoder architecture
`NumberEncoder` maps scalar `x ∈ R` to 128D embedding with mixed analytic channels:

1. Fourier channel: 32 frequencies with sin/cos + amplitude damping  
   - output dims: 64
2. Log magnitude channel: `log(|x| + eps) / log(scale)`  
   - output dims: 1
3. Smooth sign channel: `tanh(alpha * x)`  
   - output dims: 1
4. Polynomial basis (degrees 1..5), per-sample normalized  
   - output dims: 5

Raw channel concat dimension: `64 + 1 + 1 + 5 = 71`.

Projection + normalization:
- linear projection: `71 -> 127`
- one reserved dimension: `log_norm = log(||projected|| + 1e-8)`
- manual LayerNorm over the 127 projected dimensions
- final embedding: `[normalized_127, log_norm]` -> 128D

### 3.2 Decoder architecture
`NumberDecoder` is an MLP with residual skip:
- `128 -> 192 -> 192 -> 2`
- GELU nonlinearities
- skip projection `128 -> 2` (no bias), initialized to zero

Outputs:
- `log_mag` (clamped to [-14, 14])
- `sign_logit`
- reconstruction formula: `recon = tanh(sign_logit) * exp(log_mag)`

### 3.3 Training objective
`compute_loss` combines five terms:
- signed-log MSE
- sign BCE (weight 0.1)
- log-magnitude MSE (weight 0.3)
- relative MSE (phase-2 ramped; max weight 0.3)
- embedding spread loss (cosine decorrelation; weight 0.05)

### 3.4 Number-encoder training setup
Default training in `train_model`:
- steps: 500,000
- batch size: 512
- optimizer: AdamW (`lr=5e-4`, betas `(0.9, 0.999)`, weight decay `1e-5`)
- gradient clipping: 1.0
- LR schedule: warmup (2,000 steps) + cosine decay
- relative-loss phase ramp: 40% to 50% of training

Sampling distribution (`sample_training_numbers`, per batch):
- 40% positive log-uniform via `exp(U(-14,14))`
- 40% negative log-uniform
- 10% near zero uniform in `[-0.01, 0.01]`
- remainder as random integers in `[-1000, 1000)`

Checkpoint used by FE runs:
- `/tmpdir/m24047brmn/numbers/checkpoints/np_emb_v8_500k_model.pt`

---

## 4. FE GPT-2 Model (`GPT2/fe/model.py`)

### 4.1 Base transformer backbone
Backbone is nanoGPT-style decoder-only GPT:
- token embedding + position embedding
- stacked causal self-attention + MLP blocks
- final layer norm + tied LM head

Per-block skip connections are the standard residual form used in `GPT2/fe/model.py`:
- `x = x + attn(ln_1(x))`
- `x = x + mlp(ln_2(x))`

So each transformer block has two residual paths (attention and MLP), with pre-norm LayerNorm before each sublayer.

Attention path details in this implementation:
- QKV projection with causal masking
- Flash attention when available (`scaled_dot_product_attention`), fallback to explicit masked softmax otherwise
- output projection (`c_proj`) followed by residual dropout

Initialization detail also preserved from nanoGPT:
- `c_proj.weight` uses scaled std `0.02 / sqrt(2 * n_layer)` to stabilize residual accumulation in deep stacks.

Training config used in FE run:
- `n_layer=12`, `n_head=8`, `n_embd=256`, `block_size=256`, `bias=False`, `dropout=0.0`

### 4.2 FE additions
Two FE components are added:
1. **Frozen NumberEncoder** loaded from checkpoint
2. **Trainable adapter**: `Linear(128->n_embd) + GELU + Linear(n_embd->n_embd)`

At forward pass:
- identify `<NUM>` positions via `num_mask`
- gather numeric values from `num_values`
- run frozen encoder in no-grad mode
- project with adapter
- replace token embeddings at `<NUM>` positions with projected numeric embeddings

Everything else is standard GPT forward.

### 4.3 Output vocabulary and SME tokens
Special IDs in FE path:
- `NUM_TOKEN_ID = 50257` for input placeholder
- SME tokens in `50258..50280` (23 tokens total):
  - sign: 2 tokens
  - exponent: 11 tokens (`E-5..E5`)
  - digits: 10 tokens (`D0..D9`)

Model vocab is padded to 50304.

### 4.4 Parameter footprint
From `gpt2_sme_77765.log`:
- reported model size: `22.42M` trainable parameters
- optimizer groups:
  - decayed: 22,478,848 params
  - non-decayed: 6,912 params

Compared with base (`22.32M`), FE is larger mainly due to the adapter (~98k parameters); frozen number-encoder parameters are excluded from trainable counts.

---

## 5. FE Training Pipeline (`GPT2/fe/train.py`)

### 5.1 Data interface
FE training reads dual memmaps:
- `{split}.bin` (uint16 token IDs)
- `{split}_nums.bin` (float32 numeric values aligned by token position)

Batch returns:
- `x`: input token IDs
- `y`: shifted targets
- `nv`: numeric values
- `nm`: mask where `x == NUM_TOKEN_ID`

### 5.2 Optimization and schedule
Defaults used in run:
- AdamW, `lr=6e-4`, `weight_decay=0.1`, betas `(0.9, 0.95)`
- `max_iters=15000`
- cosine decay with warmup 2000, `min_lr=6e-5`
- global grad accumulation equivalent: `5*8`, divided across DDP ranks
- mixed precision (bf16 when available), `torch.compile=True`

### 5.3 Distributed run shape
From `GPT2/run_fe_train.slurm` and run log:
- 4 nodes, 2 GPUs/node (8 GPUs total)
- torchrun DDP
- tokens per iteration: `122,880`

### 5.4 Diagnostics
`fe/train.py` logs:
- standard loss
- gradient norms (total/transformer/adapter)
- `%<NUM>` tokens per batch
- detailed SME token accuracies:
  - overall, sign, exponent, digit
  - per-digit slot d0/d1/d2

Observed in `gpt2_sme_77765.log`:
- early SME accuracy low at iter 0 (`overall 0.011`)
- reaches high values by mid/late training (often ~0.9+ overall in diagnostics)

### 5.5 FE convergence snapshot
From `gpt2_sme_77765.log`:
- step 0: train 10.8180, val 10.8102
- step 5000: train 0.2491, val 0.2492
- step 10000: train 0.2243, val 0.2294 (best in this run)
- step 15000: train 0.2066, val 0.2449

Interpretation:
- strong convergence by 5k-10k
- mild overfitting from 10k to 15k

---

## 6. Base GPT-2 Baseline

### 6.1 Architecture
`GPT2/base/base.py` is a standard GPT-2 style decoder LM with:
- token + position embeddings
- causal self-attention and MLP blocks
- tied LM head

No number encoder, no adapter, no dual numeric stream, and no SME-specific path.

In other words, the base baseline is just vanilla GPT-2 with run-time config overrides.

### 6.2 Training path
`GPT2/base/train.py` uses same core training recipe and same model shape used for fair comparison:
- `n_layer=12`, `n_head=8`, `n_embd=256`, `block_size=256`
- `dropout=0.0`, `bias=False`, `vocab_size=50304`
- same optimizer and LR schedule family
- but input/output numbers are plain text tokens only

Run (`gpt2_base_77770.log`) convergence:
- step 0: train 10.7871, val 10.7892
- step 5000: train 2.1847, val 2.2140
- step 10000: train 2.0629, val 2.3210
- step 15000: train 1.9667, val 2.4167

Interpretation:
- model learns, but overfits strongly after ~5k
- numeric generation remains much less stable than FE-SME

---

## 7. Training Data Design

### 7.1 Task families
Both FE and base synthetic generators use the same 14 tasks:

Reasoning/text-output tasks (weight 2 each):
- CMP, GT, IS_POS, IS_SORTED, CHECKSORT, CHECKADD, SUM_CMP

Numeric-output tasks (weight 1 each):
- SORT, ADD, SUB, MIN, MAX, SUM, COUNT

So reasoning tasks are sampled ~2x more often than numeric-output tasks per generator entry.

### 7.2 Number sampling policy
Per sampled number:
- mixture over scales (small, medium, wide range)
- optional decimals (1-4 places)
- optional negatives

Run config used (`generate_data_77759.log`, `generate_data_base_77769.log`):
- `n_train=1,000,000`, `n_val=5,000`
- `block_size=256`
- sequence lengths roughly 2-15 for list tasks
- `number_range=1000`
- negatives allowed, floats allowed
- seed 42

Because seed and sampling logic match, FE and base task distributions are the same.

### 7.3 FE/SME data format (`GPT2/generate_data.py`)
Input side:
- numbers in prompt text replaced with `<NUM>` token (50257)
- actual numeric values stored in parallel `*_nums.bin`

Output side:
- numeric targets encoded as fixed-length SME token chunks
- each number = 5 tokens: sign + exponent + 3 mantissa digits

Storage:
- `train.bin`, `train_nums.bin`, `val.bin`, `val_nums.bin`, `meta.pkl`

Observed dataset stats (`generate_data_77759.log`):
- train: 22,148,608 tokens
- val: 112,640 tokens
- train `<NUM>` positions: 4,715,873
- val `<NUM>` positions: 23,725
- train SME output tokens: 3,456,000 (15.6%)
- val SME output tokens: 17,850 (15.8%)

### 7.4 Base data format (`GPT2/generate_data_base.py`)
Input/output side:
- numbers remain plain GPT-2 text tokens everywhere
- no `<NUM>` placeholder, no numeric side-channel

Storage:
- `train.bin`, `val.bin`, `meta.pkl`

Observed dataset stats (`generate_data_base_77769.log`):
- train: 22,894,848 tokens
- val: 115,712 tokens

Base token count is slightly larger because numbers are serialized as variable-length text instead of compressed `<NUM>` + side-channel.

---

## 8. Validation Snapshot (Current Runs)

### 8.1 FE-SME validation (`validate_sme_77768.log`)
- CE loss: 0.189661
- perplexity: 1.208840
- SME token accuracy:
  - overall 0.9574
  - sign 0.9927
  - exponent 0.9947
  - digit 0.9331
- number decode:
  - invalid rate 0.0067
  - exact value rate 0.8359
  - MAE 10.98
  - RMSE 214.68

### 8.2 Base validation (`validate_base_77778.log`)
- CE loss: 2.369499
- perplexity: 10.692031
- output exact match rate: 0.8586 (all tasks mixed)
- numeric exact example rate: 0.6368
- numeric MAE: 721.28
- numeric RMSE: 10,740.04
- many large-magnitude numeric corruption cases in worst examples

---

## 9. Practical Interpretation

1. FE-SME is not "just GPT-2 with extra tokens"; it is a hybrid:
- continuous numeric embedding injection at input via pretrained encoder + adapter
- discrete structured numeric decoding via SME token language modeling

2. The base model can learn many classification-style reasoning tasks well, but plain-text numeric generation is far less stable under this setup.

3. FE-SME materially improves numeric decoding reliability and error scale on this dataset.

4. Tradeoff: SME output is structured/quantized by design (fixed sign/exponent/3-digit mantissa representation), which improves robustness but constrains numeric precision format compared with unrestricted continuous-text regression.

---

## 10. Reproducibility Pointers

Key files:
- Number embedding model: `np_emb_torch.py`
- FE model: `GPT2/fe/model.py`
- FE training: `GPT2/fe/train.py`
- FE data generation: `GPT2/generate_data.py`
- Base model: `GPT2/base/base.py`
- Base training: `GPT2/base/train.py`
- Base data generation: `GPT2/generate_data_base.py`
- FE train launcher: `GPT2/run_fe_train.slurm`
- Base train launcher: `GPT2/run_base_train_numtasks.slurm`

Key logs:
- FE train: `GPT2/slurm_logs/gpt2_sme_77765.log`
- Base train: `GPT2/slurm_logs/gpt2_base_77770.log`
- FE data: `GPT2/slurm_logs/generate_data_77759.log`
- Base data: `GPT2/slurm_logs/generate_data_base_77769.log`
- FE val: `GPT2/slurm_logs/validate_sme_77768.log`
- Base val: `GPT2/slurm_logs/validate_base_77778.log`
