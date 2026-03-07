# 5-Digit Experiment Results: FE-SME vs Base GPT-2

## Setup

| | FE-SME | Base GPT-2 |
|---|--------|-----------|
| Parameters | 22.42M | 22.32M |
| Architecture | GPT-2 + NumberEncoder adapter | GPT-2 (standard) |
| Number representation | SME tokens (Sign-Mantissa-Exponent) | Plain text tokens |
| Input encoding | Frozen 128-dim encoder -> 256-dim adapter at `<NUM>` positions | Standard BPE token embeddings |
| Output decoding | Predict SME token sequence: `[SIGN] [EXP] [D0]...[Dk] [END]` | Predict text tokens (digits as characters) |
| Block size | 256 | 256 |
| Layers / Heads | 12 / 8 | 12 / 8 |
| Embedding dim | 256 | 256 |

## Training Configuration

| | FE-SME | Base GPT-2 |
|---|--------|-----------|
| Training examples | 5,000,000 | 5,000,000 |
| Validation examples | 10,000 | 10,000 |
| Max iterations | 35,000 | 35,000 |
| Learning rate | 4e-4 (adapter: 2e-4) | 4e-4 |
| Number range | [0, 100000] | [0, 100000] |
| Sig digits | 1-5 | 1-5 |
| Random seed | 42 | 42 |
| GPUs | 8 (4 nodes x 2) | 8 (4 nodes x 2) |
| Curriculum | None | None |

## Training Curves

### FE-SME (no overfitting)

| Step | Train Loss | Val Loss | Gap |
|------|-----------|---------|-----|
| 5,000 | 0.4657 | 0.4676 | 0.002 |
| 10,000 | 0.3644 | 0.3612 | -0.003 |
| 15,000 | 0.3437 | 0.3456 | 0.002 |
| 20,000 | 0.3041 | 0.3037 | 0.000 |
| 25,000 | 0.2874 | 0.2890 | 0.002 |
| 30,000 | 0.2771 | 0.2786 | 0.002 |
| **35,000** | **0.2741** | **0.2748** | **0.001** |

Val loss improved continuously to 35K. Best checkpoint at iter 35,000.

### Base GPT-2 (mild overfitting)

| Step | Train Loss | Val Loss | Gap |
|------|-----------|---------|-----|
| 5,000 | 1.9928 | 1.9980 | 0.005 |
| 10,000 | 1.9654 | 1.9746 | 0.010 |
| 15,000 | 1.9507 | 1.9754 | 0.025 |
| **20,000** | **1.9396** | **1.9561** | **0.016** |
| 25,000 | 1.9366 | 1.9707 | 0.034 |
| 30,000 | 1.9299 | 1.9619 | 0.032 |
| 35,000 | 1.9267 | 1.9664 | 0.039 |

Val loss best at iter 20,000. Mild overfitting after 20K despite 5M data.

### Comparison with 1M data runs (both overfit)

| Model | 1M best val loss | 5M best val loss | 1M overfit onset |
|-------|-----------------|-----------------|-----------------|
| FE-SME | 0.330 (iter 20K) | 0.275 (iter 35K) | ~20K |
| Base | 1.991 (iter 5K) | 1.956 (iter 20K) | ~5K |

FE-SME benefits more from additional data: no overfitting at all with 5M, while base still overfits mildly.

## Validation Results (Best Checkpoints)

### Overall Metrics

| Metric | FE-SME (iter 35K) | Base (iter 20K) |
|--------|-------------------|-----------------|
| Cross-entropy loss | 0.2296 | 1.9080 |
| Perplexity | 1.258 | 6.740 |
| Exact match rate | 74.87% | 92.56% (numeric only) |
| Invalid prediction rate | 0.70% (81/11542) | 0.03% (2/6722) |
| MAE | **373.62** | 16,773.98 |
| RMSE | **8,847.59** | 430,355.54 |
| Median abs error | 0.000 | 0.000 |
| P95 abs error | **1,000** | 0.000 |
| P99 abs error | **10,000** | 95,000 |
| Max abs error | **899,461** | 22,500,100 |

Note: CE loss is not directly comparable (different tokenization targets).

### SME Token Accuracy (FE-SME only)

| Component | Accuracy | Correct/Total |
|-----------|---------|--------------|
| Overall | 91.92% | 62,580/68,081 |
| Sign | 98.56% | 11,376/11,542 |
| Exponent | 99.51% | 11,485/11,542 |
| Digit (all) | 84.47% | 28,259/33,455 |
| END | 99.29% | 11,460/11,542 |
| d0 | 97.01% | 11,197/11,542 |
| d1 | 84.61% | 6,376/7,536 |
| d2 | 78.03% | 4,872/6,244 |
| d3 | 72.46% | 3,552/4,902 |
| d4 | 70.01% | 2,262/3,231 |

Left-to-right digit accuracy cascade: d0 (97%) > d1 (85%) > d2 (78%) > d3 (72%) > d4 (70%).

### Per-Task Exact Match Comparison

| Task | FE-SME | Base | Delta |
|------|--------|------|-------|
| COUNT | **100.0%** | 99.9% | +0.1% |
| MAX | 92.3% | **99.8%** | -7.5% |
| MIN | 82.4% | **99.9%** | -17.5% |
| SORT | 75.6% | **98.2%** | -22.6% |
| ADD | 66.8% | **91.7%** | -24.9% |
| SUB | 66.1% | **93.1%** | -27.0% |
| SUM | 34.7% | **63.7%** | -29.0% |

Base-only tasks (not present in FE-SME validation):

| Task | Base Exact |
|------|-----------|
| IS_POS | 100.0% |
| IS_SORTED | 100.0% |
| CMP | 100.0% |
| GT | 99.8% |
| SUM_CMP | 98.5% |
| CHECKSORT | 98.1% |
| CHECKADD | 97.5% |

### Per-Task MAE Comparison (Numeric Tasks)

| Task | FE-SME MAE | Base MAE | FE-SME advantage |
|------|-----------|---------|-----------------|
| COUNT | 0 | 0 | - |
| MAX | 147 | 1,888 | 12.8x |
| MIN | 141 | **0.53** | 0.004x |
| SORT | 94 | **49** | 0.5x |
| ADD | **1,452** | 20,805 | 14.3x |
| SUB | **357** | 4,548 | 12.7x |
| SUM | **1,961** | 182,960 | 93.3x |

FE-SME wins on MAE for compute tasks (ADD: 14x, SUB: 13x, SUM: 93x). Base wins on copy tasks (MIN, SORT).

### Per-Task P95 Absolute Error

| Task | FE-SME P95 | Base P95 |
|------|-----------|---------|
| COUNT | 0 | 0 |
| MAX | 0 | 0 |
| MIN | 0.001 | 0 |
| SORT | 0.001 | 0 |
| ADD | **1,500** | 0.0009 |
| SUB | **2,000** | 0 |
| SUM | **10,000** | 362,399 |

## Improvement from 1M to 5M Data

### FE-SME

| Task | 1M (20K iters) | 5M (35K iters) | Improvement |
|------|---------------|----------------|-------------|
| Exact match | 62.2% | 74.9% | +12.7% |
| Invalid rate | 2.01% | 0.70% | -1.3% |
| MAE | 1,230 | 374 | 3.3x better |
| MAX exact | 70.7% | 92.3% | +21.6% |
| ADD exact | 44.9% | 66.8% | +21.9% |
| SUB exact | 46.3% | 66.1% | +19.8% |
| SUM exact | 18.3% | 34.7% | +16.4% |
| d0 acc | 94.8% | 97.0% | +2.2% |
| d1 acc | 70.5% | 84.6% | +14.1% |
| d2 acc | 63.8% | 78.0% | +14.2% |
| d3 acc | 57.6% | 72.5% | +14.9% |
| d4 acc | 52.5% | 70.0% | +17.5% |

### Base GPT-2

| Task | 1M (20K iters, overfit) | 5M (20K iters, best) | Improvement |
|------|------------------------|---------------------|-------------|
| Numeric exact | 79.0% | 92.6% | +13.6% |
| Invalid rate | 0.36% | 0.03% | -0.33% |
| MAE | 29,098 | 16,774 | 1.7x better |
| MAX exact | 98.3% | 99.8% | +1.5% |
| ADD exact | 73.6% | 91.7% | +18.1% |
| SUB exact | 68.5% | 93.1% | +24.6% |
| SUM exact | 25.1% | 63.7% | +38.6% |

## Failure Mode Analysis

### Base GPT-2: Catastrophic Digit Duplication

The base model's worst errors show a consistent pattern of inserting extra digits:

| Input | Target | Prediction | Error |
|-------|--------|-----------|-------|
| SUM: 48915 100000 -6246 ... | 227,710 | 22,727,810 | 22,500,100 |
| SUM: 100000 100000 5030 ... | 226,680 | 22,626,680 | 22,400,000 |
| ADD: 29996 + 100000 | 130,000 | 13,000,000 | 12,870,000 |
| SUM: 14697 92640 ... | 109,980 | 11,009,080 | 10,899,100 |
| SUM: 1 0 ... -100000 41 | -99,984 | -9,999,984 | 9,900,000 |

The base model inserts extra digits into the output, producing numbers 10-100x larger than the correct answer. This is a structural limitation of text-based number representation — the model has no explicit length/magnitude constraint.

### FE-SME: Invalid (Unparseable) Predictions

FE-SME's worst errors are all invalid predictions (0.7% of outputs) where the model produces an unparseable SME token sequence. When FE-SME produces valid output, errors are bounded by the digit-position accuracy. The structured SME format prevents catastrophic magnitude errors.

## Key Findings

1. **Base wins on exact match** across all tasks, with the largest gap on compute tasks (ADD +25%, SUB +27%, SUM +29%).

2. **FE-SME wins on error magnitude** for compute tasks: 14x lower MAE on ADD, 13x on SUB, 93x on SUM. When FE-SME is wrong, it's wrong by a bounded amount; when base is wrong, it can be catastrophically wrong.

3. **FE-SME generalizes better.** With 5M data, FE-SME shows zero overfitting (train-val gap: 0.001 at 35K). Base still overfits mildly (gap: 0.039), peaking at 20K. The structured representation provides an inductive bias that improves sample efficiency.

4. **Digit accuracy cascade.** FE-SME learns digits left-to-right: d0 (97%) > d1 (85%) > d2 (78%) > d3 (72%) > d4 (70%). This is consistent with the carry-propagation depth hypothesis — the 12-layer transformer can reliably propagate information for ~3-4 digit positions, with diminishing accuracy beyond.

5. **5M data substantially improved both models** vs 1M, but FE-SME benefited more uniformly (no overfitting, continuous improvement to 35K) while base saturated earlier.

6. **Copy tasks favor base.** For MAX, MIN, SORT — the base model copies input text to output. FE-SME must decode `<NUM>` through the adapter and regenerate as SME tokens, a harder round-trip.
