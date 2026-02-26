# Codex Results Summary (2026-02-21)

## Scope
- Report context reviewed: `Number_Embedding_Report_v2.md`
- Standalone embedding/regression logs reviewed: `slurm_logs/*.log`
- GPT training logs reviewed:
  - SME path: `GPT2/slurm_logs/gpt2_sme_77762.log`, `GPT2/slurm_logs/gpt2_sme_77765.log`
  - FE/regression/bin paths: `GPT2/slurm_logs/gpt2_fe_*.log`
  - Adapter+head autoencoder sanity check: `GPT2/slurm_logs/test_autoencoder_77753.log`

## Key Findings

### 1) GPT-2 + SME decoding currently has the strongest end-to-end signal
- Final available run:
  - `iter 11292: loss 0.2241` at `GPT2/slurm_logs/gpt2_sme_77765.log:12394`
  - Last DIAG block at `iter 11200` with:
    - SME overall `0.910`, sign `0.991`, exp `0.944`, digit `0.871`
    - `GPT2/slurm_logs/gpt2_sme_77765.log:12297`
    - `GPT2/slurm_logs/gpt2_sme_77765.log:12302`
- Decode quality is improved but still brittle on some cases:
  - Invalid output token event:
    - `GPT2/slurm_logs/gpt2_sme_77765.log:8808`
  - Catastrophic miss examples:
    - target `-969`, pred `866` at `GPT2/slurm_logs/gpt2_sme_77765.log:10991`, `GPT2/slurm_logs/gpt2_sme_77765.log:10992`
    - target `6`, pred `300` at `GPT2/slurm_logs/gpt2_sme_77765.log:12067`, `GPT2/slurm_logs/gpt2_sme_77765.log:12068`

### 2) GPT-2 + FE (regression/bin) runs are inconclusive or weak so far
- Many FE runs were cancelled mid-training (time/step termination), e.g.:
  - `GPT2/slurm_logs/gpt2_fe_77750.log:4581`
  - `GPT2/slurm_logs/gpt2_fe_77751.log:7425`
  - `GPT2/slurm_logs/gpt2_fe_77752.log:4081`
  - `GPT2/slurm_logs/gpt2_fe_77754.log:4667`
- Representative late diagnostics:
  - FE lambda=5 run: text `0.2364`, num `17.3134`
    - `GPT2/slurm_logs/gpt2_fe_77750.log:4503`
  - FE bins runs: text around `0.21-0.23`, num around `3.63-3.84`
    - `GPT2/slurm_logs/gpt2_fe_77751.log:7371`
    - `GPT2/slurm_logs/gpt2_fe_77752.log:3998`
    - `GPT2/slurm_logs/gpt2_fe_77754.log:4615`
- A longer FE curriculum run reached 50k iters, but numeric loss remained high:
  - DIAG iter 50000: text `0.0725`, num `36.2083`
  - `GPT2/slurm_logs/gpt2_fe_77730.log:52830`
  - `GPT2/slurm_logs/gpt2_fe_77730.log:52831`

### 3) Standalone number embedding quality remains strong
- Best standalone torch run:
  - `22/22` tests passed: `slurm_logs/np_emb_torch_77570.log:245`
  - Example large value reconstruction:
    - `100000 -> 100072.38281` (0.0724%): `slurm_logs/np_emb_torch_77570.log:211`
- Visualization/eval summary:
  - mean relative error `0.2911%`: `slurm_logs/np_emb_viz_77571.log:18`

## Interpretation Against Report
- The report explicitly states:
  - learned decoding has fundamental precision limits
  - output-side decoding is the hard bottleneck
  - recommended hybrid with reserved dimensions / analytic decode
  - references:
    - `Number_Embedding_Report_v2.md:575`
    - `Number_Embedding_Report_v2.md:600`
    - `Number_Embedding_Report_v2.md:622`
    - `Number_Embedding_Report_v2.md:661`
    - `Number_Embedding_Report_v2.md:801`
    - `Number_Embedding_Report_v2.md:883`

## Recommended Direction
1. Keep SME decoding as a benchmark and fallback, not the final architecture.
2. Move toward analytic numeric outputs (number head + analytic decode) for production-quality numeric correctness.
3. During transition, harden SME with constrained decoding and verifier/reranking to reduce invalid/catastrophic outputs.
4. Evaluate both token-level and value-level metrics (exact match, MAE, catastrophic error rate) before deciding final deployment path.

## Decision
- Continue SME only as a short-term baseline.
- Main project direction should shift to analytic/hybrid numeric output to stay aligned with the original number-embedding objective.
