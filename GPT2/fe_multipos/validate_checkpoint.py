#!/usr/bin/env python3
"""
Validate a multi-position SME checkpoint on the full validation split.

Loads a checkpoint produced by fe_multipos/train.py and runs deterministic
validation. Same metrics as fe_unfreeze/validate_checkpoint.py but loads
the additional pos.bin file for multi-position support.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import GPTConfig, GPT, NUM_TOKEN_ID
from prepare import (
    SME_SIGN_POS,
    SME_SIGN_NEG,
    SME_EXP_BASE,
    SME_EXP_OFFSET,
    SME_N_EXP,
    SME_DIGIT_BASE,
    SME_END,
    SME_ALL_TOKENS,
    NUM_POSITIONS,
    parse_sme_number_tokens,
    sme_tokens_to_number,
)


TASKS = [
    "CMP", "GT", "IS_POS", "IS_SORTED", "CHECKSORT", "CHECKADD", "SUM_CMP",
    "SORT", "ADD", "SUB", "MIN", "MAX", "SUM", "COUNT",
]


@dataclass
class TaskStats:
    count: int = 0
    valid: int = 0
    invalid: int = 0
    exact_token: int = 0
    exact_value: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    abs_errs: List[float] = None  # type: ignore

    def __post_init__(self):
        if self.abs_errs is None:
            self.abs_errs = []

    def add(self, is_valid, is_exact_token, is_exact_value, abs_err):
        self.count += 1
        if is_exact_token:
            self.exact_token += 1
        if is_valid:
            self.valid += 1
            if is_exact_value:
                self.exact_value += 1
            assert abs_err is not None
            self.abs_err_sum += abs_err
            self.sq_err_sum += abs_err * abs_err
            self.abs_errs.append(abs_err)
        else:
            self.invalid += 1


class RunningSMECounts:
    def __init__(self):
        self.counts = {
            "overall": [0, 0],
            "sign": [0, 0],
            "exp": [0, 0],
            "digit": [0, 0],
            "end": [0, 0],
        }

    def _add(self, key, correct, total):
        self.counts[key][0] += int(correct)
        self.counts[key][1] += int(total)

    def update(self, logits, targets):
        preds = logits.argmax(dim=-1)
        sme_set = torch.tensor(sorted(SME_ALL_TOKENS), device=targets.device)
        sme_mask = torch.isin(targets, sme_set)
        if sme_mask.any():
            c = (preds[sme_mask] == targets[sme_mask]).sum().item()
            t = sme_mask.sum().item()
            self._add("overall", c, t)

        sign_mask = (targets == SME_SIGN_POS) | (targets == SME_SIGN_NEG)
        if sign_mask.any():
            self._add("sign", (preds[sign_mask] == targets[sign_mask]).sum().item(), sign_mask.sum().item())

        exp_mask = (targets >= SME_EXP_BASE) & (targets < SME_EXP_BASE + SME_N_EXP)
        if exp_mask.any():
            self._add("exp", (preds[exp_mask] == targets[exp_mask]).sum().item(), exp_mask.sum().item())

        digit_mask = (targets >= SME_DIGIT_BASE) & (targets <= SME_DIGIT_BASE + 9)
        if digit_mask.any():
            self._add("digit", (preds[digit_mask] == targets[digit_mask]).sum().item(), digit_mask.sum().item())

        end_mask = (targets == SME_END)
        if end_mask.any():
            self._add("end", (preds[end_mask] == targets[end_mask]).sum().item(), end_mask.sum().item())

        bsz, T = targets.shape
        tgt_rows = targets.detach().cpu().tolist()
        pred_rows = preds.detach().cpu().tolist()
        for b in range(bsz):
            row_t = tgt_rows[b]
            row_p = pred_rows[b]
            pos = 0
            while pos < T:
                if row_t[pos] not in (SME_SIGN_POS, SME_SIGN_NEG):
                    pos += 1
                    continue
                parsed, next_pos = parse_sme_number_tokens(row_t, start_idx=pos)
                if parsed is None:
                    pos += 1
                    continue
                n_digits = len(parsed) - 3
                for di in range(n_digits):
                    dpos = pos + 2 + di
                    tgt = row_t[dpos]
                    if SME_DIGIT_BASE <= tgt <= SME_DIGIT_BASE + 9:
                        key = f"d{di}"
                        if key not in self.counts:
                            self.counts[key] = [0, 0]
                        self._add(key, int(row_p[dpos] == tgt), 1)
                pos = max(next_pos, pos + 1)

    def accuracy(self, key):
        c, t = self.counts[key]
        return (c / t) if t else 0.0

    def total(self, key):
        return self.counts[key][1]


def percentile(vals, p):
    if not vals:
        return 0.0
    return float(np.percentile(np.array(vals, dtype=np.float64), p))


def decode_context(token_ids, num_values, enc):
    parts = []
    for i, tok in enumerate(token_ids):
        if tok == NUM_TOKEN_ID:
            parts.append(f"<{num_values[i]:g}>")
        elif tok in SME_ALL_TOKENS:
            if tok == SME_SIGN_POS: parts.append("S+")
            elif tok == SME_SIGN_NEG: parts.append("S-")
            elif SME_EXP_BASE <= tok < SME_EXP_BASE + SME_N_EXP:
                parts.append(f"E{(tok - SME_EXP_BASE) - SME_EXP_OFFSET}")
            elif SME_DIGIT_BASE <= tok <= SME_DIGIT_BASE + 9:
                parts.append(f"D{tok - SME_DIGIT_BASE}")
            elif tok == SME_END: parts.append("END")
            else: parts.append(f"?{tok}")
        elif tok == 50256:
            break
        else:
            try: parts.append(enc.decode([tok]))
            except Exception: parts.append(f"[{tok}]")
    return "".join(parts)


def infer_task(context):
    best_idx = None
    best_task = None
    for task in TASKS:
        idx = context.find(f"{task}:")
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_task = task
    if best_task is not None:
        return best_task
    m = re.search(r"\b([A-Z_]{2,}):", context)
    if m:
        return m.group(1)
    return "UNKNOWN"


def clean_state_dict(state_dict):
    unwanted_prefix = "_orig_mod."
    out = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            out[k[len(unwanted_prefix):]] = v
        else:
            out[k] = v
    return out


def make_batches(ids, nums, pos_data, block_size, batch_blocks, device):
    """Yield batches of (x, y, nv, nm, pi) from the validation stream."""
    total_tokens = len(ids)
    n_blocks = total_tokens // block_size
    for b0 in range(0, n_blocks, batch_blocks):
        b1 = min(b0 + batch_blocks, n_blocks)
        cur = b1 - b0
        x = np.zeros((cur, block_size), dtype=np.int64)
        y = np.full((cur, block_size), fill_value=-1, dtype=np.int64)
        nv = np.zeros((cur, block_size), dtype=np.float32)
        pi = np.full((cur, block_size), fill_value=-1, dtype=np.int8)
        for bi, block_idx in enumerate(range(b0, b1)):
            s = block_idx * block_size
            e = s + block_size
            x[bi] = ids[s:e].astype(np.int64)
            nv[bi] = nums[s:e].astype(np.float32)
            pi[bi] = pos_data[s:e].copy()
            y_slice = ids[s + 1 : e + 1].astype(np.int64)
            y[bi, : len(y_slice)] = y_slice
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        nvt = torch.from_numpy(nv).to(device)
        nmt = xt.eq(NUM_TOKEN_ID)
        pit = torch.from_numpy(pi.astype(np.int8)).to(device)
        yield b0, b1, xt, yt, nvt, nmt, pit


def main():
    p = argparse.ArgumentParser(description="Validate multi-position SME checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--batch-blocks", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k-errors", type=int, default=25)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    print("=" * 70)
    print("Multi-Position SME Checkpoint Validation (Full Val Set)")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Device:     {device}")
    print(f"Batch blk:  {args.batch_blocks}")
    print(f"Top-K err:  {args.top_k_errors}")
    print(f"Max blocks: {args.max_blocks if args.max_blocks > 0 else 'ALL'}")

    val_bin = os.path.join(args.data_dir, "val.bin")
    val_nums_bin = os.path.join(args.data_dir, "val_nums.bin")
    val_pos_bin = os.path.join(args.data_dir, "val_pos.bin")

    for f in [val_bin, val_nums_bin, val_pos_bin]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f}")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    model.to(device)
    model.eval()

    print("-" * 70)
    print(f"Model block_size:  {model.config.block_size}")
    print(f"Model vocab_size:  {model.config.vocab_size}")
    print(f"Model num_positions: {model.config.num_positions}")
    print(f"Checkpoint iter:   {checkpoint.get('iter_num', 'N/A')}")
    print(f"Best val loss:     {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Num emb ckpt:      {model_args.get('num_emb_checkpoint', '')}")

    meta_pkl = os.path.join(args.data_dir, "meta.pkl")
    if os.path.exists(meta_pkl):
        import pickle
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        print(f"Meta vocab_size:   {meta.get('vocab_size', 'N/A')}")
        print(f"Meta num_positions:{meta.get('num_positions', 'N/A')}")

    ids = np.memmap(val_bin, dtype=np.uint16, mode="r")
    nums = np.memmap(val_nums_bin, dtype=np.float32, mode="r")
    pos_data = np.memmap(val_pos_bin, dtype=np.int8, mode="r")

    if len(ids) != len(nums) or len(ids) != len(pos_data):
        raise ValueError(f"Binary file length mismatch: ids={len(ids)}, nums={len(nums)}, pos={len(pos_data)}")

    total_blocks = len(ids) // model.config.block_size
    if args.max_blocks > 0:
        total_blocks = min(total_blocks, args.max_blocks)

    num_token_count = int((ids[: total_blocks * model.config.block_size] == NUM_TOKEN_ID).sum())
    print("-" * 70)
    print(f"Validation tokens: {len(ids):,}")
    print(f"Validation blocks: {total_blocks:,}")
    print(f"<NUM> tokens:      {num_token_count:,} ({num_token_count // NUM_POSITIONS:,} unique numbers)")
    print(f"SME token ids:     [{min(SME_ALL_TOKENS)}, {max(SME_ALL_TOKENS)}]")

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        print("Warning: tiktoken not available")

    sme_counts = RunningSMECounts()
    task_stats: Dict[str, TaskStats] = {}
    abs_errs_all: List[float] = []
    rel_errs_nonzero: List[float] = []
    worst_examples: List[Tuple[float, Dict]] = []
    loss_sum = 0.0
    valid_token_count = 0

    for b0, b1, x, y, nv, nm, pi in make_batches(
        ids=ids, nums=nums, pos_data=pos_data,
        block_size=model.config.block_size,
        batch_blocks=args.batch_blocks, device=device,
    ):
        if b0 >= total_blocks:
            break
        if b1 > total_blocks:
            keep = total_blocks - b0
            x, y, nv, nm, pi = x[:keep], y[:keep], nv[:keep], nm[:keep], pi[:keep]
            b1 = total_blocks

        logits, _ = model(x, y, num_values=nv, num_mask=nm, pos_indices=pi)
        ls = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            ignore_index=-1, reduction="sum",
        )
        loss_sum += float(ls.item())
        valid_token_count += int((y != -1).sum().item())

        sme_counts.update(logits, y)

        preds = logits.argmax(dim=-1)
        x_cpu = x.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        nv_cpu = nv.detach().cpu().numpy()
        p_cpu = preds.detach().cpu().numpy()

        B, T = y_cpu.shape
        for bi in range(B):
            row_t = y_cpu[bi].tolist()
            row_p = p_cpu[bi].tolist()
            sign_positions = np.where(
                (y_cpu[bi] == SME_SIGN_POS) | (y_cpu[bi] == SME_SIGN_NEG)
            )[0]
            for t in sign_positions.tolist():
                tgt_toks, _ = parse_sme_number_tokens(row_t, start_idx=t)
                if tgt_toks is None:
                    continue
                pred_toks, _ = parse_sme_number_tokens(row_p, start_idx=t)
                tgt_val = sme_tokens_to_number(tgt_toks)
                pred_val = sme_tokens_to_number(pred_toks) if pred_toks is not None else None
                if tgt_val is None:
                    continue

                task_start = 0
                for s in range(t - 1, -1, -1):
                    if int(x_cpu[bi, s]) == 50256:
                        task_start = s + 1
                        break
                if enc is not None:
                    context = decode_context(
                        x_cpu[bi, task_start:t].tolist(),
                        nv_cpu[bi, task_start:t].tolist(), enc)
                else:
                    context = f"[row={bi} pos={t}]"
                task = infer_task(context)
                stats = task_stats.setdefault(task, TaskStats())

                exact_token = (pred_toks == tgt_toks)
                if pred_val is None:
                    stats.add(False, exact_token, False, None)
                    record = {
                        "task": task, "context": context, "target": tgt_val,
                        "pred": None, "target_tokens": tgt_toks,
                        "pred_tokens": pred_toks, "abs_err": None,
                        "rel_err": None, "valid": False,
                    }
                    worst_examples.append((float("inf"), record))
                    continue

                abs_err = abs(float(tgt_val) - float(pred_val))
                if abs(tgt_val) > 1e-12:
                    rel_err = abs_err / abs(float(tgt_val))
                    rel_errs_nonzero.append(rel_err)
                else:
                    rel_err = math.inf if abs_err > 0 else 0.0

                exact_value = abs_err == 0.0
                stats.add(True, exact_token, exact_value, abs_err)
                abs_errs_all.append(abs_err)
                record = {
                    "task": task, "context": context,
                    "target": float(tgt_val), "pred": float(pred_val),
                    "target_tokens": tgt_toks, "pred_tokens": pred_toks,
                    "abs_err": abs_err,
                    "rel_err": rel_err if math.isfinite(rel_err) else None,
                    "valid": True,
                }
                worst_examples.append((abs_err, record))

    worst_examples.sort(key=lambda x: x[0], reverse=True)
    worst_examples = worst_examples[: args.top_k_errors]

    mean_loss = loss_sum / max(valid_token_count, 1)
    ppl = float(math.exp(mean_loss)) if mean_loss < 30 else float("inf")

    total_num_preds = sum(s.count for s in task_stats.values())
    total_valid_num = sum(s.valid for s in task_stats.values())
    total_invalid_num = sum(s.invalid for s in task_stats.values())
    total_exact_token = sum(s.exact_token for s in task_stats.values())
    total_exact_value = sum(s.exact_value for s in task_stats.values())
    sq_err_sum = sum(s.sq_err_sum for s in task_stats.values())

    mae = (sum(abs_errs_all) / len(abs_errs_all)) if abs_errs_all else 0.0
    rmse = math.sqrt(sq_err_sum / total_valid_num) if total_valid_num else 0.0
    med_abs = percentile(abs_errs_all, 50)
    p90_abs = percentile(abs_errs_all, 90)
    p95_abs = percentile(abs_errs_all, 95)
    p99_abs = percentile(abs_errs_all, 99)
    max_abs = max(abs_errs_all) if abs_errs_all else 0.0
    mape = float(np.mean(np.array(rel_errs_nonzero, dtype=np.float64))) if rel_errs_nonzero else 0.0

    print("=" * 70)
    print("Validation Report")
    print("=" * 70)
    print(f"Tokens evaluated:            {valid_token_count:,}")
    print(f"Cross-entropy loss (avg):    {mean_loss:.6f}")
    print(f"Perplexity:                  {ppl:.6f}")

    print("-" * 70)
    print("SME Token Accuracy")
    print("-" * 70)
    for key in ["overall", "sign", "exp", "digit", "end"]:
        print(f"{key:>8}: {sme_counts.accuracy(key):.4f} "
              f"({sme_counts.counts[key][0]}/{sme_counts.counts[key][1]})")
    digit_keys = sorted(
        [k for k in sme_counts.counts if k.startswith("d") and k[1:].isdigit() and sme_counts.counts[k][1] > 0],
        key=lambda k: int(k[1:]))
    for key in digit_keys:
        print(f"{key:>8}: {sme_counts.accuracy(key):.4f} "
              f"({sme_counts.counts[key][0]}/{sme_counts.counts[key][1]})")

    print("-" * 70)
    print("Number Decode Metrics (from SME number starts)")
    print("-" * 70)
    print(f"Number predictions total:    {total_num_preds:,}")
    print(f"Valid decoded predictions:   {total_valid_num:,}")
    print(f"Invalid decoded predictions: {total_invalid_num:,}")
    print(f"Invalid rate:                {(total_invalid_num / total_num_preds if total_num_preds else 0.0):.4f}")
    print(f"Exact token-seq rate:        {(total_exact_token / total_num_preds if total_num_preds else 0.0):.4f}")
    print(f"Exact value rate:            {(total_exact_value / total_num_preds if total_num_preds else 0.0):.4f}")
    print(f"MAE:                         {mae:.6f}")
    print(f"RMSE:                        {rmse:.6f}")
    print(f"Median abs err:              {med_abs:.6f}")
    print(f"P90/P95/P99 abs err:         {p90_abs:.6f} / {p95_abs:.6f} / {p99_abs:.6f}")
    print(f"Max abs err:                 {max_abs:.6f}")
    print(f"MAPE (|target|>0):           {mape:.6f}")

    print("-" * 70)
    print("Per-Task Metrics")
    print("-" * 70)
    print(f"{'Task':<12} {'N':>8} {'Valid':>8} {'Invalid':>8} "
          f"{'ExactTok':>10} {'ExactVal':>10} {'MAE':>10} {'P95':>10}")
    for task in sorted(task_stats.keys()):
        s = task_stats[task]
        mae_task = (s.abs_err_sum / s.valid) if s.valid else 0.0
        p95_task = percentile(s.abs_errs, 95) if s.valid else 0.0
        print(f"{task:<12} {s.count:8d} {s.valid:8d} {s.invalid:8d} "
              f"{(s.exact_token / s.count if s.count else 0.0):10.4f} "
              f"{(s.exact_value / s.count if s.count else 0.0):10.4f} "
              f"{mae_task:10.4f} {p95_task:10.4f}")

    print("-" * 70)
    print(f"Top {len(worst_examples)} Worst Absolute-Error Examples")
    print("-" * 70)
    for i, (_, ex) in enumerate(worst_examples, 1):
        print(f"[{i:02d}] task={ex['task']} valid={ex['valid']}")
        print(f"  context: {ex['context']}")
        print(f"  target:  {ex['target']}")
        print(f"  pred:    {ex['pred']}")
        print(f"  abs_err: {ex['abs_err']}")
        print(f"  rel_err: {ex['rel_err']}")
        print(f"  tgt_toks:{ex['target_tokens']}")
        print(f"  pred_toks:{ex['pred_tokens']}")

    summary = {
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "num_positions": model.config.num_positions,
        "tokens_evaluated": valid_token_count,
        "cross_entropy": mean_loss,
        "perplexity": ppl,
        "sme_accuracy": {
            k: {"accuracy": sme_counts.accuracy(k),
                "correct": sme_counts.counts[k][0],
                "total": sme_counts.counts[k][1]}
            for k in sme_counts.counts if sme_counts.counts[k][1] > 0
        },
        "number_metrics": {
            "total": total_num_preds,
            "valid": total_valid_num,
            "invalid": total_invalid_num,
            "invalid_rate": (total_invalid_num / total_num_preds if total_num_preds else 0.0),
            "exact_token_rate": (total_exact_token / total_num_preds if total_num_preds else 0.0),
            "exact_value_rate": (total_exact_value / total_num_preds if total_num_preds else 0.0),
            "mae": mae, "rmse": rmse,
            "median_abs_err": med_abs,
            "p90_abs_err": p90_abs, "p95_abs_err": p95_abs, "p99_abs_err": p99_abs,
            "max_abs_err": max_abs, "mape_nonzero": mape,
        },
        "per_task": {
            t: {
                "count": s.count, "valid": s.valid, "invalid": s.invalid,
                "exact_token_rate": (s.exact_token / s.count if s.count else 0.0),
                "exact_value_rate": (s.exact_value / s.count if s.count else 0.0),
                "mae": (s.abs_err_sum / s.valid if s.valid else 0.0),
                "p95_abs_err": percentile(s.abs_errs, 95) if s.valid else 0.0,
            }
            for t, s in task_stats.items()
        },
    }

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("-" * 70)
        print(f"Saved JSON summary: {args.output_json}")

    print("=" * 70)
    print("Validation finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
