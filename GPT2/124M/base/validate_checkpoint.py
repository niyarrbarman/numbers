#!/usr/bin/env python3
"""
Validate a base GPT-2 checkpoint on the full validation split.

This script loads a checkpoint produced by GPT2/base/train.py and evaluates
val.bin deterministically over all validation blocks.

It reports:
  - token-level cross-entropy and perplexity
  - output-token accuracy on task outputs (after the "→" marker)
  - example-level exact output match
  - per-task metrics
  - numeric-task decode metrics (validity, exactness, MAE/RMSE/percentiles)
  - top-K worst numeric examples
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

from model import GPTConfig, GPT


EOT_TOKEN = 50256
CLASSIFICATION_TASKS = set()  # no classification tasks in 124M
NUMERIC_TASKS = {
    "ADD",
    "SUB",
    "SUM",
    "MIN",
    "MAX",
}
ALL_TASKS = sorted(NUMERIC_TASKS)


@dataclass
class TaskStats:
    count: int = 0
    output_exact: int = 0
    token_exact: int = 0
    tok_correct: int = 0
    tok_total: int = 0
    numeric_examples: int = 0
    numeric_valid: int = 0
    numeric_invalid: int = 0
    numeric_exact: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    abs_errs: List[float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.abs_errs is None:
            self.abs_errs = []


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    unwanted_prefix = "_orig_mod."
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            out[k[len(unwanted_prefix) :]] = v
        else:
            out[k] = v
    return out


def percentile(vals: Sequence[float], p: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def parse_numbers(text: str) -> List[float]:
    vals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return [float(v) for v in vals]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def infer_task(full_text: str) -> str:
    m = re.match(r"\s*([A-Z_]+):", full_text)
    if m:
        return m.group(1)
    return "UNKNOWN"


def make_batches(
    ids: np.memmap,
    block_size: int,
    batch_blocks: int,
    device: torch.device,
    total_blocks: int,
):
    for b0 in range(0, total_blocks, batch_blocks):
        b1 = min(b0 + batch_blocks, total_blocks)
        cur = b1 - b0
        x = np.zeros((cur, block_size), dtype=np.int64)
        y = np.full((cur, block_size), fill_value=-1, dtype=np.int64)
        for bi, block_idx in enumerate(range(b0, b1)):
            s = block_idx * block_size
            e = s + block_size
            x[bi] = ids[s:e].astype(np.int64)
            y_slice = ids[s + 1 : e + 1].astype(np.int64)
            y[bi, : len(y_slice)] = y_slice
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        yield b0, b1, xt, yt


def find_output_start(ex_ids: List[int], arrow_patterns: List[List[int]], enc) -> int:
    # Primary path: find token-level arrow pattern and split right after it.
    best_end = -1
    n = len(ex_ids)
    for pat in arrow_patterns:
        m = len(pat)
        if m == 0 or m > n:
            continue
        for i in range(0, n - m + 1):
            if ex_ids[i : i + m] == pat and i + m > best_end:
                best_end = i + m
    if best_end != -1:
        return best_end

    # Fallback: locate last arrow in decoded text, then map to token boundary.
    text = enc.decode(ex_ids)
    arrow_char = text.rfind("→")
    if arrow_char == -1:
        return -1
    target_char = arrow_char + 1
    consumed = 0
    for ti, tok in enumerate(ex_ids):
        piece = enc.decode([tok])
        consumed += len(piece)
        if consumed >= target_char:
            return ti + 1
    return len(ex_ids)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate full val set for base GPT-2 checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="Path to ckpt*.pt")
    p.add_argument("--data-dir", type=str, required=True, help="Path containing val.bin")
    p.add_argument("--batch-blocks", type=int, default=64, help="Blocks per forward batch")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k-errors", type=int, default=25, help="How many worst numeric examples to print")
    p.add_argument("--max-blocks", type=int, default=0, help="Debug cap; 0 means full val set")
    p.add_argument("--output-json", type=str, default="", help="Optional JSON summary path")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    enc = tiktoken.get_encoding("gpt2")
    arrow_patterns = []
    for candidate in ["→", " →", " ->"]:
        pat = enc.encode_ordinary(candidate)
        if pat and pat not in arrow_patterns:
            arrow_patterns.append(pat)

    print("=" * 70)
    print("Base GPT-2 Checkpoint Validation (Full Val Set)")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Device:     {device}")
    print(f"Batch blk:  {args.batch_blocks}")
    print(f"Top-K err:  {args.top_k_errors}")
    print(f"Max blocks: {args.max_blocks if args.max_blocks > 0 else 'ALL'}")

    val_bin = os.path.join(args.data_dir, "val.bin")
    meta_pkl = os.path.join(args.data_dir, "meta.pkl")
    if not os.path.exists(val_bin):
        raise FileNotFoundError(f"Missing {val_bin}")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    model.to(device)
    model.eval()

    print("-" * 70)
    print(f"Model block_size: {model.config.block_size}")
    print(f"Model vocab_size: {model.config.vocab_size}")
    print(f"Checkpoint iter:  {checkpoint.get('iter_num', 'N/A')}")
    print(f"Best val loss:    {checkpoint.get('best_val_loss', 'N/A')}")

    if os.path.exists(meta_pkl):
        import pickle

        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        print(f"Meta dataset:     {meta.get('dataset', 'N/A')}")
        print(f"Meta vocab_size:  {meta.get('vocab_size', 'N/A')}")

    ids = np.memmap(val_bin, dtype=np.uint16, mode="r")
    if len(ids) % model.config.block_size != 0:
        raise ValueError(
            f"Validation stream length {len(ids)} is not divisible by block_size {model.config.block_size}"
        )

    total_blocks = len(ids) // model.config.block_size
    if args.max_blocks > 0:
        total_blocks = min(total_blocks, args.max_blocks)
    total_tokens = total_blocks * model.config.block_size

    print("-" * 70)
    print(f"Validation tokens: {total_tokens:,}")
    print(f"Validation blocks: {total_blocks:,}")
    print(f"EOT tokens:        {int((ids[:total_tokens] == EOT_TOKEN).sum()):,}")

    pred_for_pos = np.full((total_tokens,), fill_value=-1, dtype=np.int32)
    loss_sum = 0.0
    valid_token_count = 0

    for b0, b1, x, y in make_batches(
        ids=ids,
        block_size=model.config.block_size,
        batch_blocks=args.batch_blocks,
        device=device,
        total_blocks=total_blocks,
    ):
        logits, _ = model(x, y)
        ls = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-1,
            reduction="sum",
        )
        loss_sum += float(ls.item())
        valid_token_count += int((y != -1).sum().item())

        preds = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)
        cur = b1 - b0
        bs = model.config.block_size
        for bi in range(cur):
            block_idx = b0 + bi
            s = block_idx * bs
            dst_start = s + 1
            if dst_start >= total_tokens:
                continue
            n = min(bs, total_tokens - dst_start)
            pred_for_pos[dst_start : dst_start + n] = preds[bi, :n]

    mean_loss = loss_sum / max(valid_token_count, 1)
    perplexity = float(math.exp(mean_loss)) if mean_loss < 30 else float("inf")

    # Example-level and task-level metrics
    ids_eval = np.asarray(ids[:total_tokens], dtype=np.int64)
    eot_positions = np.where(ids_eval == EOT_TOKEN)[0].tolist()

    task_stats: Dict[str, TaskStats] = {}
    examples_total = 0
    examples_evaluated = 0
    output_tok_correct = 0
    output_tok_total = 0
    output_exact_total = 0
    token_exact_total = 0
    num_abs_errs_all: List[float] = []
    rel_errs_nonzero: List[float] = []
    worst_examples: List[Tuple[float, Dict[str, object]]] = []

    start = 0
    for eot in eot_positions:
        if eot <= start:
            start = eot + 1
            continue
        ex_start, ex_end = start, eot  # ex_end excludes EOT
        start = eot + 1
        examples_total += 1

        ex_ids = ids_eval[ex_start:ex_end].tolist()
        if not ex_ids:
            continue

        full_text = enc.decode(ex_ids)
        if "→" not in full_text:
            continue

        task = infer_task(full_text)
        stats = task_stats.setdefault(task, TaskStats())

        out_rel = find_output_start(ex_ids, arrow_patterns, enc)
        if out_rel < 0 or out_rel >= len(ex_ids):
            continue

        tgt_out_ids = ex_ids[out_rel:]
        global_out_start = ex_start + out_rel
        global_out_end = ex_end

        pred_out_ids: List[int] = []
        missing = False
        for g in range(global_out_start, global_out_end):
            pid = int(pred_for_pos[g])
            if pid < 0:
                missing = True
                break
            pred_out_ids.append(pid)
        if missing or len(pred_out_ids) != len(tgt_out_ids) or len(tgt_out_ids) == 0:
            continue

        examples_evaluated += 1
        stats.count += 1

        tok_correct = sum(int(a == b) for a, b in zip(pred_out_ids, tgt_out_ids))
        tok_total = len(tgt_out_ids)
        tok_exact = tok_correct == tok_total
        output_tok_correct += tok_correct
        output_tok_total += tok_total
        token_exact_total += int(tok_exact)
        stats.tok_correct += tok_correct
        stats.tok_total += tok_total
        stats.token_exact += int(tok_exact)

        tgt_out_text = normalize_text(enc.decode(tgt_out_ids))
        pred_out_text = normalize_text(enc.decode(pred_out_ids))

        if task in CLASSIFICATION_TASKS:
            out_exact = pred_out_text.upper() == tgt_out_text.upper()
        else:
            out_exact = pred_out_text == tgt_out_text
        output_exact_total += int(out_exact)
        stats.output_exact += int(out_exact)

        # Numeric decode metrics only for numeric tasks.
        if task in NUMERIC_TASKS:
            stats.numeric_examples += 1
            tgt_nums = parse_numbers(tgt_out_text)
            pred_nums = parse_numbers(pred_out_text)
            prompt = normalize_text(full_text.split("→", 1)[0])

            if len(tgt_nums) == 0 or len(tgt_nums) != len(pred_nums):
                stats.numeric_invalid += 1
                worst_examples.append(
                    (
                        float("inf"),
                        {
                            "task": task,
                            "prompt": prompt,
                            "target_text": tgt_out_text,
                            "pred_text": pred_out_text,
                            "target_nums": tgt_nums,
                            "pred_nums": pred_nums,
                            "valid_numeric": False,
                            "abs_err_max": None,
                            "abs_err_mean": None,
                        },
                    )
                )
            else:
                stats.numeric_valid += 1
                abs_errs = [abs(t - p) for t, p in zip(tgt_nums, pred_nums)]
                sq_errs = [(t - p) * (t - p) for t, p in zip(tgt_nums, pred_nums)]
                for t, ae in zip(tgt_nums, abs_errs):
                    num_abs_errs_all.append(ae)
                    if abs(t) > 1e-12:
                        rel_errs_nonzero.append(ae / abs(t))
                stats.abs_err_sum += float(sum(abs_errs))
                stats.sq_err_sum += float(sum(sq_errs))
                stats.abs_errs.extend(abs_errs)
                if all(ae == 0.0 for ae in abs_errs):
                    stats.numeric_exact += 1

                worst_examples.append(
                    (
                        max(abs_errs) if abs_errs else 0.0,
                        {
                            "task": task,
                            "prompt": prompt,
                            "target_text": tgt_out_text,
                            "pred_text": pred_out_text,
                            "target_nums": tgt_nums,
                            "pred_nums": pred_nums,
                            "valid_numeric": True,
                            "abs_err_max": max(abs_errs) if abs_errs else 0.0,
                            "abs_err_mean": (sum(abs_errs) / len(abs_errs)) if abs_errs else 0.0,
                        },
                    )
                )

    worst_examples.sort(key=lambda x: x[0], reverse=True)
    worst_examples = worst_examples[: args.top_k_errors]

    num_valid_values = sum(s.numeric_valid for s in task_stats.values())
    num_invalid_values = sum(s.numeric_invalid for s in task_stats.values())
    num_examples = sum(s.numeric_examples for s in task_stats.values())
    num_exact_examples = sum(s.numeric_exact for s in task_stats.values())
    mae = float(np.mean(np.asarray(num_abs_errs_all))) if num_abs_errs_all else 0.0
    rmse = (
        math.sqrt(
            sum(s.sq_err_sum for s in task_stats.values()) / max(sum(len(s.abs_errs) for s in task_stats.values()), 1)
        )
        if num_abs_errs_all
        else 0.0
    )
    med_abs = percentile(num_abs_errs_all, 50)
    p90_abs = percentile(num_abs_errs_all, 90)
    p95_abs = percentile(num_abs_errs_all, 95)
    p99_abs = percentile(num_abs_errs_all, 99)
    max_abs = max(num_abs_errs_all) if num_abs_errs_all else 0.0
    mape = float(np.mean(np.asarray(rel_errs_nonzero))) if rel_errs_nonzero else 0.0

    print("=" * 70)
    print("Validation Report")
    print("=" * 70)
    print(f"Tokens evaluated:              {valid_token_count:,}")
    print(f"Cross-entropy loss (avg):      {mean_loss:.6f}")
    print(f"Perplexity:                    {perplexity:.6f}")
    print("-" * 70)
    print(f"Examples parsed (EOT-bounded): {examples_total:,}")
    print(f"Examples evaluated:            {examples_evaluated:,}")
    print(
        f"Output token accuracy:         "
        f"{(output_tok_correct / output_tok_total if output_tok_total else 0.0):.4f} "
        f"({output_tok_correct}/{output_tok_total})"
    )
    print(
        f"Output exact-match rate:       "
        f"{(output_exact_total / examples_evaluated if examples_evaluated else 0.0):.4f} "
        f"({output_exact_total}/{examples_evaluated})"
    )
    print(
        f"Output token-seq exact rate:   "
        f"{(token_exact_total / examples_evaluated if examples_evaluated else 0.0):.4f} "
        f"({token_exact_total}/{examples_evaluated})"
    )

    print("-" * 70)
    print("Numeric Task Metrics")
    print("-" * 70)
    print(f"Numeric examples:              {num_examples:,}")
    print(f"Numeric valid parses:          {num_valid_values:,}")
    print(f"Numeric invalid parses:        {num_invalid_values:,}")
    print(
        f"Numeric exact-example rate:    "
        f"{(num_exact_examples / num_examples if num_examples else 0.0):.4f} "
        f"({num_exact_examples}/{num_examples})"
    )
    print(f"MAE (value-level):             {mae:.6f}")
    print(f"RMSE (value-level):            {rmse:.6f}")
    print(f"Median abs err:                {med_abs:.6f}")
    print(f"P90/P95/P99 abs err:           {p90_abs:.6f} / {p95_abs:.6f} / {p99_abs:.6f}")
    print(f"Max abs err:                   {max_abs:.6f}")
    print(f"MAPE (|target|>0):             {mape:.6f}")

    print("-" * 70)
    print("Per-Task Metrics")
    print("-" * 70)
    print(
        f"{'Task':<12} {'N':>8} {'OutExact':>10} {'TokAcc':>10} "
        f"{'TokExact':>10} {'NumValid':>10} {'NumInv':>8} {'NumMAE':>10} {'NumP95':>10}"
    )
    for task in sorted(task_stats.keys()):
        s = task_stats[task]
        tok_acc = (s.tok_correct / s.tok_total) if s.tok_total else 0.0
        out_exact_rate = (s.output_exact / s.count) if s.count else 0.0
        tok_exact_rate = (s.token_exact / s.count) if s.count else 0.0
        num_mae = (s.abs_err_sum / len(s.abs_errs)) if s.abs_errs else 0.0
        num_p95 = percentile(s.abs_errs, 95) if s.abs_errs else 0.0
        print(
            f"{task:<12} {s.count:8d} {out_exact_rate:10.4f} {tok_acc:10.4f} "
            f"{tok_exact_rate:10.4f} {s.numeric_valid:10d} {s.numeric_invalid:8d} "
            f"{num_mae:10.4f} {num_p95:10.4f}"
        )

    print("-" * 70)
    print(f"Top {len(worst_examples)} Worst Numeric Examples")
    print("-" * 70)
    for i, (_, ex) in enumerate(worst_examples, 1):
        print(f"[{i:02d}] task={ex['task']} valid_numeric={ex['valid_numeric']}")
        print(f"  prompt:      {ex['prompt']}")
        print(f"  target_text: {ex['target_text']}")
        print(f"  pred_text:   {ex['pred_text']}")
        print(f"  target_nums: {ex['target_nums']}")
        print(f"  pred_nums:   {ex['pred_nums']}")
        print(f"  abs_err_max: {ex['abs_err_max']}")
        print(f"  abs_err_mean:{ex['abs_err_mean']}")

    summary = {
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "tokens_evaluated": valid_token_count,
        "cross_entropy": mean_loss,
        "perplexity": perplexity,
        "examples_total": examples_total,
        "examples_evaluated": examples_evaluated,
        "output_token_accuracy": (output_tok_correct / output_tok_total if output_tok_total else 0.0),
        "output_exact_rate": (output_exact_total / examples_evaluated if examples_evaluated else 0.0),
        "output_token_seq_exact_rate": (token_exact_total / examples_evaluated if examples_evaluated else 0.0),
        "numeric_metrics": {
            "numeric_examples": num_examples,
            "numeric_valid": num_valid_values,
            "numeric_invalid": num_invalid_values,
            "numeric_exact_example_rate": (num_exact_examples / num_examples if num_examples else 0.0),
            "mae": mae,
            "rmse": rmse,
            "median_abs_err": med_abs,
            "p90_abs_err": p90_abs,
            "p95_abs_err": p95_abs,
            "p99_abs_err": p99_abs,
            "max_abs_err": max_abs,
            "mape_nonzero": mape,
        },
        "per_task": {
            task: {
                "count": s.count,
                "output_exact_rate": (s.output_exact / s.count if s.count else 0.0),
                "token_accuracy": (s.tok_correct / s.tok_total if s.tok_total else 0.0),
                "token_seq_exact_rate": (s.token_exact / s.count if s.count else 0.0),
                "numeric_examples": s.numeric_examples,
                "numeric_valid": s.numeric_valid,
                "numeric_invalid": s.numeric_invalid,
                "numeric_exact_rate": (s.numeric_exact / s.numeric_examples if s.numeric_examples else 0.0),
                "numeric_mae": (s.abs_err_sum / len(s.abs_errs) if s.abs_errs else 0.0),
                "numeric_p95_abs_err": percentile(s.abs_errs, 95) if s.abs_errs else 0.0,
            }
            for task, s in task_stats.items()
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


if __name__ == "__main__":
    main()
