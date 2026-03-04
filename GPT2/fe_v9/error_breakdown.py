#!/usr/bin/env python3
"""
Detailed FE-v9 exactness error breakdown.

Classifies each number prediction into:
  - exact
  - miss_invalid_decode
  - miss_correct_mantissa_wrong_exponent
  - miss_valid_but_wrong_digit

This is designed to answer:
  "Where does FE lose exactness versus Base?"
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from model import GPTConfig, GPT
from prepare import (
    SME_SIGN_POS,
    SME_SIGN_NEG,
    parse_sme_number_tokens,
    sme_tokens_to_number,
)
from validate_checkpoint import clean_state_dict, make_batches, decode_context, infer_task


@dataclass
class Counts:
    total: int = 0
    exact: int = 0
    miss_invalid_decode: int = 0
    miss_correct_mantissa_wrong_exponent: int = 0
    miss_valid_but_wrong_digit: int = 0
    # Extra diagnostics for the "valid but wrong" bucket.
    valid_sign_only_mismatch: int = 0
    valid_exponent_only_mismatch: int = 0
    valid_digit_only_mismatch: int = 0
    valid_digit_and_exponent_mismatch: int = 0
    valid_sign_and_other_mismatch: int = 0

    def add_exact(self) -> None:
        self.total += 1
        self.exact += 1

    def add_invalid(self) -> None:
        self.total += 1
        self.miss_invalid_decode += 1

    def add_cmwe(self) -> None:
        self.total += 1
        self.miss_correct_mantissa_wrong_exponent += 1
        self.valid_exponent_only_mismatch += 1

    def add_valid_wrong(
        self,
        *,
        sign_match: bool,
        exp_match: bool,
        digits_match: bool,
    ) -> None:
        self.total += 1
        self.miss_valid_but_wrong_digit += 1

        if not sign_match and exp_match and digits_match:
            self.valid_sign_only_mismatch += 1
        elif sign_match and exp_match and not digits_match:
            self.valid_digit_only_mismatch += 1
        elif sign_match and not exp_match and not digits_match:
            self.valid_digit_and_exponent_mismatch += 1
        else:
            self.valid_sign_and_other_mismatch += 1

    @property
    def misses(self) -> int:
        return self.total - self.exact

    def as_dict(self) -> Dict[str, object]:
        misses = self.misses
        out: Dict[str, object] = {
            "total": self.total,
            "exact": self.exact,
            "exact_rate": (self.exact / self.total) if self.total else 0.0,
            "misses": misses,
            "miss_invalid_decode": self.miss_invalid_decode,
            "miss_correct_mantissa_wrong_exponent": self.miss_correct_mantissa_wrong_exponent,
            "miss_valid_but_wrong_digit": self.miss_valid_but_wrong_digit,
            "miss_breakdown_over_total": {
                "invalid_decode": (self.miss_invalid_decode / self.total) if self.total else 0.0,
                "correct_mantissa_wrong_exponent": (
                    self.miss_correct_mantissa_wrong_exponent / self.total
                )
                if self.total
                else 0.0,
                "valid_but_wrong_digit": (
                    self.miss_valid_but_wrong_digit / self.total
                )
                if self.total
                else 0.0,
            },
            "miss_breakdown_over_misses": {
                "invalid_decode": (self.miss_invalid_decode / misses) if misses else 0.0,
                "correct_mantissa_wrong_exponent": (
                    self.miss_correct_mantissa_wrong_exponent / misses
                )
                if misses
                else 0.0,
                "valid_but_wrong_digit": (
                    self.miss_valid_but_wrong_digit / misses
                )
                if misses
                else 0.0,
            },
            "valid_miss_subtypes": {
                "sign_only_mismatch": self.valid_sign_only_mismatch,
                "exponent_only_mismatch": self.valid_exponent_only_mismatch,
                "digit_only_mismatch": self.valid_digit_only_mismatch,
                "digit_and_exponent_mismatch": self.valid_digit_and_exponent_mismatch,
                "sign_and_other_mismatch": self.valid_sign_and_other_mismatch,
            },
        }
        return out


def pct(x: int, den: int) -> float:
    return (100.0 * x / den) if den else 0.0


def classify_valid_miss(tgt_toks: List[int], pred_toks: List[int]) -> str:
    t_sign, t_exp, t_digits = tgt_toks[0], tgt_toks[1], tgt_toks[2:-1]
    p_sign, p_exp, p_digits = pred_toks[0], pred_toks[1], pred_toks[2:-1]

    sign_match = (p_sign == t_sign)
    exp_match = (p_exp == t_exp)
    digits_match = (p_digits == t_digits)

    if sign_match and digits_match and not exp_match:
        return "cmwe"
    return "valid_wrong_digit"


def main() -> None:
    ap = argparse.ArgumentParser(description="Detailed FE-v9 exactness error breakdown")
    ap.add_argument("--ckpt", required=True, help="Path to FE-v9 checkpoint (.pt)")
    ap.add_argument("--data-dir", required=True, help="Path containing val.bin and val_nums.bin")
    ap.add_argument("--batch-blocks", type=int, default=64, help="Validation batch size in 256-token blocks")
    ap.add_argument("--max-blocks", type=int, default=0, help="Cap number of val blocks (0 = full)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-json", default="", help="Optional output JSON path")
    ap.add_argument(
        "--base-exact-rate",
        type=float,
        default=None,
        help="Optional Base exact rate (0..1) to report point-gap decomposition",
    )
    ap.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="How many example contexts to keep per miss category",
    )
    args = ap.parse_args()

    val_bin = os.path.join(args.data_dir, "val.bin")
    val_nums_bin = os.path.join(args.data_dir, "val_nums.bin")
    if not os.path.exists(val_bin):
        raise FileNotFoundError(f"Missing {val_bin}")
    if not os.path.exists(val_nums_bin):
        raise FileNotFoundError(f"Missing {val_nums_bin}")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    print("=" * 80)
    print("FE-v9 Exactness Error Breakdown")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Device:     {device}")
    print(f"Batch blk:  {args.batch_blocks}")
    print(f"Max blocks: {args.max_blocks if args.max_blocks > 0 else 'ALL'}")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    model.to(device)
    model.eval()

    ids = np.memmap(val_bin, dtype=np.uint16, mode="r")
    nums = np.memmap(val_nums_bin, dtype=np.float32, mode="r")
    if len(ids) != len(nums):
        raise ValueError(f"val.bin and val_nums.bin mismatch: {len(ids)} vs {len(nums)}")
    if len(ids) % model.config.block_size != 0:
        raise ValueError(
            f"Validation stream length {len(ids)} is not divisible by block_size {model.config.block_size}"
        )

    total_blocks = len(ids) // model.config.block_size
    if args.max_blocks > 0:
        total_blocks = min(total_blocks, args.max_blocks)

    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None
        print("Warning: tiktoken unavailable; task names may appear as UNKNOWN.")

    global_counts = Counts()
    per_task: Dict[str, Counts] = defaultdict(Counts)
    category_examples: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for b0, b1, x, y, nv, nm in make_batches(
        ids=ids,
        nums=nums,
        block_size=model.config.block_size,
        batch_blocks=args.batch_blocks,
        device=device,
    ):
        if b0 >= total_blocks:
            break
        if b1 > total_blocks:
            keep = total_blocks - b0
            x = x[:keep]
            y = y[:keep]
            nv = nv[:keep]
            nm = nm[:keep]

        logits, _ = model(x, y, num_values=nv, num_mask=nm)
        preds = logits.argmax(dim=-1)

        x_cpu = x.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        p_cpu = preds.detach().cpu().numpy()
        nv_cpu = nv.detach().cpu().numpy()

        bsz, T = y_cpu.shape
        for bi in range(bsz):
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

                task = "UNKNOWN"
                context = ""
                if enc is not None:
                    task_start = 0
                    for s in range(t - 1, -1, -1):
                        if int(x_cpu[bi, s]) == 50256:
                            task_start = s + 1
                            break
                    context = decode_context(
                        token_ids=x_cpu[bi, task_start:t].tolist(),
                        num_values=nv_cpu[bi, task_start:t].tolist(),
                        enc=enc,
                    )
                    task = infer_task(context)

                g = global_counts
                ts = per_task[task]

                if pred_val is None:
                    g.add_invalid()
                    ts.add_invalid()
                    if len(category_examples["miss_invalid_decode"]) < args.show_examples:
                        category_examples["miss_invalid_decode"].append(
                            {
                                "task": task,
                                "context": context,
                                "target_tokens": tgt_toks,
                                "pred_tokens": pred_toks,
                            }
                        )
                    continue

                abs_err = abs(float(tgt_val) - float(pred_val))
                if abs_err == 0.0:
                    g.add_exact()
                    ts.add_exact()
                    continue

                cls = classify_valid_miss(tgt_toks, pred_toks)
                t_sign, t_exp, t_digits = tgt_toks[0], tgt_toks[1], tgt_toks[2:-1]
                p_sign, p_exp, p_digits = pred_toks[0], pred_toks[1], pred_toks[2:-1]
                sign_match = (p_sign == t_sign)
                exp_match = (p_exp == t_exp)
                digits_match = (p_digits == t_digits)

                if cls == "cmwe":
                    g.add_cmwe()
                    ts.add_cmwe()
                    if (
                        len(
                            category_examples["miss_correct_mantissa_wrong_exponent"]
                        )
                        < args.show_examples
                    ):
                        category_examples["miss_correct_mantissa_wrong_exponent"].append(
                            {
                                "task": task,
                                "context": context,
                                "target_tokens": tgt_toks,
                                "pred_tokens": pred_toks,
                                "target": float(tgt_val),
                                "pred": float(pred_val),
                            }
                        )
                else:
                    g.add_valid_wrong(
                        sign_match=sign_match,
                        exp_match=exp_match,
                        digits_match=digits_match,
                    )
                    ts.add_valid_wrong(
                        sign_match=sign_match,
                        exp_match=exp_match,
                        digits_match=digits_match,
                    )
                    if len(category_examples["miss_valid_but_wrong_digit"]) < args.show_examples:
                        category_examples["miss_valid_but_wrong_digit"].append(
                            {
                                "task": task,
                                "context": context,
                                "target_tokens": tgt_toks,
                                "pred_tokens": pred_toks,
                                "target": float(tgt_val),
                                "pred": float(pred_val),
                            }
                        )

    g = global_counts
    gd = g.as_dict()

    print("-" * 80)
    print("Overall")
    print("-" * 80)
    print(f"Number predictions total: {g.total:,}")
    print(f"Exact value predictions:  {g.exact:,} ({pct(g.exact, g.total):.2f}%)")
    print(f"Misses total:             {g.misses:,} ({pct(g.misses, g.total):.2f}%)")
    print()
    print("Miss breakdown (fraction of misses):")
    print(
        "  invalid SME decode:               "
        f"{g.miss_invalid_decode:,} ({pct(g.miss_invalid_decode, g.misses):.2f}%)"
    )
    print(
        "  correct mantissa, wrong exponent: "
        f"{g.miss_correct_mantissa_wrong_exponent:,} "
        f"({pct(g.miss_correct_mantissa_wrong_exponent, g.misses):.2f}%)"
    )
    print(
        "  valid but wrong digit:            "
        f"{g.miss_valid_but_wrong_digit:,} ({pct(g.miss_valid_but_wrong_digit, g.misses):.2f}%)"
    )

    print()
    print("Valid-miss subtypes (diagnostic):")
    valid_misses = g.miss_correct_mantissa_wrong_exponent + g.miss_valid_but_wrong_digit
    print(
        "  sign-only mismatch:               "
        f"{g.valid_sign_only_mismatch:,} ({pct(g.valid_sign_only_mismatch, valid_misses):.2f}%)"
    )
    print(
        "  exponent-only mismatch:           "
        f"{g.valid_exponent_only_mismatch:,} ({pct(g.valid_exponent_only_mismatch, valid_misses):.2f}%)"
    )
    print(
        "  digit-only mismatch:              "
        f"{g.valid_digit_only_mismatch:,} ({pct(g.valid_digit_only_mismatch, valid_misses):.2f}%)"
    )
    print(
        "  digit+exponent mismatch:          "
        f"{g.valid_digit_and_exponent_mismatch:,} "
        f"({pct(g.valid_digit_and_exponent_mismatch, valid_misses):.2f}%)"
    )
    print(
        "  sign+other mismatch:              "
        f"{g.valid_sign_and_other_mismatch:,} "
        f"({pct(g.valid_sign_and_other_mismatch, valid_misses):.2f}%)"
    )

    if args.base_exact_rate is not None:
        fe_exact_rate = (g.exact / g.total) if g.total else 0.0
        base_exact_rate = max(0.0, min(1.0, float(args.base_exact_rate)))
        gap = base_exact_rate - fe_exact_rate
        inv_pts = (g.miss_invalid_decode / g.total) if g.total else 0.0
        cmwe_pts = (g.miss_correct_mantissa_wrong_exponent / g.total) if g.total else 0.0
        wd_pts = (g.miss_valid_but_wrong_digit / g.total) if g.total else 0.0
        print("-" * 80)
        print("Exactness gap decomposition (optional, vs Base)")
        print("-" * 80)
        print(f"Base exact rate: {base_exact_rate:.4f} ({base_exact_rate * 100:.2f} pts)")
        print(f"FE exact rate:   {fe_exact_rate:.4f} ({fe_exact_rate * 100:.2f} pts)")
        print(f"Gap (Base-FE):   {gap:.4f} ({gap * 100:.2f} pts)")
        print(
            f"Miss points by bucket: invalid={inv_pts * 100:.2f}, "
            f"cmwe={cmwe_pts * 100:.2f}, valid_wrong_digit={wd_pts * 100:.2f}"
        )
        print(
            "Upper-bound gain if invalid decode fixed perfectly: "
            f"+{inv_pts * 100:.2f} pts"
        )

    print("-" * 80)
    print("Per-task miss breakdown (fraction of task misses)")
    print("-" * 80)
    print(
        f"{'Task':<12} {'N':>8} {'Exact%':>8} {'Inv%miss':>10} "
        f"{'CMWE%miss':>10} {'WrongDig%miss':>14}"
    )
    for task in sorted(per_task.keys()):
        c = per_task[task]
        misses = c.misses
        print(
            f"{task:<12} {c.total:8d} {pct(c.exact, c.total):8.2f} "
            f"{pct(c.miss_invalid_decode, misses):10.2f} "
            f"{pct(c.miss_correct_mantissa_wrong_exponent, misses):10.2f} "
            f"{pct(c.miss_valid_but_wrong_digit, misses):14.2f}"
        )

    if category_examples:
        print("-" * 80)
        print(f"Sample miss examples (up to {args.show_examples} per category)")
        print("-" * 80)
        for key in [
            "miss_invalid_decode",
            "miss_correct_mantissa_wrong_exponent",
            "miss_valid_but_wrong_digit",
        ]:
            examples = category_examples.get(key, [])
            print(f"[{key}]")
            if not examples:
                print("  (none)")
                continue
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. task={ex.get('task', 'UNKNOWN')}")
                if ex.get("context"):
                    print(f"     context: {ex['context']}")
                print(f"     tgt_toks: {ex['target_tokens']}")
                print(f"     pred_toks:{ex['pred_tokens']}")
                if "target" in ex:
                    print(f"     target: {ex['target']}")
                if "pred" in ex:
                    print(f"     pred:   {ex['pred']}")

    summary = {
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "batch_blocks": args.batch_blocks,
        "max_blocks": args.max_blocks,
        "base_exact_rate": args.base_exact_rate,
        "global": gd,
        "per_task": {task: c.as_dict() for task, c in sorted(per_task.items())},
        "examples": category_examples,
    }

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print("-" * 80)
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
