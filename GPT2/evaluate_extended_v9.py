#!/usr/bin/env python3
"""
Extended evaluation for FE-v9 (SME output) only.

Analyses:
  1. Conditional MAE
  2. Difficulty buckets
  3. SUM length generalization
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import tiktoken

from evaluate_extended import (
    find_arrow_patterns,
    generate_sum_examples,
    load_checkpoint,
    load_module,
    print_conditional_mae,
    print_difficulty_buckets,
    print_sum_generalization,
    process_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended evaluation for FE-v9 only "
                    "(conditional MAE, difficulty buckets, SUM generalization)"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-blocks", type=int, default=64)
    parser.add_argument("--max-blocks", type=int, default=0, help="0 = full val set")
    parser.add_argument(
        "--sum-gen-count",
        type=int,
        default=200,
        help="SUM examples per list length",
    )
    parser.add_argument("--sum-gen-seed", type=int, default=42)
    parser.add_argument(
        "--sum-gen-range",
        type=int,
        default=100,
        help="Max integer for SUM generation",
    )
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    enc = tiktoken.get_encoding("gpt2")
    arrow_patterns = find_arrow_patterns(enc)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure parent project root is visible for np_emb_v9 imports used by fe_v9/model.py.
    parent_dir = os.path.join(script_dir, "..")
    parent_dir_abs = os.path.abspath(parent_dir)
    if parent_dir_abs not in [os.path.abspath(p) for p in sys.path]:
        sys.path.insert(0, parent_dir_abs)

    print("=" * 70)
    print("Extended Evaluation: FE-v9")
    print("=" * 70)
    print(f"Checkpoint:   {args.ckpt}")
    print(f"Data dir:     {args.data_dir}")
    print(f"Device:       {device}")
    print(f"Batch blocks: {args.batch_blocks}")
    print(f"Max blocks:   {args.max_blocks if args.max_blocks > 0 else 'ALL'}")

    model_module = load_module(
        os.path.join(script_dir, "fe_v9", "model.py"),
        "v9_model_mod",
    )
    prepare_module = load_module(
        os.path.join(script_dir, "fe_v9", "prepare.py"),
        "v9_prepare_mod",
    )

    sum_lengths = [2, 3, 5, 8, 10, 15, 20, 30]
    sum_examples = generate_sum_examples(
        sum_lengths,
        args.sum_gen_count,
        number_range=args.sum_gen_range,
        seed=args.sum_gen_seed,
    )
    print(
        f"Generated {len(sum_examples)} SUM examples for lengths {sum_lengths} "
        f"(per length={args.sum_gen_count})"
    )

    model, checkpoint = load_checkpoint(args.ckpt, model_module, device)
    print(f"Checkpoint iter: {checkpoint.get('iter_num', 'N/A')}")
    print(f"Best val loss:   {checkpoint.get('best_val_loss', 'N/A')}")

    _, cond_mae, buckets, sum_gen = process_model(
        "FE-v9",
        model,
        args.data_dir,
        device,
        enc,
        arrow_patterns,
        args.batch_blocks,
        args.max_blocks,
        sum_examples,
        has_nums=True,
        output_format="sme",
        sme_module=prepare_module,
    )

    results_cond = {"FE-v9": cond_mae}
    results_buckets = {"FE-v9": buckets}
    results_sum = {"FE-v9": sum_gen}
    model_names = ["FE-v9"]

    print_conditional_mae(results_cond, model_names)
    print_difficulty_buckets(results_buckets, model_names)
    print_sum_generalization(results_sum, model_names, training_max_len=8)

    if args.output_json:
        summary = {
            "model": "FE-v9",
            "checkpoint": args.ckpt,
            "data_dir": args.data_dir,
            "checkpoint_iter": checkpoint.get("iter_num", None),
            "best_val_loss": checkpoint.get("best_val_loss", None),
            "conditional_mae": results_cond,
            "difficulty_buckets": results_buckets,
            "sum_generalization": {
                name: {str(k): v for k, v in data.items()}
                for name, data in results_sum.items()
            },
        }
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved extended JSON: {args.output_json}")


if __name__ == "__main__":
    main()
