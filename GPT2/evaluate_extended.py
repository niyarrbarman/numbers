#!/usr/bin/env python3
"""
Extended evaluation comparing Base, FE-Unfreeze, and FE-TextDec.

Analyses:
  1. Conditional MAE — MAE computed only over incorrect (non-exact) numeric predictions
  2. Difficulty buckets — exact match & MAE stratified by digit count, magnitude, list length
  3. SUM length generalization — SUM tasks with list lengths beyond training distribution

Run from the GPT2/ directory:
    python evaluate_extended.py \
        --base-ckpt /path/to/base/ckpt_best.pt \
        --base-data /path/to/numtasks_base_5dig_5m \
        --unfreeze-ckpt /path/to/unfreeze/ckpt_best.pt \
        --unfreeze-data /path/to/numtasks_sme_5dig_5m \
        --textdec-ckpt /path/to/textdec/ckpt_best.pt \
        --textdec-data /path/to/numtasks_textdec_5dig
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

# =============================================================================
# Constants
# =============================================================================

EOT_TOKEN = 50256
NUM_TOKEN_ID = 50257
SPACE_TOKEN = 220  # GPT-2 BPE for " "

NUMERIC_TASKS = {"SORT", "ADD", "SUB", "MIN", "MAX", "SUM", "COUNT"}
LIST_TASKS = {"SORT", "MIN", "MAX", "SUM", "COUNT", "IS_SORTED"}
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class NumericExample:
    """One numeric-task example with parsed results and difficulty features."""
    task: str
    input_nums: List[float]
    target_nums: List[float]
    pred_nums: List[float]
    is_valid: bool        # predicted correct number of output values
    is_exact: bool        # all target values == predicted values
    abs_errs: List[float]
    max_abs_err: float
    # Difficulty features
    list_length: int
    max_target_digits: int
    max_target_magnitude: int


# =============================================================================
# Utility functions
# =============================================================================

def parse_numbers(text: str) -> List[float]:
    return [float(v) for v in NUMBER_RE.findall(text)]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def infer_task(text: str) -> str:
    m = re.match(r"\s*([A-Z_]+):", text)
    return m.group(1) if m else "UNKNOWN"


def digit_count_of_int(x: float) -> int:
    """Number of digits in the integer part of |x|."""
    ax = abs(x)
    if ax < 1:
        return 0
    return len(str(int(ax)))


def digit_bucket(x: float) -> str:
    n = digit_count_of_int(x)
    if n <= 1: return "1-dig"
    if n == 2: return "2-dig"
    if n == 3: return "3-dig"
    if n == 4: return "4-dig"
    if n == 5: return "5-dig"
    return "6+-dig"


def magnitude_bucket(x: float) -> str:
    ax = abs(x)
    if ax < 1e-6:   return "<1e-6"
    if ax < 1:       return "[1e-6,1)"
    if ax < 10:      return "[1,10)"
    if ax < 100:     return "[10,100)"
    if ax < 1000:    return "[100,1K)"
    if ax < 10000:   return "[1K,10K)"
    if ax < 100000:  return "[10K,100K)"
    return ">=100K"


def list_length_bucket(n: int) -> str:
    if n <= 3:  return str(n)
    if n <= 5:  return "4-5"
    if n <= 8:  return "6-8"
    if n <= 10: return "9-10"
    if n <= 15: return "11-15"
    if n <= 20: return "16-20"
    return "21-30"


def max_digit_count(nums: List[float]) -> int:
    if not nums:
        return 0
    return max(digit_count_of_int(n) for n in nums)


def max_magnitude(nums: List[float]) -> int:
    if not nums:
        return 0
    mags = []
    for n in nums:
        if abs(n) > 0:
            mags.append(int(math.floor(math.log10(abs(n)))))
        else:
            mags.append(0)
    return max(mags)


# =============================================================================
# Model loading via importlib
# =============================================================================

def load_module(module_path: str, module_name: str):
    """Load a Python module from file path without namespace conflicts."""
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.abspath(module_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_state_dict(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def load_checkpoint(ckpt_path, model_module, device):
    """Load a checkpoint and return (model, checkpoint_dict)."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = model_module.GPTConfig(**checkpoint["model_args"])
    model = model_module.GPT(config)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


# =============================================================================
# Arrow detection
# =============================================================================

def find_arrow_patterns(enc):
    patterns = []
    for candidate in ["→", " →", " ->"]:
        pat = enc.encode_ordinary(candidate)
        if pat and pat not in patterns:
            patterns.append(pat)
    return patterns


def find_output_start(ex_ids, arrow_patterns, enc, skip_tokens=None):
    """Find token index right after the last arrow pattern."""
    skip = set(skip_tokens or [])
    best_end = -1
    n = len(ex_ids)
    for pat in arrow_patterns:
        m = len(pat)
        if m == 0 or m > n:
            continue
        for i in range(0, n - m + 1):
            if ex_ids[i:i + m] == pat and i + m > best_end:
                best_end = i + m
    if best_end != -1:
        return best_end
    # Fallback: text-based search
    decodable = [t for t in ex_ids if t not in skip and 0 <= t <= 50256]
    if not decodable:
        return -1
    text = enc.decode(decodable)
    arrow_pos = text.rfind("→")
    if arrow_pos == -1:
        return -1
    target_char = arrow_pos + 1
    consumed = 0
    for ti, tok in enumerate(ex_ids):
        if tok in skip or tok > 50256:
            continue
        consumed += len(enc.decode([tok]))
        if consumed >= target_char:
            return ti + 1
    return len(ex_ids)


# =============================================================================
# Teacher-forced forward pass (shared by all models)
# =============================================================================

def run_teacher_forced(model, ids, nums, block_size, batch_blocks, device,
                       has_nums=False, total_blocks=None):
    """Run teacher-forced forward pass, return pred_for_pos array."""
    if total_blocks is None:
        total_blocks = len(ids) // block_size
    total_tokens = total_blocks * block_size
    pred_for_pos = np.full(total_tokens, -1, dtype=np.int32)

    for b0 in range(0, total_blocks, batch_blocks):
        b1 = min(b0 + batch_blocks, total_blocks)
        cur = b1 - b0
        x = np.zeros((cur, block_size), dtype=np.int64)
        y = np.full((cur, block_size), -1, dtype=np.int64)
        nv = np.zeros((cur, block_size), dtype=np.float32) if has_nums else None

        for bi, bidx in enumerate(range(b0, b1)):
            s = bidx * block_size
            x[bi] = ids[s:s + block_size].astype(np.int64)
            y_s = ids[s + 1:s + block_size + 1].astype(np.int64)
            y[bi, :len(y_s)] = y_s
            if has_nums:
                nv[bi] = nums[s:s + block_size].astype(np.float32)

        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)

        if has_nums:
            nvt = torch.from_numpy(nv).to(device)
            nmt = xt.eq(NUM_TOKEN_ID)
            logits, _ = model(xt, yt, num_values=nvt, num_mask=nmt)
        else:
            logits, _ = model(xt, yt)

        preds = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        for bi in range(cur):
            s = (b0 + bi) * block_size + 1
            n = min(block_size, total_tokens - s)
            if s < total_tokens:
                pred_for_pos[s:s + n] = preds[bi, :n]

    return pred_for_pos


# =============================================================================
# Example extraction — text output format (base, textdec)
# =============================================================================

def extract_text_examples(ids_eval, pred_for_pos, enc, arrow_patterns,
                          nums_arr=None) -> List[NumericExample]:
    """Extract numeric examples from a text-output validation set."""
    total_tokens = len(ids_eval)
    eot_positions = np.where(ids_eval == EOT_TOKEN)[0].tolist()
    has_nums = nums_arr is not None
    skip_tokens = {NUM_TOKEN_ID} if has_nums else set()

    examples = []
    ex_start = 0

    for eot in eot_positions:
        if eot <= ex_start:
            ex_start = eot + 1
            continue

        ex_ids = ids_eval[ex_start:eot].tolist()
        ex_start_global = ex_start
        ex_start = eot + 1

        if not ex_ids:
            continue

        # Decode text (skip NUM tokens for FE models)
        decodable = [t for t in ex_ids if t not in skip_tokens]
        if not decodable:
            continue
        full_text = enc.decode(decodable)
        if "→" not in full_text:
            continue

        task = infer_task(full_text)
        if task not in NUMERIC_TASKS:
            continue

        out_rel = find_output_start(ex_ids, arrow_patterns, enc,
                                    skip_tokens=skip_tokens)
        if out_rel < 0 or out_rel >= len(ex_ids):
            continue

        tgt_out_ids = ex_ids[out_rel:]
        global_out_start = ex_start_global + out_rel

        # Get predicted output tokens
        pred_out_ids = []
        missing = False
        for g in range(global_out_start, ex_start_global + len(ex_ids)):
            pid = int(pred_for_pos[g])
            if pid < 0:
                missing = True
                break
            pred_out_ids.append(pid)

        if missing or len(pred_out_ids) != len(tgt_out_ids) or not tgt_out_ids:
            continue

        # Parse target and predicted numbers from decoded text
        tgt_text = normalize_text(enc.decode(tgt_out_ids))
        pred_text = normalize_text(enc.decode(pred_out_ids))
        tgt_nums = parse_numbers(tgt_text)
        pred_nums = parse_numbers(pred_text)

        if not tgt_nums:
            continue

        # Get input numbers
        if has_nums:
            input_nums = []
            for i, tok in enumerate(ex_ids[:out_rel]):
                if tok == NUM_TOKEN_ID:
                    gpos = ex_start_global + i
                    if gpos < len(nums_arr):
                        input_nums.append(float(nums_arr[gpos]))
        else:
            before_arrow = full_text.split("→")[0]
            m = re.match(r"\s*[A-Z_]+:\s*(.*)", before_arrow)
            if m:
                num_part = m.group(1)
                num_part = re.sub(r'\b(vs|CHECK)\b', ' ', num_part)
                num_part = num_part.replace('+', ' ').replace('=', ' ')
                input_nums = parse_numbers(num_part)
            else:
                input_nums = []

        list_length = len(input_nums)
        is_valid = len(pred_nums) == len(tgt_nums)
        if is_valid:
            abs_errs = [abs(t - p) for t, p in zip(tgt_nums, pred_nums)]
            is_exact = all(e == 0.0 for e in abs_errs)
            max_ae = max(abs_errs) if abs_errs else 0.0
        else:
            abs_errs = []
            is_exact = False
            max_ae = float('inf')

        examples.append(NumericExample(
            task=task,
            input_nums=input_nums,
            target_nums=tgt_nums,
            pred_nums=pred_nums if is_valid else [],
            is_valid=is_valid,
            is_exact=is_exact,
            abs_errs=abs_errs,
            max_abs_err=max_ae,
            list_length=list_length,
            max_target_digits=max_digit_count(tgt_nums),
            max_target_magnitude=max_magnitude(tgt_nums),
        ))

    return examples


# =============================================================================
# Example extraction — SME output format (unfreeze)
# =============================================================================

def extract_sme_examples(ids_eval, pred_for_pos, nums_arr, enc,
                         arrow_patterns, sme_module) -> List[NumericExample]:
    """Extract numeric examples from an SME-output validation set."""
    total_tokens = len(ids_eval)
    eot_positions = np.where(ids_eval == EOT_TOKEN)[0].tolist()

    parse_sme = sme_module.parse_sme_number_tokens
    sme_to_num = sme_module.sme_tokens_to_number
    SME_SIGN_POS = sme_module.SME_SIGN_POS
    SME_SIGN_NEG = sme_module.SME_SIGN_NEG
    SME_ALL = sme_module.SME_ALL_TOKENS
    skip_tokens = {NUM_TOKEN_ID} | SME_ALL

    examples = []
    ex_start = 0

    for eot in eot_positions:
        if eot <= ex_start:
            ex_start = eot + 1
            continue

        ex_ids = ids_eval[ex_start:eot].tolist()
        ex_start_global = ex_start
        ex_start = eot + 1

        if not ex_ids:
            continue

        # Decode text (skip NUM and SME tokens)
        decodable = [t for t in ex_ids
                     if t not in skip_tokens and 0 <= t <= 50256]
        if not decodable:
            continue
        full_text = enc.decode(decodable)
        if "→" not in full_text:
            continue

        task = infer_task(full_text)
        if task not in NUMERIC_TASKS:
            continue

        out_rel = find_output_start(ex_ids, arrow_patterns, enc,
                                    skip_tokens=skip_tokens)
        if out_rel < 0 or out_rel >= len(ex_ids):
            continue

        tgt_out_ids = ex_ids[out_rel:]
        global_out_start = ex_start_global + out_rel

        # Get predicted output tokens
        pred_out_ids = []
        missing = False
        for g in range(global_out_start, ex_start_global + len(ex_ids)):
            pid = int(pred_for_pos[g])
            if pid < 0:
                missing = True
                break
            pred_out_ids.append(pid)

        if missing or len(pred_out_ids) != len(tgt_out_ids) or not tgt_out_ids:
            continue

        # Parse SME numbers from target and predicted token sequences
        tgt_nums = _parse_all_sme(tgt_out_ids, parse_sme, sme_to_num,
                                  SME_SIGN_POS, SME_SIGN_NEG)
        pred_nums = _parse_all_sme(pred_out_ids, parse_sme, sme_to_num,
                                   SME_SIGN_POS, SME_SIGN_NEG)

        if not tgt_nums:
            continue

        # Get input numbers from nums array
        input_nums = []
        for i, tok in enumerate(ex_ids[:out_rel]):
            if tok == NUM_TOKEN_ID:
                gpos = ex_start_global + i
                if gpos < len(nums_arr):
                    input_nums.append(float(nums_arr[gpos]))

        list_length = len(input_nums)
        is_valid = len(pred_nums) == len(tgt_nums)
        if is_valid:
            abs_errs = [abs(t - p) for t, p in zip(tgt_nums, pred_nums)]
            is_exact = all(e == 0.0 for e in abs_errs)
            max_ae = max(abs_errs) if abs_errs else 0.0
        else:
            abs_errs = []
            is_exact = False
            max_ae = float('inf')

        examples.append(NumericExample(
            task=task,
            input_nums=input_nums,
            target_nums=tgt_nums,
            pred_nums=pred_nums if is_valid else [],
            is_valid=is_valid,
            is_exact=is_exact,
            abs_errs=abs_errs,
            max_abs_err=max_ae,
            list_length=list_length,
            max_target_digits=max_digit_count(tgt_nums),
            max_target_magnitude=max_magnitude(tgt_nums),
        ))

    return examples


def _parse_all_sme(token_ids, parse_fn, to_num_fn, sign_pos, sign_neg):
    """Parse all SME numbers from a token sequence."""
    nums = []
    pos = 0
    while pos < len(token_ids):
        if token_ids[pos] in (sign_pos, sign_neg):
            parsed, next_pos = parse_fn(token_ids, start_idx=pos)
            if parsed is not None:
                val = to_num_fn(parsed)
                if val is not None:
                    nums.append(val)
                pos = next_pos
                continue
        pos += 1
    return nums


# =============================================================================
# Analysis 1: Conditional MAE
# =============================================================================

def compute_conditional_mae(examples: List[NumericExample]) -> Dict:
    """MAE computed only over non-exact (incorrect) examples."""
    by_task = defaultdict(lambda: {
        "all_errs": [], "wrong_errs": [],
        "n_total": 0, "n_exact": 0, "n_valid": 0
    })

    for ex in examples:
        stats = by_task[ex.task]
        stats["n_total"] += 1
        if not ex.is_valid:
            continue
        stats["n_valid"] += 1
        stats["all_errs"].extend(ex.abs_errs)
        if ex.is_exact:
            stats["n_exact"] += 1
        else:
            stats["wrong_errs"].extend(ex.abs_errs)

    result = {}
    all_errs = []
    all_wrong_errs = []
    total = 0
    total_exact = 0

    for task in sorted(by_task.keys()):
        s = by_task[task]
        mae = float(np.mean(s["all_errs"])) if s["all_errs"] else 0.0
        cond_mae = float(np.mean(s["wrong_errs"])) if s["wrong_errs"] else 0.0
        exact_rate = s["n_exact"] / s["n_total"] if s["n_total"] else 0.0
        n_wrong = s["n_valid"] - s["n_exact"]
        result[task] = {
            "n": s["n_total"],
            "n_valid": s["n_valid"],
            "n_exact": s["n_exact"],
            "n_wrong": n_wrong,
            "exact_rate": exact_rate,
            "mae": mae,
            "conditional_mae": cond_mae,
        }
        all_errs.extend(s["all_errs"])
        all_wrong_errs.extend(s["wrong_errs"])
        total += s["n_total"]
        total_exact += s["n_exact"]

    result["_overall"] = {
        "n": total,
        "n_exact": total_exact,
        "exact_rate": total_exact / total if total else 0.0,
        "mae": float(np.mean(all_errs)) if all_errs else 0.0,
        "conditional_mae": float(np.mean(all_wrong_errs)) if all_wrong_errs else 0.0,
    }
    return result


# =============================================================================
# Analysis 2: Difficulty buckets
# =============================================================================

def _bucket_stats(exs):
    """Compute stats for a bucket of examples."""
    n = len(exs)
    if n == 0:
        return {"n": 0, "exact_rate": 0, "mae": 0, "cond_mae": 0}
    exact = sum(1 for e in exs if e.is_exact)
    valid = [e for e in exs if e.is_valid]
    all_errs = [ae for e in valid for ae in e.abs_errs]
    wrong_errs = [ae for e in valid if not e.is_exact for ae in e.abs_errs]
    return {
        "n": n,
        "exact_rate": exact / n,
        "mae": float(np.mean(all_errs)) if all_errs else 0.0,
        "cond_mae": float(np.mean(wrong_errs)) if wrong_errs else 0.0,
    }


def compute_difficulty_buckets(examples: List[NumericExample]) -> Dict:
    """Compute metrics stratified by digit count, magnitude, list length."""
    result = {}

    # By digit count of max target value
    digit_groups = defaultdict(list)
    for ex in examples:
        if ex.target_nums:
            bucket = digit_bucket(max(abs(n) for n in ex.target_nums))
            digit_groups[bucket].append(ex)
    digit_order = ["1-dig", "2-dig", "3-dig", "4-dig", "5-dig", "6+-dig"]
    result["by_digits"] = {b: _bucket_stats(digit_groups.get(b, []))
                           for b in digit_order}

    # By magnitude of max target value
    mag_groups = defaultdict(list)
    for ex in examples:
        if ex.target_nums:
            bucket = magnitude_bucket(max(abs(n) for n in ex.target_nums))
            mag_groups[bucket].append(ex)
    mag_order = ["<1e-6", "[1e-6,1)", "[1,10)", "[10,100)", "[100,1K)",
                 "[1K,10K)", "[10K,100K)", ">=100K"]
    result["by_magnitude"] = {b: _bucket_stats(mag_groups.get(b, []))
                              for b in mag_order}

    # By list length (list tasks only)
    len_groups = defaultdict(list)
    for ex in examples:
        if ex.task in LIST_TASKS:
            bucket = list_length_bucket(ex.list_length)
            len_groups[bucket].append(ex)
    len_order = ["2", "3", "4-5", "6-8", "9-10"]
    result["by_list_length"] = {b: _bucket_stats(len_groups.get(b, []))
                                for b in len_order}

    # Per-task by digit count
    result["per_task_digits"] = {}
    for task in sorted(NUMERIC_TASKS):
        task_exs = [e for e in examples if e.task == task]
        task_digit_groups = defaultdict(list)
        for ex in task_exs:
            if ex.target_nums:
                bucket = digit_bucket(max(abs(n) for n in ex.target_nums))
                task_digit_groups[bucket].append(ex)
        result["per_task_digits"][task] = {
            b: _bucket_stats(task_digit_groups.get(b, []))
            for b in digit_order
        }

    return result


# =============================================================================
# Analysis 3: SUM length generalization
# =============================================================================

def generate_sum_examples(list_lengths, n_per_length, number_range=100,
                          seed=42):
    """Generate SUM tasks with specific list lengths.

    Uses simple integers for clean verification.
    Returns list of (numbers, correct_sum, list_length).
    """
    rng = random.Random(seed)
    examples = []
    for length in list_lengths:
        for _ in range(n_per_length):
            nums = [rng.randint(1, number_range) for _ in range(length)]
            examples.append((nums, sum(nums), length))
    return examples


def tokenize_sum_for_base(nums, result, enc):
    """Tokenize a SUM example for base model (all text)."""
    input_text = "SUM: " + " ".join(str(n) for n in nums) + " →"
    output_text = " " + str(result)
    ids = enc.encode_ordinary(input_text + output_text) + [EOT_TOKEN]
    return ids, None


def tokenize_sum_for_fe_text(nums, result, enc, process_fn):
    """Tokenize a SUM example for FE-TextDec (NUM input + text output)."""
    input_text = "SUM: " + " ".join(str(n) for n in nums) + " →"
    ids, num_vals = process_fn(input_text)
    # Remove trailing EOT (process_fn appends one)
    ids = ids[:-1]
    num_vals = num_vals[:-1]
    out_ids = enc.encode_ordinary(" " + str(result))
    ids.extend(out_ids)
    num_vals.extend([0.0] * len(out_ids))
    ids.append(EOT_TOKEN)
    num_vals.append(0.0)
    return ids, num_vals


def tokenize_sum_for_fe_sme(nums, result, enc, process_fn, sme_fn):
    """Tokenize a SUM example for FE-Unfreeze (NUM input + SME output)."""
    input_text = "SUM: " + " ".join(str(n) for n in nums) + " →"
    ids, num_vals = process_fn(input_text)
    ids = ids[:-1]
    num_vals = num_vals[:-1]
    # Space token before SME (matches training data format)
    ids.append(SPACE_TOKEN)
    num_vals.append(0.0)
    sme = sme_fn(result)
    ids.extend(sme)
    num_vals.extend([0.0] * len(sme))
    ids.append(EOT_TOKEN)
    num_vals.append(0.0)
    return ids, num_vals


def pack_examples(tokenized_examples, block_size):
    """Pack tokenized examples into block-aligned arrays.

    Args:
        tokenized_examples: list of (ids, nums_or_None) tuples
        block_size: int

    Returns:
        (ids_arr, nums_arr_or_None, n_blocks)
    """
    has_nums = any(n is not None for _, n in tokenized_examples)
    all_ids = []
    all_nums = []
    block_ids = []
    block_nums = []
    n_blocks = 0

    for ids, nums in tokenized_examples:
        if len(ids) > block_size:
            continue
        if len(block_ids) + len(ids) > block_size:
            pad = block_size - len(block_ids)
            block_ids.extend([EOT_TOKEN] * pad)
            if has_nums:
                block_nums.extend([0.0] * pad)
            all_ids.extend(block_ids)
            all_nums.extend(block_nums)
            n_blocks += 1
            block_ids = []
            block_nums = []
        block_ids.extend(ids)
        if has_nums:
            block_nums.extend(nums if nums else [0.0] * len(ids))

    if block_ids:
        pad = block_size - len(block_ids)
        block_ids.extend([EOT_TOKEN] * pad)
        if has_nums:
            block_nums.extend([0.0] * pad)
        all_ids.extend(block_ids)
        all_nums.extend(block_nums)
        n_blocks += 1

    ids_arr = np.array(all_ids, dtype=np.uint16)
    nums_arr = np.array(all_nums, dtype=np.float32) if has_nums else None
    return ids_arr, nums_arr, n_blocks


def run_sum_gen_for_model(model, sum_examples, tokenize_fn, enc,
                          arrow_patterns, block_size, batch_blocks,
                          device, has_nums, extract_fn_kwargs):
    """Run SUM generalization test for one model.

    Returns dict mapping list_length -> {n, exact_rate, mae, cond_mae}.
    """
    # Tokenize all SUM examples
    tokenized = []
    for nums, result, length in sum_examples:
        ids, nvals = tokenize_fn(nums, result)
        tokenized.append((ids, nvals))

    ids_packed, nums_packed, n_blocks = pack_examples(tokenized, block_size)
    if n_blocks == 0:
        return {}

    # Forward pass
    pred_for_pos = run_teacher_forced(
        model, ids_packed, nums_packed, block_size, batch_blocks, device,
        has_nums=has_nums, total_blocks=n_blocks
    )

    tt = n_blocks * block_size
    ids_eval = ids_packed[:tt].astype(np.int64)
    nums_eval = nums_packed[:tt] if nums_packed is not None else None

    # Extract examples
    sme_module = extract_fn_kwargs.get("sme_module")
    if sme_module is not None:
        examples = extract_sme_examples(
            ids_eval, pred_for_pos, nums_eval, enc, arrow_patterns, sme_module
        )
    else:
        examples = extract_text_examples(
            ids_eval, pred_for_pos, enc, arrow_patterns,
            nums_arr=nums_eval
        )

    # Group by list length
    by_len = defaultdict(lambda: {
        "n": 0, "exact": 0, "errs": [], "wrong_errs": []
    })
    for ex in examples:
        if ex.task != "SUM":
            continue
        s = by_len[ex.list_length]
        s["n"] += 1
        if ex.is_valid:
            s["errs"].extend(ex.abs_errs)
            if ex.is_exact:
                s["exact"] += 1
            else:
                s["wrong_errs"].extend(ex.abs_errs)

    result = {}
    for length in sorted(by_len.keys()):
        s = by_len[length]
        result[length] = {
            "n": s["n"],
            "exact_rate": s["exact"] / s["n"] if s["n"] else 0.0,
            "mae": float(np.mean(s["errs"])) if s["errs"] else 0.0,
            "cond_mae": float(np.mean(s["wrong_errs"])) if s["wrong_errs"] else 0.0,
        }
    return result


# =============================================================================
# Reporting
# =============================================================================

def print_conditional_mae(results: Dict[str, Dict], model_names: List[str]):
    """Print Analysis 1: Conditional MAE comparison table."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: CONDITIONAL MAE")
    print("  MAE over incorrect (non-exact-match) examples only")
    print("=" * 80)

    # Header
    print(f"\n{'':12s}", end="")
    for name in model_names:
        print(f"  {'--- ' + name + ' ---':^33s}", end="")
    print()
    header = f"{'Task':<12}"
    for _ in model_names:
        header += f"  {'Exact%':>7s} {'MAE':>10s} {'CondMAE':>10s}"
    print(header)
    print("-" * (12 + 33 * len(model_names)))

    all_tasks = sorted(
        set().union(*(set(r.keys()) - {"_overall"} for r in results.values()))
    )
    for task in all_tasks:
        row = f"{task:<12}"
        for name in model_names:
            s = results[name].get(task, {
                "exact_rate": 0, "mae": 0, "conditional_mae": 0
            })
            row += f"  {s['exact_rate'] * 100:6.1f}% {s['mae']:10.2f} {s['conditional_mae']:10.2f}"
        print(row)

    # Overall
    print("-" * (12 + 33 * len(model_names)))
    row = f"{'OVERALL':<12}"
    for name in model_names:
        s = results[name]["_overall"]
        row += f"  {s['exact_rate'] * 100:6.1f}% {s['mae']:10.2f} {s['conditional_mae']:10.2f}"
    print(row)


def print_difficulty_buckets(results: Dict[str, Dict], model_names: List[str]):
    """Print Analysis 2: Difficulty bucket comparison tables."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: DIFFICULTY-CONTROLLED EVALUATION BUCKETS")
    print("=" * 80)

    for bucket_type, label in [
        ("by_digits", "BY DIGIT COUNT (of max |target|)"),
        ("by_magnitude", "BY MAGNITUDE (of max |target|)"),
        ("by_list_length", "BY LIST LENGTH (SORT/MIN/MAX/SUM/COUNT only)"),
    ]:
        print(f"\n--- {label} ---")
        print(f"{'':12s}", end="")
        for name in model_names:
            print(f"  {'--- ' + name + ' ---':^37s}", end="")
        print()
        header = f"{'Bucket':<12}"
        for _ in model_names:
            header += f"  {'N':>6s} {'Exact%':>7s} {'MAE':>10s} {'CondMAE':>10s}"
        print(header)
        print("-" * (12 + 37 * len(model_names)))

        first = results[model_names[0]]
        buckets = list(first[bucket_type].keys())

        for b in buckets:
            row = f"{b:<12}"
            for name in model_names:
                s = results[name][bucket_type].get(
                    b, {"n": 0, "exact_rate": 0, "mae": 0, "cond_mae": 0}
                )
                if s["n"] > 0:
                    row += (f"  {s['n']:6d} {s['exact_rate'] * 100:6.1f}% "
                            f"{s['mae']:10.2f} {s['cond_mae']:10.2f}")
                else:
                    row += f"  {0:6d} {'---':>7s} {'---':>10s} {'---':>10s}"
            print(row)

    # Per-task digit breakdown (compact)
    print(f"\n--- PER-TASK EXACT MATCH % BY DIGIT COUNT ---")
    digit_order = ["1-dig", "2-dig", "3-dig", "4-dig", "5-dig", "6+-dig"]
    for task in sorted(NUMERIC_TASKS):
        print(f"\n  {task}:")
        print(f"  {'Digits':<8}", end="")
        for name in model_names:
            print(f"  {name:>12s}", end="")
        print()
        for b in digit_order:
            row = f"  {b:<8}"
            any_data = False
            for name in model_names:
                s = results[name]["per_task_digits"].get(task, {}).get(
                    b, {"n": 0, "exact_rate": 0}
                )
                if s["n"] > 0:
                    row += f"  {s['exact_rate'] * 100:11.1f}%"
                    any_data = True
                else:
                    row += f"  {'---':>12s}"
            if any_data:
                print(row)


def print_sum_generalization(results: Dict[str, Dict], model_names: List[str],
                             training_max_len=8):
    """Print Analysis 3: SUM length generalization."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: SUM LENGTH GENERALIZATION")
    print(f"  Training range: 2-{training_max_len} numbers per SUM")
    print(f"  Test: integers 1-100, teacher-forced evaluation")
    print("=" * 80)

    print(f"\n{'':10s}", end="")
    for name in model_names:
        print(f"  {'--- ' + name + ' ---':^37s}", end="")
    print()
    header = f"{'Length':<10}"
    for _ in model_names:
        header += f"  {'N':>6s} {'Exact%':>7s} {'MAE':>10s} {'CondMAE':>10s}"
    print(header)
    print("-" * (10 + 37 * len(model_names)))

    all_lengths = sorted(
        set().union(*(set(r.keys()) for r in results.values()))
    )
    for length in all_lengths:
        ood = " *" if length > training_max_len else ""
        row = f"{str(length) + ood:<10}"
        for name in model_names:
            s = results[name].get(length, {
                "n": 0, "exact_rate": 0, "mae": 0, "cond_mae": 0
            })
            if s["n"] > 0:
                row += (f"  {s['n']:6d} {s['exact_rate'] * 100:6.1f}% "
                        f"{s['mae']:10.2f} {s['cond_mae']:10.2f}")
            else:
                row += f"  {0:6d} {'---':>7s} {'---':>10s} {'---':>10s}"
        print(row)

    print(f"\n  * = out-of-distribution (list length > {training_max_len})")


# =============================================================================
# Process one model variant
# =============================================================================

def process_model(name, model, data_dir, device, enc, arrow_patterns,
                  batch_blocks, max_blocks, sum_examples,
                  has_nums=False, output_format="text",
                  sme_module=None, prepare_module=None):
    """Process one model: val set analysis + SUM generalization.

    Returns (examples, cond_mae, buckets, sum_gen).
    """
    block_size = model.config.block_size

    # --- Load validation data ---
    val_bin = os.path.join(data_dir, "val.bin")
    ids = np.memmap(val_bin, dtype=np.uint16, mode="r")
    nums = None
    if has_nums:
        val_nums_bin = os.path.join(data_dir, "val_nums.bin")
        nums = np.memmap(val_nums_bin, dtype=np.float32, mode="r")

    total_blocks = len(ids) // block_size
    if max_blocks > 0:
        total_blocks = min(total_blocks, max_blocks)
    total_tokens = total_blocks * block_size
    print(f"  Val tokens: {total_tokens:,}, blocks: {total_blocks:,}")

    # --- Teacher-forced forward pass ---
    pred_for_pos = run_teacher_forced(
        model, ids, nums, block_size, batch_blocks, device,
        has_nums=has_nums, total_blocks=total_blocks
    )
    ids_eval = ids[:total_tokens].astype(np.int64)
    nums_eval = nums[:total_tokens] if nums is not None else None

    # --- Extract examples ---
    if output_format == "sme":
        examples = extract_sme_examples(
            ids_eval, pred_for_pos, nums_eval, enc, arrow_patterns, sme_module
        )
    else:
        examples = extract_text_examples(
            ids_eval, pred_for_pos, enc, arrow_patterns, nums_arr=nums_eval
        )
    print(f"  Extracted {len(examples)} numeric examples")

    # --- Analyses 1 & 2 ---
    cond_mae = compute_conditional_mae(examples)
    buckets = compute_difficulty_buckets(examples)

    # --- Analysis 3: SUM generalization ---
    print("  Running SUM generalization...")
    if output_format == "sme":
        process_fn = sme_module.process_text_with_numbers
        sme_fn = sme_module.number_to_sme_tokens
        tok_fn = lambda nums, result: tokenize_sum_for_fe_sme(
            nums, result, enc, process_fn, sme_fn)
        extract_kwargs = {"sme_module": sme_module}
    elif has_nums:
        process_fn = prepare_module.process_text_with_numbers
        tok_fn = lambda nums, result: tokenize_sum_for_fe_text(
            nums, result, enc, process_fn)
        extract_kwargs = {}
    else:
        tok_fn = lambda nums, result: tokenize_sum_for_base(nums, result, enc)
        extract_kwargs = {}

    sum_gen = run_sum_gen_for_model(
        model, sum_examples, tok_fn, enc, arrow_patterns,
        block_size, batch_blocks, device, has_nums=has_nums,
        extract_fn_kwargs=extract_kwargs
    )

    return examples, cond_mae, buckets, sum_gen


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extended evaluation: conditional MAE, difficulty "
                    "buckets, SUM generalization"
    )
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--base-data", type=str, required=True)
    parser.add_argument("--unfreeze-ckpt", type=str, required=True)
    parser.add_argument("--unfreeze-data", type=str, required=True)
    parser.add_argument("--textdec-ckpt", type=str, required=True)
    parser.add_argument("--textdec-data", type=str, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-blocks", type=int, default=64)
    parser.add_argument("--max-blocks", type=int, default=0,
                        help="0 = full val set")
    parser.add_argument("--sum-gen-count", type=int, default=200,
                        help="SUM examples per list length")
    parser.add_argument("--sum-gen-seed", type=int, default=42)
    parser.add_argument("--sum-gen-range", type=int, default=100,
                        help="Max integer for SUM generation")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    enc = tiktoken.get_encoding("gpt2")
    arrow_patterns = find_arrow_patterns(enc)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_names = ["Base", "FE-Unfreeze", "FE-TextDec"]

    # --- Load modules via importlib ---
    print("Loading model modules...")

    # Add parent dir for np_emb_torch (used by FE model modules)
    parent_dir = os.path.join(script_dir, "..")
    if os.path.abspath(parent_dir) not in [os.path.abspath(p) for p in sys.path]:
        sys.path.insert(0, os.path.abspath(parent_dir))

    base_module = load_module(
        os.path.join(script_dir, "base", "base.py"), "base_model_mod")
    unfreeze_module = load_module(
        os.path.join(script_dir, "fe_unfreeze", "model.py"), "unfreeze_model_mod")
    textdec_module = load_module(
        os.path.join(script_dir, "fe_textdec", "model.py"), "textdec_model_mod")

    # Prepare modules (for tokenization and SME parsing)
    sme_prepare = load_module(
        os.path.join(script_dir, "fe_unfreeze", "prepare.py"), "sme_prepare_mod")
    textdec_prepare = load_module(
        os.path.join(script_dir, "fe_textdec", "prepare.py"), "textdec_prepare_mod")

    # --- Generate SUM examples (shared across all models) ---
    sum_lengths = [2, 3, 5, 8, 10, 15, 20, 30]
    sum_examples = generate_sum_examples(
        sum_lengths, args.sum_gen_count,
        number_range=args.sum_gen_range, seed=args.sum_gen_seed
    )
    print(f"Generated {len(sum_examples)} SUM examples for "
          f"lengths {sum_lengths}")

    all_cond_mae = {}
    all_buckets = {}
    all_sum_gen = {}

    # =================================================================
    # Process Base model
    # =================================================================
    print("\n" + "=" * 70)
    print("Processing Base model...")
    print("=" * 70)
    model, ckpt = load_checkpoint(args.base_ckpt, base_module, device)
    print(f"  Checkpoint iter: {ckpt.get('iter_num', 'N/A')}")
    print(f"  Block size: {model.config.block_size}")

    _, cond_mae, buckets, sum_gen = process_model(
        "Base", model, args.base_data, device, enc, arrow_patterns,
        args.batch_blocks, args.max_blocks, sum_examples,
        has_nums=False, output_format="text"
    )
    all_cond_mae["Base"] = cond_mae
    all_buckets["Base"] = buckets
    all_sum_gen["Base"] = sum_gen

    del model
    torch.cuda.empty_cache()
    print("  Base model done.")

    # =================================================================
    # Process FE-Unfreeze model
    # =================================================================
    print("\n" + "=" * 70)
    print("Processing FE-Unfreeze model...")
    print("=" * 70)
    model, ckpt = load_checkpoint(args.unfreeze_ckpt, unfreeze_module, device)
    print(f"  Checkpoint iter: {ckpt.get('iter_num', 'N/A')}")
    print(f"  Block size: {model.config.block_size}")

    _, cond_mae, buckets, sum_gen = process_model(
        "FE-Unfreeze", model, args.unfreeze_data, device, enc, arrow_patterns,
        args.batch_blocks, args.max_blocks, sum_examples,
        has_nums=True, output_format="sme", sme_module=sme_prepare
    )
    all_cond_mae["FE-Unfreeze"] = cond_mae
    all_buckets["FE-Unfreeze"] = buckets
    all_sum_gen["FE-Unfreeze"] = sum_gen

    del model
    torch.cuda.empty_cache()
    print("  FE-Unfreeze model done.")

    # =================================================================
    # Process FE-TextDec model
    # =================================================================
    print("\n" + "=" * 70)
    print("Processing FE-TextDec model...")
    print("=" * 70)
    model, ckpt = load_checkpoint(args.textdec_ckpt, textdec_module, device)
    print(f"  Checkpoint iter: {ckpt.get('iter_num', 'N/A')}")
    print(f"  Block size: {model.config.block_size}")

    _, cond_mae, buckets, sum_gen = process_model(
        "FE-TextDec", model, args.textdec_data, device, enc, arrow_patterns,
        args.batch_blocks, args.max_blocks, sum_examples,
        has_nums=True, output_format="text", prepare_module=textdec_prepare
    )
    all_cond_mae["FE-TextDec"] = cond_mae
    all_buckets["FE-TextDec"] = buckets
    all_sum_gen["FE-TextDec"] = sum_gen

    del model
    torch.cuda.empty_cache()
    print("  FE-TextDec model done.")

    # =================================================================
    # Print results
    # =================================================================
    print_conditional_mae(all_cond_mae, model_names)
    print_difficulty_buckets(all_buckets, model_names)
    print_sum_generalization(all_sum_gen, model_names, training_max_len=8)

    # Save JSON
    if args.output_json:
        summary = {
            "conditional_mae": all_cond_mae,
            "difficulty_buckets": all_buckets,
            "sum_generalization": {
                name: {str(k): v for k, v in data.items()}
                for name, data in all_sum_gen.items()
            },
        }
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved JSON summary: {args.output_json}")

    print("\n" + "=" * 80)
    print("Extended evaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
