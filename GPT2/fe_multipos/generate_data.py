"""
Generate synthetic numerical task data for multi-position number-aware GPT-2.

Same as the parent generate_data.py but uses multi-position tokenization:
each input number produces NUM_POSITIONS consecutive <NUM> tokens instead
of a single token. This gives the transformer multiple attention targets
per input number.

Output also includes {split}_pos.bin with position indices (int8).

Usage:
    python fe_multipos/generate_data.py --out-dir /path/to/output
"""

import os
import sys
import argparse
import pickle
import random
import math

import numpy as np
from tqdm import tqdm

# Import from local prepare.py (multi-position variant)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import (
    process_text_with_numbers, NUM_TOKEN_ID, NUM_POSITIONS,
    number_to_sme_tokens, sme_tokens_to_number,
    SME_ALL_TOKENS, SME_EXP_MIN, SME_EXP_MAX, SME_MAX_DIGITS,
)
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = 50256


# =============================================================================
# Number sampling (unchanged from parent)
# =============================================================================

def _max_exp_for_range(number_range):
    nr = max(float(number_range), 10.0 ** SME_EXP_MIN)
    hi = int(math.floor(math.log10(nr)))
    return max(SME_EXP_MIN, min(SME_EXP_MAX, hi))


def _canonicalize_float(val, sig_digits=SME_MAX_DIGITS):
    return float(format(float(val), f".{sig_digits}g"))


def _sample_digits_from_bands(bands, max_digits=SME_MAX_DIGITS):
    valid = []
    total_w = 0.0
    for lo, hi, w in bands:
        lo = max(1, int(lo))
        hi = min(int(hi), max_digits)
        if hi < lo or w <= 0:
            continue
        w = float(w)
        valid.append((lo, hi, w))
        total_w += w
    if not valid:
        return random.randint(1, max_digits)
    r = random.random() * total_w
    c = 0.0
    for lo, hi, w in valid:
        c += w
        if r <= c:
            return random.randint(lo, hi)
    lo, hi, _ = valid[-1]
    return random.randint(lo, hi)


def build_sig_digits_sampler(progress, use_curriculum=True, max_digits=SME_MAX_DIGITS):
    if max_digits < 1:
        raise ValueError("max_digits must be >= 1")
    if not use_curriculum:
        return lambda: random.randint(1, max_digits)
    progress = min(1.0, max(0.0, float(progress)))
    short_hi = min(4, max_digits)
    med_lo = min(5, max_digits)
    med_hi = min(8, max_digits)
    long_lo = min(9, max_digits)
    if progress < (1.0 / 3.0):
        bands = [(1, short_hi, 0.9), (med_lo, med_hi, 0.1)]
        return lambda: _sample_digits_from_bands(bands, max_digits=max_digits)
    if progress < (2.0 / 3.0):
        bands = [
            (1, short_hi, 0.35),
            (med_lo, med_hi, 0.50),
            (long_lo, max_digits, 0.15),
        ]
        return lambda: _sample_digits_from_bands(bands, max_digits=max_digits)
    return lambda: random.randint(1, max_digits)


def build_fixed_sig_digits_sampler(min_digits, max_digits):
    min_digits = int(min_digits)
    max_digits = int(max_digits)
    if min_digits < 1 or max_digits < 1:
        raise ValueError("sig_digits bounds must be >= 1")
    if min_digits > max_digits:
        raise ValueError("sig_digits_min must be <= sig_digits_max")
    max_cap = SME_MAX_DIGITS
    min_digits = min(min_digits, max_cap)
    max_digits = min(max_digits, max_cap)
    return lambda: random.randint(min_digits, max_digits)


def sample_number(number_range, allow_negative, allow_float, sig_digits_sampler=None):
    max_exp = _max_exp_for_range(number_range)
    max_abs = max(float(number_range), 10.0 ** SME_EXP_MIN)
    return_as_int = False
    if allow_float:
        mode = random.random()
        if mode < 0.65:
            exp = random.randint(SME_EXP_MIN, max_exp)
            max_mantissa = max_abs / (10.0 ** exp)
            max_mantissa = max(1.0, min(9.999999999999, max_mantissa))
            if max_mantissa <= 1.0 + 1e-12:
                mantissa = 1.0
            else:
                mantissa = random.uniform(1.0, max_mantissa)
            sig_digits = (
                int(sig_digits_sampler()) if sig_digits_sampler is not None
                else random.randint(1, SME_MAX_DIGITS)
            )
            sig_digits = max(1, min(sig_digits, SME_MAX_DIGITS))
            mantissa = _canonicalize_float(mantissa, sig_digits=sig_digits)
            val = mantissa * (10.0 ** exp)
        elif mode < 0.85:
            val = random.uniform(0.0, max_abs)
            sig_digits = (
                int(sig_digits_sampler()) if sig_digits_sampler is not None
                else random.randint(1, SME_MAX_DIGITS)
            )
            sig_digits = max(1, min(sig_digits, SME_MAX_DIGITS))
            val = _canonicalize_float(val, sig_digits=sig_digits)
        else:
            val = random.randint(0, max(1, int(max_abs)))
            return_as_int = True
            if random.random() < 0.25:
                val *= 10.0 ** random.randint(0, max(0, min(4, max_exp)))
                return_as_int = False
        if random.random() < 0.25:
            val = int(round(val))
            return_as_int = True
    else:
        val = int(random.randint(0, max(1, int(max_abs))))
        return_as_int = True
    val = max(0.0, min(val, max_abs))
    if return_as_int:
        val = int(round(val))
    else:
        val = _canonicalize_float(val)
    if allow_negative and random.random() < 0.3:
        val = -val
    if not allow_float:
        val = int(round(val))
    return val


def sample_numbers(n, number_range, allow_negative, allow_float, sig_digits_sampler=None):
    return [
        sample_number(number_range, allow_negative, allow_float,
                      sig_digits_sampler=sig_digits_sampler)
        for _ in range(n)
    ]


def fmt(val):
    if isinstance(val, int):
        return str(val)
    return f"{float(val):.15g}"


# =============================================================================
# Task tokenization (SME-aware, multi-position)
# =============================================================================

def tokenize_task(input_text, output, max_digits=SME_MAX_DIGITS):
    """Tokenize a task with multi-position input and SME output encoding.

    Returns:
        (ids, nums, pos_indices) — three parallel lists ready for block packing
    """
    # Tokenize input (numbers → k × <NUM>)
    ids, nums, pos_indices = process_text_with_numbers(input_text)
    # Remove trailing EOT (we'll add our own at the end)
    ids = ids[:-1]
    nums = nums[:-1]
    pos_indices = pos_indices[:-1]

    if isinstance(output, str):
        out_ids = enc.encode_ordinary(" " + output)
        ids.extend(out_ids)
        nums.extend([0.0] * len(out_ids))
        pos_indices.extend([-1] * len(out_ids))
    elif isinstance(output, list):
        ids.append(220)  # space token before first number
        nums.append(0.0)
        pos_indices.append(-1)
        for i, val in enumerate(output):
            if i > 0:
                ids.append(220)
                nums.append(0.0)
                pos_indices.append(-1)
            sme = number_to_sme_tokens(val, max_digits=max_digits)
            ids.extend(sme)
            nums.extend([0.0] * len(sme))
            pos_indices.extend([-1] * len(sme))
    else:
        ids.append(220)
        nums.append(0.0)
        pos_indices.append(-1)
        sme = number_to_sme_tokens(output, max_digits=max_digits)
        ids.extend(sme)
        nums.extend([0.0] * len(sme))
        pos_indices.extend([-1] * len(sme))

    # Append EOT
    ids.append(EOT_TOKEN)
    nums.append(0.0)
    pos_indices.append(-1)

    return ids, nums, pos_indices


# =============================================================================
# REASONING task generators (unchanged from parent)
# =============================================================================

def gen_cmp(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    if a > b: label = "GREATER"
    elif a < b: label = "LESS"
    else: label = "EQUAL"
    return f"CMP: {fmt(a)} {fmt(b)} →", label

def gen_gt(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    label = "YES" if a > b else "NO"
    return f"GT: {fmt(a)} {fmt(b)} →", label

def gen_is_pos(cfg):
    a = sample_number(cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    label = "YES" if a > 0 else "NO"
    return f"IS_POS: {fmt(a)} →", label

def gen_is_sorted(cfg):
    n = random.randint(cfg['min_len'], min(10, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    if random.random() < 0.5:
        nums = sorted(nums)
    inp = " ".join(fmt(x) for x in nums)
    is_sorted = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
    label = "YES" if is_sorted else "NO"
    return f"IS_SORTED: {inp} →", label

def gen_checksort(cfg):
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    correct_sorted = sorted(nums)
    if random.random() < 0.5:
        proposed = list(correct_sorted)
        label = "YES"
    else:
        proposed = list(correct_sorted)
        if len(proposed) >= 2:
            i, j = random.sample(range(len(proposed)), 2)
            proposed[i], proposed[j] = proposed[j], proposed[i]
        label = "NO" if proposed != correct_sorted else "YES"
    out = " ".join(fmt(x) for x in proposed)
    return f"CHECKSORT: {inp} → {out} CHECK →", label

def gen_checkadd(cfg):
    mod = cfg.get('max_output_digits', SME_MAX_DIGITS)
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    correct = _canonicalize_float(a + b, sig_digits=mod)
    if isinstance(a, int) and isinstance(b, int):
        correct = int(correct)
    if random.random() < 0.5:
        c = correct
        label = "YES"
    else:
        noise = sample_number(max(abs(correct) * 0.5, 10), True, cfg['flt'], cfg.get('sig_digits_sampler'))
        c = _canonicalize_float(correct + noise, sig_digits=mod) if cfg['flt'] else int(correct + noise)
        label = "NO" if c != correct else "YES"
    return f"CHECKADD: {fmt(a)} + {fmt(b)} = {fmt(c)} →", label

def gen_sum_cmp(cfg):
    a, b, c, d = sample_numbers(4, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    sum1, sum2 = a + b, c + d
    if sum1 > sum2: label = "FIRST"
    elif sum1 < sum2: label = "SECOND"
    else: label = "EQUAL"
    return f"SUM_CMP: {fmt(a)} + {fmt(b)} vs {fmt(c)} + {fmt(d)} →", label

# =============================================================================
# REGRESSION task generators (unchanged from parent)
# =============================================================================

def gen_sort(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    return f"SORT: {inp} →", sorted(nums)

def gen_add(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a + b, sig_digits=cfg.get('max_output_digits', SME_MAX_DIGITS))
    if isinstance(a, int) and isinstance(b, int): result = int(result)
    return f"ADD: {fmt(a)} + {fmt(b)} →", result

def gen_sub(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a - b, sig_digits=cfg.get('max_output_digits', SME_MAX_DIGITS))
    if isinstance(a, int) and isinstance(b, int): result = int(result)
    return f"SUB: {fmt(a)} - {fmt(b)} →", result

def gen_min(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    return f"MIN: {inp} →", min(nums)

def gen_max(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    return f"MAX: {inp} →", max(nums)

def gen_sum(cfg):
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(sum(nums), sig_digits=cfg.get('max_output_digits', SME_MAX_DIGITS))
    if all(isinstance(x, int) for x in nums): result = int(result)
    inp = " ".join(fmt(x) for x in nums)
    return f"SUM: {inp} →", result

def gen_count(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    return f"COUNT: {inp} →", n

# =============================================================================
# Task registry
# =============================================================================

REASONING_TASK_GENERATORS = [
    gen_cmp, gen_gt, gen_is_pos, gen_is_sorted,
    gen_checksort, gen_checkadd, gen_sum_cmp,
]
NUMERIC_TASK_GENERATORS = [
    gen_sort, gen_add, gen_sub, gen_min, gen_max, gen_sum, gen_count,
]

def build_task_generators(reasoning_weight, numeric_weight):
    out = []
    for g in REASONING_TASK_GENERATORS: out.append((g, int(reasoning_weight)))
    for g in NUMERIC_TASK_GENERATORS:  out.append((g, int(numeric_weight)))
    return out


# =============================================================================
# Block packing (extended for pos_indices)
# =============================================================================

def pack_into_blocks(examples, block_size):
    """Pack tokenized examples into fixed-size blocks.

    Returns:
        all_ids, all_nums, all_pos, n_blocks, examples_packed
    """
    all_ids = []
    all_nums = []
    all_pos = []

    block_ids = []
    block_nums = []
    block_pos = []
    n_blocks = 0
    examples_packed = 0

    for ids, nums, pos_indices in examples:
        example_len = len(ids)
        if example_len > block_size:
            continue
        if len(block_ids) + example_len > block_size:
            pad_len = block_size - len(block_ids)
            block_ids.extend([EOT_TOKEN] * pad_len)
            block_nums.extend([0.0] * pad_len)
            block_pos.extend([-1] * pad_len)
            all_ids.extend(block_ids)
            all_nums.extend(block_nums)
            all_pos.extend(block_pos)
            n_blocks += 1
            block_ids = []
            block_nums = []
            block_pos = []
        block_ids.extend(ids)
        block_nums.extend(nums)
        block_pos.extend(pos_indices)
        examples_packed += 1

    if block_ids:
        pad_len = block_size - len(block_ids)
        block_ids.extend([EOT_TOKEN] * pad_len)
        block_nums.extend([0.0] * pad_len)
        block_pos.extend([-1] * pad_len)
        all_ids.extend(block_ids)
        all_nums.extend(block_nums)
        all_pos.extend(block_pos)
        n_blocks += 1

    return all_ids, all_nums, all_pos, n_blocks, examples_packed


def save_split(ids, nums, pos_indices, split, out_dir):
    """Save tokenized data to triple .bin files (tokens, nums, pos)."""
    os.makedirs(out_dir, exist_ok=True)
    arr_len = len(ids)

    tok_path = os.path.join(out_dir, f'{split}.bin')
    num_path = os.path.join(out_dir, f'{split}_nums.bin')
    pos_path = os.path.join(out_dir, f'{split}_pos.bin')

    tok_arr = np.memmap(tok_path, dtype=np.uint16, mode='w+', shape=(arr_len,))
    num_arr = np.memmap(num_path, dtype=np.float32, mode='w+', shape=(arr_len,))
    pos_arr = np.memmap(pos_path, dtype=np.int8, mode='w+', shape=(arr_len,))

    tok_arr[:] = np.array(ids, dtype=np.uint16)
    num_arr[:] = np.array(nums, dtype=np.float32)
    pos_arr[:] = np.array(pos_indices, dtype=np.int8)

    tok_arr.flush()
    num_arr.flush()
    pos_arr.flush()

    num_count = int(np.sum(np.array(ids, dtype=np.uint32) == NUM_TOKEN_ID))
    sme_count = int(np.sum(np.isin(np.array(ids, dtype=np.uint32), list(SME_ALL_TOKENS))))
    unique_nums = num_count // NUM_POSITIONS
    print(f"  {split}: {arr_len:,} tokens, {unique_nums:,} numbers "
          f"({num_count:,} NUM tokens, {NUM_POSITIONS} pos each), "
          f"{sme_count:,} SME tokens ({sme_count / arr_len * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-position numerical task data")
    parser.add_argument("--n-train", type=int, default=5000000)
    parser.add_argument("--n-val", type=int, default=50000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--number-range", type=float, default=1000000000.0)
    parser.add_argument("--allow-negative", action="store_true", default=True)
    parser.add_argument("--no-negative", action="store_true")
    parser.add_argument("--allow-float", action="store_true", default=True)
    parser.add_argument("--integers-only", action="store_true")
    parser.add_argument("--reasoning-weight", type=int, default=1)
    parser.add_argument("--numeric-weight", type=int, default=2)
    parser.add_argument("--digit-curriculum", action="store_true", default=True)
    parser.add_argument("--no-digit-curriculum", dest="digit_curriculum", action="store_false")
    parser.add_argument("--sig-digits-min", type=int, default=None)
    parser.add_argument("--sig-digits-max", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_negative: args.allow_negative = False
    if args.integers_only: args.allow_float = False
    if args.reasoning_weight <= 0 or args.numeric_weight <= 0:
        raise ValueError("reasoning-weight and numeric-weight must be > 0")
    if args.sig_digits_min is not None or args.sig_digits_max is not None:
        smin = 1 if args.sig_digits_min is None else int(args.sig_digits_min)
        smax = SME_MAX_DIGITS if args.sig_digits_max is None else int(args.sig_digits_max)
        if smin < 1 or smax < 1 or smin > smax:
            raise ValueError("invalid sig-digits range; require 1 <= min <= max")
        args.sig_digits_min = min(smin, SME_MAX_DIGITS)
        args.sig_digits_max = min(smax, SME_MAX_DIGITS)
    else:
        args.sig_digits_min = None
        args.sig_digits_max = None
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'numtasks_sme_multipos')

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_generators = build_task_generators(args.reasoning_weight, args.numeric_weight)
    generators = [g for g, _ in task_generators]
    weights = [w for _, w in task_generators]

    max_output_digits = args.sig_digits_max if args.sig_digits_max is not None else SME_MAX_DIGITS

    cfg = {
        'range': args.number_range,
        'neg': args.allow_negative,
        'flt': args.allow_float,
        'min_len': args.min_len,
        'max_len': args.max_len,
        'sig_digits_sampler': (lambda: random.randint(1, SME_MAX_DIGITS)),
        'max_output_digits': max_output_digits,
    }

    task_names = [g.__name__[4:].upper() for g in generators]
    reasoning_names = [g.__name__[4:].upper() for g in REASONING_TASK_GENERATORS]
    numeric_names = [g.__name__[4:].upper() for g in NUMERIC_TASK_GENERATORS]

    print("=" * 60)
    print("NUMERICAL TASK DATA GENERATOR (MULTI-POSITION SME)")
    print("=" * 60)
    print(f"  NUM_POSITIONS:   {NUM_POSITIONS}")
    print(f"  Tasks:           {', '.join(task_names)}")
    print(f"  Reasoning ({args.reasoning_weight}x): {', '.join(reasoning_names)}")
    print(f"  SME output ({args.numeric_weight}x): {', '.join(numeric_names)}")
    print(f"  Train examples:  {args.n_train:,}")
    print(f"  Val examples:    {args.n_val:,}")
    print(f"  Block size:      {args.block_size}")
    print(f"  Sequence length: {args.min_len}-{args.max_len}")
    print(f"  Number range:    [-{args.number_range:g}, {args.number_range:g}]")
    if args.sig_digits_min is not None:
        print(f"  Digit range:     fixed {args.sig_digits_min}-{args.sig_digits_max}")
    else:
        print(f"  Digit curriculum:{'on' if args.digit_curriculum else 'off'}")
    print(f"  Output dir:      {args.out_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    for split, n_examples in [('train', args.n_train), ('val', args.n_val)]:
        print(f"Generating {split} ({n_examples:,} examples)...")

        examples_raw = []
        task_counts = {name: 0 for name in task_names}
        fixed_sig_sampler = None
        if args.sig_digits_min is not None:
            fixed_sig_sampler = build_fixed_sig_digits_sampler(
                args.sig_digits_min, args.sig_digits_max)
        for i in tqdm(range(n_examples), desc=f"generating {split}"):
            if fixed_sig_sampler is not None:
                cfg['sig_digits_sampler'] = fixed_sig_sampler
            else:
                progress = (i / max(1, n_examples - 1)) if split == 'train' else 1.0
                cfg['sig_digits_sampler'] = build_sig_digits_sampler(
                    progress=progress, use_curriculum=args.digit_curriculum,
                    max_digits=SME_MAX_DIGITS)
            gen = random.choices(generators, weights=weights, k=1)[0]
            input_text, output = gen(cfg)
            examples_raw.append((input_text, output))
            task_counts[gen.__name__[4:].upper()] += 1

        if split == 'train':
            print("\nSample examples:")
            seen = set()
            for input_text, output in examples_raw:
                task_type = input_text.split(":")[0]
                if task_type not in seen:
                    if isinstance(output, str):
                        print(f"  {input_text} {output}")
                    elif isinstance(output, list):
                        out_str = " ".join(fmt(x) for x in output)
                        print(f"  {input_text} [{out_str}]  (SME)")
                    else:
                        print(f"  {input_text} {fmt(output)}  (SME)")
                    seen.add(task_type)
                if len(seen) == len(generators):
                    break
            print()

        # Tokenize with multi-position + SME encoding
        print(f"Tokenizing {split} (multi-pos k={NUM_POSITIONS}, SME max_digits={max_output_digits})...")
        tokenized = []
        for input_text, output in tqdm(examples_raw, desc=f"tokenizing {split}"):
            ids, nums, pos_indices = tokenize_task(input_text, output, max_digits=max_output_digits)
            tokenized.append((ids, nums, pos_indices))

        lengths = [len(ids) for ids, _, _ in tokenized]
        print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}")
        skipped = sum(1 for l in lengths if l > args.block_size)
        if skipped:
            print(f"  WARNING: {skipped} examples exceed block_size={args.block_size}, "
                  f"will be skipped")

        print(f"Packing into blocks of {args.block_size}...")
        all_ids, all_nums, all_pos, n_blocks, n_packed = pack_into_blocks(
            tokenized, args.block_size)
        print(f"  {n_packed:,} examples packed into {n_blocks:,} blocks")
        print(f"  Examples per block: {n_packed / n_blocks:.1f} avg")
        print(f"  Task distribution: {task_counts}")

        print(f"Saving {split}...")
        save_split(all_ids, all_nums, all_pos, split, args.out_dir)
        print()

    # Save meta
    meta = {
        'vocab_size': 50304,
        'num_token_id': NUM_TOKEN_ID,
        'num_positions': NUM_POSITIONS,
        'sme_tokens': True,
        'dataset': 'numtasks_sme_multipos',
        'tasks': task_names,
        'n_train': args.n_train,
        'n_val': args.n_val,
        'block_size': args.block_size,
        'number_range': args.number_range,
        'sme_exp_min': SME_EXP_MIN,
        'sme_exp_max': SME_EXP_MAX,
        'sme_max_digits': SME_MAX_DIGITS,
        'task_weights': {
            'reasoning': args.reasoning_weight,
            'numeric': args.numeric_weight,
        },
        'digit_curriculum': bool(args.digit_curriculum),
        'sig_digits_min': args.sig_digits_min,
        'sig_digits_max': args.sig_digits_max,
    }
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print("Tokenization example:")
    input_text, output = examples_raw[0]
    ids, nums, pos_indices = tokenize_task(input_text, output)
    print(f"  Input: {input_text}")
    print(f"  Output: {output}")
    print(f"  Tokens ({len(ids)}): {ids[:40]}...")
    print(f"  Pos idx: {pos_indices[:40]}...")

    print(f"\nDone. To train:")
    print(f"  python fe_multipos/train.py data_dir={args.out_dir}")


if __name__ == '__main__':
    main()
