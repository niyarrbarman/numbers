"""
Generate synthetic numerical task data for number-aware GPT-2 training.

REASONING tasks (text output depends on number values — trains via text_loss):
  CMP:        42 17 → GREATER  (or LESS, EQUAL)
  GT:         42 17 → YES  (is first > second?)
  IS_POS:     -3.5 → NO  (is number positive?)
  IS_SORTED:  3 5 82 → YES  (is sequence sorted ascending?)
  CHECKSORT:  5 3 82 → 3 5 82 CHECK → YES  (verify sort correctness)
  CHECKADD:   3 + 5 = 8 → YES  (verify addition)
  SUM_CMP:    3 + 5 vs 2 + 7 → FIRST  (which pair sums higher? or SECOND/EQUAL)

REGRESSION tasks (number output — now encoded as SME tokens):
  SORT: 5 82 -3 → <S->E0D3END <S+>E0D5END <S+>E1D8D2END
  ADD:  3 + 5 → <S+>E0D8END
  SUB:  10 - 3 → <S+>E0D7END
  MIN:  5 82 3 → <S+>E0D3END
  MAX:  5 82 3 → <S+>E1D8D2END
  SUM:  10 20 30 → <S+>E1D6END
  COUNT also kept

Input numbers use <NUM> token with adapter embeddings.
Output numbers use SME (Sign-Mantissa-Exponent) text tokens.
All loss is now text cross-entropy — no separate num_loss.

Examples are packed into fixed-size blocks separated by EOT tokens.
Output is saved in the dual .bin format expected by fe/train.py.

Usage:
    python generate_data.py --out-dir /path/to/output
    python generate_data.py --number-range 1000000000 --n-train 1000000
"""

import os
import sys
import argparse
import pickle
import random
import math

import numpy as np
from tqdm import tqdm

# Import the tokenizer from fe/prepare.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fe'))
from fe.prepare import (
    process_text_with_numbers, NUM_TOKEN_ID,
    number_to_sme_tokens, sme_tokens_to_number,
    SME_ALL_TOKENS, SME_EXP_MIN, SME_EXP_MAX, SME_MAX_DIGITS,
)
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = 50256


# =============================================================================
# Number sampling
# =============================================================================

def _max_exp_for_range(number_range):
    """Highest exponent that can fit in the configured absolute range."""
    nr = max(float(number_range), 10.0 ** SME_EXP_MIN)
    hi = int(math.floor(math.log10(nr)))
    return max(SME_EXP_MIN, min(SME_EXP_MAX, hi))


def _canonicalize_float(val, sig_digits=SME_MAX_DIGITS):
    """Round to a stable number of significant digits for reproducible text."""
    return float(format(float(val), f".{sig_digits}g"))


def _sample_digits_from_bands(bands, max_digits=SME_MAX_DIGITS):
    """Sample a digit count from weighted [lo, hi] bands."""
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
    """Create a per-example significant-digit sampler.

    Curriculum phases:
      1) mostly short mantissas (1-4 digits)
      2) introduce medium (5-8)
      3) full 1-max distribution
    """
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
        # Phase 1: mostly short.
        bands = [(1, short_hi, 0.9), (med_lo, med_hi, 0.1)]
        return lambda: _sample_digits_from_bands(bands, max_digits=max_digits)
    if progress < (2.0 / 3.0):
        # Phase 2: medium emphasized, with some short and some long.
        bands = [
            (1, short_hi, 0.35),
            (med_lo, med_hi, 0.50),
            (long_lo, max_digits, 0.15),
        ]
        return lambda: _sample_digits_from_bands(bands, max_digits=max_digits)
    # Phase 3: full range.
    return lambda: random.randint(1, max_digits)


def build_fixed_sig_digits_sampler(min_digits, max_digits):
    """Create a fixed-range significant-digit sampler."""
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
    """Sample one number with broad exponent coverage and mixed precision."""
    max_exp = _max_exp_for_range(number_range)
    max_abs = max(float(number_range), 10.0 ** SME_EXP_MIN)
    return_as_int = False

    if allow_float:
        mode = random.random()

        if mode < 0.65:
            # Main mode: sample exponent explicitly so E-9..E9 are all seen.
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
            # Uniform background coverage.
            val = random.uniform(0.0, max_abs)
            sig_digits = (
                int(sig_digits_sampler()) if sig_digits_sampler is not None
                else random.randint(1, SME_MAX_DIGITS)
            )
            sig_digits = max(1, min(sig_digits, SME_MAX_DIGITS))
            val = _canonicalize_float(val, sig_digits=sig_digits)
        else:
            # Integer/round-number bias for short mantissas.
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
    """Sample a list of n random numbers."""
    return [
        sample_number(
            number_range,
            allow_negative,
            allow_float,
            sig_digits_sampler=sig_digits_sampler,
        )
        for _ in range(n)
    ]


def fmt(val):
    """Format a number for text representation."""
    if isinstance(val, int):
        return str(val)
    return f"{float(val):.15g}"


# =============================================================================
# Task tokenization (SME-aware)
# =============================================================================

def tokenize_task(input_text, output, max_digits=SME_MAX_DIGITS):
    """Tokenize a task with SME encoding for output numbers.

    Args:
        input_text: str — input part (numbers become <NUM> with embeddings)
        output: str | number | list[number] — output part
        max_digits: int — max mantissa digits for SME encoding of output numbers

    Returns:
        (ids, nums) — parallel lists ready for block packing
    """
    # Tokenize input (numbers → <NUM>)
    ids, nums = process_text_with_numbers(input_text)
    # Remove trailing EOT (we'll add our own at the end)
    ids = ids[:-1]
    nums = nums[:-1]

    if isinstance(output, str):
        # Text output (YES, NO, GREATER, etc.)
        out_ids = enc.encode_ordinary(" " + output)
        ids.extend(out_ids)
        nums.extend([0.0] * len(out_ids))
    elif isinstance(output, list):
        # Multiple number outputs → SME tokens with spaces between
        ids.append(220)  # space token before first number
        nums.append(0.0)
        for i, val in enumerate(output):
            if i > 0:
                ids.append(220)  # space between numbers
                nums.append(0.0)
            sme = number_to_sme_tokens(val, max_digits=max_digits)
            ids.extend(sme)
            nums.extend([0.0] * len(sme))
    else:
        # Single number output → SME tokens
        ids.append(220)  # space token before number
        nums.append(0.0)
        sme = number_to_sme_tokens(output, max_digits=max_digits)
        ids.extend(sme)
        nums.extend([0.0] * len(sme))

    # Append EOT
    ids.append(EOT_TOKEN)
    nums.append(0.0)

    return ids, nums


# =============================================================================
# REASONING task generators — text output depends on number values
# Each returns (input_text, output_str)
# =============================================================================

def gen_cmp(cfg):
    """Compare two numbers → GREATER/LESS/EQUAL"""
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    if a > b:
        label = "GREATER"
    elif a < b:
        label = "LESS"
    else:
        label = "EQUAL"
    return f"CMP: {fmt(a)} {fmt(b)} →", label


def gen_gt(cfg):
    """Is first > second? → YES/NO"""
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    label = "YES" if a > b else "NO"
    return f"GT: {fmt(a)} {fmt(b)} →", label


def gen_is_pos(cfg):
    """Is number positive? → YES/NO"""
    a = sample_number(cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    label = "YES" if a > 0 else "NO"
    return f"IS_POS: {fmt(a)} →", label


def gen_is_sorted(cfg):
    """Is sequence sorted ascending? → YES/NO"""
    n = random.randint(cfg['min_len'], min(10, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    # 50% of the time, actually sort it so we get balanced YES/NO
    if random.random() < 0.5:
        nums = sorted(nums)
    inp = " ".join(fmt(x) for x in nums)
    is_sorted = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
    label = "YES" if is_sorted else "NO"
    return f"IS_SORTED: {inp} →", label


def gen_checksort(cfg):
    """Verify if a proposed sort is correct → YES/NO"""
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    correct_sorted = sorted(nums)
    # 50% correct, 50% wrong (swap two random elements)
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
    # All numbers here are INPUT (embedded via <NUM>), output is just text
    return f"CHECKSORT: {inp} → {out} CHECK →", label


def gen_checkadd(cfg):
    """Verify if a + b = c is correct → YES/NO"""
    mod = cfg.get('max_output_digits', SME_MAX_DIGITS)
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    correct = _canonicalize_float(a + b, sig_digits=mod)
    if isinstance(a, int) and isinstance(b, int):
        correct = int(correct)
    # 50% correct, 50% wrong (add noise to result)
    if random.random() < 0.5:
        c = correct
        label = "YES"
    else:
        # Perturb by a meaningful amount
        noise = sample_number(
            max(abs(correct) * 0.5, 10),
            True,
            cfg['flt'],
            cfg.get('sig_digits_sampler'),
        )
        c = _canonicalize_float(correct + noise, sig_digits=mod) if cfg['flt'] else int(correct + noise)
        label = "NO" if c != correct else "YES"
    # All numbers (a, b, c) are INPUT — output is just YES/NO
    return f"CHECKADD: {fmt(a)} + {fmt(b)} = {fmt(c)} →", label


def gen_sum_cmp(cfg):
    """Which pair sums to more? → FIRST/SECOND/EQUAL"""
    a, b, c, d = sample_numbers(4, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    sum1 = a + b
    sum2 = c + d
    if sum1 > sum2:
        label = "FIRST"
    elif sum1 < sum2:
        label = "SECOND"
    else:
        label = "EQUAL"
    return f"SUM_CMP: {fmt(a)} + {fmt(b)} vs {fmt(c)} + {fmt(d)} →", label


# =============================================================================
# REGRESSION task generators — number output (encoded as SME tokens)
# Each returns (input_text, number_or_list)
# =============================================================================

def gen_sort(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    return f"SORT: {inp} →", sorted(nums)


def gen_add(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a + b, sig_digits=cfg.get('max_output_digits', SME_MAX_DIGITS))
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"ADD: {fmt(a)} + {fmt(b)} →", result


def gen_sub(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a - b, sig_digits=cfg.get('max_output_digits', SME_MAX_DIGITS))
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
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
    if all(isinstance(x, int) for x in nums):
        result = int(result)
    inp = " ".join(fmt(x) for x in nums)
    return f"SUM: {inp} →", result


def gen_count(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    inp = " ".join(fmt(x) for x in nums)
    return f"COUNT: {inp} →", n


# =============================================================================
# Task registry with weights
# =============================================================================

REASONING_TASK_GENERATORS = [
    gen_cmp,
    gen_gt,
    gen_is_pos,
    gen_is_sorted,
    gen_checksort,
    gen_checkadd,
    gen_sum_cmp,
]

NUMERIC_TASK_GENERATORS = [
    gen_sort,
    gen_add,
    gen_sub,
    gen_min,
    gen_max,
    gen_sum,
    gen_count,
]


def build_task_generators(reasoning_weight, numeric_weight):
    out = []
    for g in REASONING_TASK_GENERATORS:
        out.append((g, int(reasoning_weight)))
    for g in NUMERIC_TASK_GENERATORS:
        out.append((g, int(numeric_weight)))
    return out


# =============================================================================
# Block packing
# =============================================================================

def pack_into_blocks(examples, block_size):
    """Pack tokenized examples into fixed-size blocks.

    Multiple examples are packed per block, separated by EOT tokens.
    Remaining space is filled with EOT (target=-1 handled by ignore_index).

    Returns:
        all_ids:  list[int]   — token IDs, length is a multiple of block_size
        all_nums: list[float] — parallel number values
        n_blocks: int         — number of blocks created
    """
    all_ids = []
    all_nums = []

    # Current block being filled
    block_ids = []
    block_nums = []
    n_blocks = 0
    examples_packed = 0

    for ids, nums in examples:
        example_len = len(ids)  # includes EOT from tokenize_task

        # If this single example is too long for a block, skip it
        if example_len > block_size:
            continue

        # If adding this example would overflow the block, flush current block
        if len(block_ids) + example_len > block_size:
            # Pad remainder with EOT
            pad_len = block_size - len(block_ids)
            block_ids.extend([EOT_TOKEN] * pad_len)
            block_nums.extend([0.0] * pad_len)
            all_ids.extend(block_ids)
            all_nums.extend(block_nums)
            n_blocks += 1
            block_ids = []
            block_nums = []

        # Add example to current block
        block_ids.extend(ids)
        block_nums.extend(nums)
        examples_packed += 1

    # Flush last partial block
    if block_ids:
        pad_len = block_size - len(block_ids)
        block_ids.extend([EOT_TOKEN] * pad_len)
        block_nums.extend([0.0] * pad_len)
        all_ids.extend(block_ids)
        all_nums.extend(block_nums)
        n_blocks += 1

    return all_ids, all_nums, n_blocks, examples_packed


def save_split(ids, nums, split, out_dir):
    """Save tokenized data to dual .bin files."""
    os.makedirs(out_dir, exist_ok=True)
    arr_len = len(ids)

    tok_path = os.path.join(out_dir, f'{split}.bin')
    num_path = os.path.join(out_dir, f'{split}_nums.bin')

    tok_arr = np.memmap(tok_path, dtype=np.uint16, mode='w+', shape=(arr_len,))
    num_arr = np.memmap(num_path, dtype=np.float32, mode='w+', shape=(arr_len,))

    tok_arr[:] = np.array(ids, dtype=np.uint16)
    num_arr[:] = np.array(nums, dtype=np.float32)

    tok_arr.flush()
    num_arr.flush()

    num_count = int(np.sum(np.array(ids, dtype=np.uint32) == NUM_TOKEN_ID))
    sme_count = int(np.sum(np.isin(np.array(ids, dtype=np.uint32), list(SME_ALL_TOKENS))))
    print(f"  {split}: {arr_len:,} tokens, {num_count:,} <NUM> embeddings, "
          f"{sme_count:,} SME tokens ({sme_count / arr_len * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-task numerical data for number-aware GPT-2")
    parser.add_argument("--n-train", type=int, default=5000000,
                        help="Number of training examples (default: 5000000)")
    parser.add_argument("--n-val", type=int, default=50000,
                        help="Number of validation examples (default: 50000)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Block size for packing (must match train.py block_size)")
    parser.add_argument("--min-len", type=int, default=2,
                        help="Min sequence length for list tasks (default: 2)")
    parser.add_argument("--max-len", type=int, default=10,
                        help="Max sequence length for list tasks (default: 10)")
    parser.add_argument("--number-range", type=float, default=1000000000.0,
                        help="Max absolute value of numbers (default: 1e9)")
    parser.add_argument("--allow-negative", action="store_true", default=True)
    parser.add_argument("--no-negative", action="store_true")
    parser.add_argument("--allow-float", action="store_true", default=True)
    parser.add_argument("--integers-only", action="store_true")
    parser.add_argument("--reasoning-weight", type=int, default=1,
                        help="Sampling weight for reasoning tasks (default: 1)")
    parser.add_argument("--numeric-weight", type=int, default=2,
                        help="Sampling weight for numeric-output tasks (default: 2)")
    parser.add_argument("--digit-curriculum", action="store_true", default=True,
                        help="Enable mantissa digit curriculum (default: on)")
    parser.add_argument("--no-digit-curriculum", dest="digit_curriculum", action="store_false",
                        help="Disable mantissa digit curriculum")
    parser.add_argument("--sig-digits-min", type=int, default=None,
                        help="Force minimum significant digits for sampled floats")
    parser.add_argument("--sig-digits-max", type=int, default=None,
                        help="Force maximum significant digits for sampled floats")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_negative:
        args.allow_negative = False
    if args.integers_only:
        args.allow_float = False
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
                                    'fe', 'data', 'numtasks_sme_vardig_e9')

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_generators = build_task_generators(
        reasoning_weight=args.reasoning_weight,
        numeric_weight=args.numeric_weight,
    )
    generators = [g for g, _ in task_generators]
    weights = [w for _, w in task_generators]

    # max_output_digits: cap result precision to match input constraint
    if args.sig_digits_max is not None:
        max_output_digits = args.sig_digits_max
    else:
        max_output_digits = SME_MAX_DIGITS

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
    print("NUMERICAL TASK DATA GENERATOR (SME OUTPUT)")
    print("=" * 60)
    print(f"  Tasks:           {', '.join(task_names)}")
    print(f"  Reasoning ({args.reasoning_weight}x): {', '.join(reasoning_names)}")
    print(f"  SME output ({args.numeric_weight}x): {', '.join(numeric_names)}")
    print(f"  Train examples:  {args.n_train:,}")
    print(f"  Val examples:    {args.n_val:,}")
    print(f"  Block size:      {args.block_size}")
    print(f"  Sequence length: {args.min_len}-{args.max_len}")
    print(f"  Number range:    [-{args.number_range:g}, {args.number_range:g}]")
    print(f"  SME exponents:   E{SME_EXP_MIN}..E{SME_EXP_MAX}")
    print(f"  Max digits:      {SME_MAX_DIGITS} (+ END)")
    if args.sig_digits_min is not None:
        print(f"  Digit range:     fixed {args.sig_digits_min}-{args.sig_digits_max}")
        print(f"  Output digits:   capped at {max_output_digits}")
        print("  Digit curriculum:off (fixed range overrides curriculum)")
    else:
        print(f"  Digit curriculum:{'on' if args.digit_curriculum else 'off'} "
              f"(phase1 1-4, phase2 5-8, phase3 1-{SME_MAX_DIGITS})")
    print(f"  Allow float:     {args.allow_float}")
    print(f"  Output dir:      {args.out_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    # --- Generate and tokenize examples ---
    for split, n_examples in [('train', args.n_train), ('val', args.n_val)]:
        print(f"Generating {split} ({n_examples:,} examples)...")

        # Generate structured examples
        examples_raw = []
        task_counts = {name: 0 for name in task_names}
        fixed_sig_sampler = None
        if args.sig_digits_min is not None:
            fixed_sig_sampler = build_fixed_sig_digits_sampler(
                args.sig_digits_min,
                args.sig_digits_max,
            )
        for i in tqdm(range(n_examples), desc=f"generating {split}"):
            if fixed_sig_sampler is not None:
                cfg['sig_digits_sampler'] = fixed_sig_sampler
            else:
                progress = (i / max(1, n_examples - 1)) if split == 'train' else 1.0
                cfg['sig_digits_sampler'] = build_sig_digits_sampler(
                    progress=progress,
                    use_curriculum=args.digit_curriculum,
                    max_digits=SME_MAX_DIGITS,
                )
            gen = random.choices(generators, weights=weights, k=1)[0]
            input_text, output = gen(cfg)
            examples_raw.append((input_text, output))
            task_counts[gen.__name__[4:].upper()] += 1

        # Show samples
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

            # Show SME encoding example
            print("\nSME encoding example:")
            for input_text, output in examples_raw:
                if not isinstance(output, str):
                    val = output if not isinstance(output, list) else output[0]
                    sme = number_to_sme_tokens(val)
                    decoded = sme_tokens_to_number(sme)
                    print(f"  Value: {val} → SME tokens: {sme} → decoded: {decoded}")
                    break
            print()

        # Tokenize with SME encoding
        print(f"Tokenizing {split} (with SME for output numbers, max_digits={max_output_digits})...")
        tokenized = []
        for input_text, output in tqdm(examples_raw, desc=f"tokenizing {split}"):
            ids, nums = tokenize_task(input_text, output, max_digits=max_output_digits)
            tokenized.append((ids, nums))

        # Report token lengths
        lengths = [len(ids) for ids, _ in tokenized]
        print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}")
        skipped = sum(1 for l in lengths if l > args.block_size)
        if skipped:
            print(f"  WARNING: {skipped} examples exceed block_size={args.block_size}, "
                  f"will be skipped")

        # Pack into blocks
        print(f"Packing into blocks of {args.block_size}...")
        all_ids, all_nums, n_blocks, n_packed = pack_into_blocks(
            tokenized, args.block_size)
        print(f"  {n_packed:,} examples packed into {n_blocks:,} blocks")
        print(f"  Examples per block: {n_packed / n_blocks:.1f} avg")

        # Task distribution
        print(f"  Task distribution: {task_counts}")

        # Save
        print(f"Saving {split}...")
        save_split(all_ids, all_nums, split, args.out_dir)
        print()

    # Save meta
    meta = {
        'vocab_size': 50304,
        'num_token_id': NUM_TOKEN_ID,
        'sme_tokens': True,
        'dataset': 'numtasks_sme_vardig_e9',
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

    # Show tokenization of one example
    print("Tokenization example:")
    input_text, output = examples_raw[0]
    ids, nums = tokenize_task(input_text, output)
    print(f"  Input: {input_text}")
    print(f"  Output: {output}")
    print(f"  Tokens: {ids}")
    print(f"  Nums:   {nums}")
    print(f"  Length:  {len(ids)} tokens")

    print(f"\nDone. To train:")
    print(f"  python fe/train.py dataset=numtasks_sme_vardig_e9 data_dir={args.out_dir}")


if __name__ == '__main__':
    main()
