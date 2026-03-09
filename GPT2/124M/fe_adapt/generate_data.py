"""
Generate synthetic numerical task data for Baby Luciole + NumberEncoder adapter.

Uses luciole_50k SentencePiece tokenizer (NOT GPT-2 tiktoken).
  - INPUT numbers are replaced with <NUM> tokens
  - OUTPUT numbers remain as plain text tokens

Produces two binary files per split:
  - {split}.bin      : uint16 token IDs (with NUM_TOKEN_ID at input number positions)
  - {split}_nums.bin : float32 values (float at NUM positions, 0.0 elsewhere)
"""

import os
import sys
import argparse
import pickle
import random
import math

import numpy as np
from tqdm import tqdm

# Import from local prepare.py (luciole_50k tokenizer)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import process_text_with_numbers, NUM_TOKEN_ID, EOT_TOKEN_ID, _get_tokenizer

# Number range constants
MAX_SIG_DIGITS = 15
EXP_MIN = -9
EXP_MAX = 9


# =============================================================================
# Number sampling
# =============================================================================

def _max_exp_for_range(number_range):
    nr = max(float(number_range), 10.0 ** EXP_MIN)
    hi = int(math.floor(math.log10(nr)))
    return max(EXP_MIN, min(EXP_MAX, hi))


def _canonicalize_float(val, sig_digits=MAX_SIG_DIGITS):
    return float(format(float(val), f".{sig_digits}g"))


def _sample_digits_from_bands(bands, max_digits=MAX_SIG_DIGITS):
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


def build_sig_digits_sampler(progress, use_curriculum=True, max_digits=MAX_SIG_DIGITS):
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
    max_cap = MAX_SIG_DIGITS
    min_digits = min(min_digits, max_cap)
    max_digits = min(max_digits, max_cap)
    return lambda: random.randint(min_digits, max_digits)


def sample_number(number_range, allow_negative, allow_float, sig_digits_sampler=None):
    max_exp = _max_exp_for_range(number_range)
    max_abs = max(float(number_range), 10.0 ** EXP_MIN)
    return_as_int = False
    if allow_float:
        mode = random.random()
        if mode < 0.65:
            exp = random.randint(EXP_MIN, max_exp)
            max_mantissa = max_abs / (10.0 ** exp)
            max_mantissa = max(1.0, min(9.999999999999, max_mantissa))
            if max_mantissa <= 1.0 + 1e-12:
                mantissa = 1.0
            else:
                mantissa = random.uniform(1.0, max_mantissa)
            sig_digits = (
                int(sig_digits_sampler()) if sig_digits_sampler is not None
                else random.randint(1, MAX_SIG_DIGITS)
            )
            sig_digits = max(1, min(sig_digits, MAX_SIG_DIGITS))
            mantissa = _canonicalize_float(mantissa, sig_digits=sig_digits)
            val = mantissa * (10.0 ** exp)
        elif mode < 0.85:
            val = random.uniform(0.0, max_abs)
            sig_digits = (
                int(sig_digits_sampler()) if sig_digits_sampler is not None
                else random.randint(1, MAX_SIG_DIGITS)
            )
            sig_digits = max(1, min(sig_digits, MAX_SIG_DIGITS))
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
# Task tokenization (NUM input + plain text output)
# =============================================================================

def tokenize_task(input_text, output, max_digits=MAX_SIG_DIGITS):
    """Tokenize a task with NUM-encoded input and plain-text output."""
    tokenizer = _get_tokenizer()

    # Tokenize input (numbers -> single <NUM> token each)
    ids, nums = process_text_with_numbers(input_text)
    # Remove trailing EOT (we'll add our own at the end)
    ids = ids[:-1]
    nums = nums[:-1]

    # Tokenize output as plain text
    if isinstance(output, str):
        output_text = output
    elif isinstance(output, list):
        output_text = " ".join(fmt(x) for x in output)
    else:
        output_text = fmt(output)

    out_ids = tokenizer.encode(" " + output_text, add_special_tokens=False)
    ids.extend(out_ids)
    nums.extend([0.0] * len(out_ids))

    # Append EOT
    ids.append(EOT_TOKEN_ID)
    nums.append(0.0)

    return ids, nums


# =============================================================================
# Task generators
# =============================================================================

def gen_add(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a + b, sig_digits=cfg.get('max_output_digits', MAX_SIG_DIGITS))
    if isinstance(a, int) and isinstance(b, int): result = int(result)
    return f"ADD: {fmt(a)} + {fmt(b)} →", result

def gen_sub(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a - b, sig_digits=cfg.get('max_output_digits', MAX_SIG_DIGITS))
    if isinstance(a, int) and isinstance(b, int): result = int(result)
    return f"SUB: {fmt(a)} - {fmt(b)} →", result

def gen_sum(cfg):
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(sum(nums), sig_digits=cfg.get('max_output_digits', MAX_SIG_DIGITS))
    if all(isinstance(x, int) for x in nums): result = int(result)
    inp = " ".join(fmt(x) for x in nums)
    return f"SUM: {inp} →", result

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


TASK_GENERATORS = [gen_add, gen_sub, gen_sum, gen_min, gen_max]

def build_task_generators():
    return [(g, 1) for g in TASK_GENERATORS]


# =============================================================================
# Block packing
# =============================================================================

def pack_into_blocks(examples, block_size):
    all_ids = []
    all_nums = []
    block_ids = []
    block_nums = []
    n_blocks = 0
    examples_packed = 0

    for ids, nums in examples:
        example_len = len(ids)
        if example_len > block_size:
            continue
        if len(block_ids) + example_len > block_size:
            pad_len = block_size - len(block_ids)
            block_ids.extend([EOT_TOKEN_ID] * pad_len)
            block_nums.extend([0.0] * pad_len)
            all_ids.extend(block_ids)
            all_nums.extend(block_nums)
            n_blocks += 1
            block_ids = []
            block_nums = []
        block_ids.extend(ids)
        block_nums.extend(nums)
        examples_packed += 1

    if block_ids:
        pad_len = block_size - len(block_ids)
        block_ids.extend([EOT_TOKEN_ID] * pad_len)
        block_nums.extend([0.0] * pad_len)
        all_ids.extend(block_ids)
        all_nums.extend(block_nums)
        n_blocks += 1

    return all_ids, all_nums, n_blocks, examples_packed


def save_split(ids, nums, split, out_dir):
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
    print(f"  {split}: {arr_len:,} tokens, {num_count:,} <NUM> tokens "
          f"({num_count / arr_len * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate task data for Baby Luciole adapter (luciole_50k tokenizer)")
    parser.add_argument("--n-train", type=int, default=500000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--number-range", type=float, default=1000000000.0)
    parser.add_argument("--allow-negative", action="store_true", default=True)
    parser.add_argument("--no-negative", action="store_true")
    parser.add_argument("--allow-float", action="store_true", default=True)
    parser.add_argument("--integers-only", action="store_true")
    parser.add_argument("--digit-curriculum", action="store_true", default=True)
    parser.add_argument("--no-digit-curriculum", dest="digit_curriculum", action="store_false")
    parser.add_argument("--sig-digits-min", type=int, default=None)
    parser.add_argument("--sig-digits-max", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_negative: args.allow_negative = False
    if args.integers_only: args.allow_float = False
    if args.sig_digits_min is not None or args.sig_digits_max is not None:
        smin = 1 if args.sig_digits_min is None else int(args.sig_digits_min)
        smax = MAX_SIG_DIGITS if args.sig_digits_max is None else int(args.sig_digits_max)
        if smin < 1 or smax < 1 or smin > smax:
            raise ValueError("invalid sig-digits range; require 1 <= min <= max")
        args.sig_digits_min = min(smin, MAX_SIG_DIGITS)
        args.sig_digits_max = min(smax, MAX_SIG_DIGITS)
    else:
        args.sig_digits_min = None
        args.sig_digits_max = None
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'numtasks_124M_fe')

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_generators = build_task_generators()
    generators = [g for g, _ in task_generators]
    weights = [w for _, w in task_generators]

    max_output_digits = args.sig_digits_max if args.sig_digits_max is not None else MAX_SIG_DIGITS

    cfg = {
        'range': args.number_range,
        'neg': args.allow_negative,
        'flt': args.allow_float,
        'min_len': args.min_len,
        'max_len': args.max_len,
        'sig_digits_sampler': (lambda: random.randint(1, MAX_SIG_DIGITS)),
        'max_output_digits': max_output_digits,
    }

    task_names = [g.__name__[4:].upper() for g in generators]

    print("=" * 60)
    print("BABY LUCIOLE ADAPTER DATA GENERATOR")
    print("  Tokenizer: luciole_50k (SentencePiece)")
    print("=" * 60)
    print(f"  Tasks:           {', '.join(task_names)}")
    print(f"  Train examples:  {args.n_train:,}")
    print(f"  Val examples:    {args.n_val:,}")
    print(f"  Block size:      {args.block_size}")
    print(f"  Number range:    [-{args.number_range:g}, {args.number_range:g}]")
    if args.sig_digits_min is not None:
        print(f"  Digit range:     fixed {args.sig_digits_min}-{args.sig_digits_max}")
    else:
        print(f"  Digit curriculum:{'on' if args.digit_curriculum else 'off'}")
    print(f"  Output dir:      {args.out_dir}")
    print()

    # Verify tokenizer loads
    tokenizer = _get_tokenizer()
    test_ids = tokenizer.encode("ADD: 42 + 7 →", add_special_tokens=False)
    print(f"  Tokenizer test: 'ADD: 42 + 7 →' -> {test_ids}")
    print()

    first_example = None
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
                    max_digits=MAX_SIG_DIGITS)
            gen = random.choices(generators, weights=weights, k=1)[0]
            input_text, output = gen(cfg)
            examples_raw.append((input_text, output))
            task_counts[gen.__name__[4:].upper()] += 1

        if first_example is None and examples_raw:
            first_example = examples_raw[0]

        if split == 'train':
            print("\nSample examples:")
            seen = set()
            for input_text, output in examples_raw:
                task_type = input_text.split(":")[0]
                if task_type not in seen:
                    if isinstance(output, list):
                        out_str = " ".join(fmt(x) for x in output)
                        print(f"  {input_text} [{out_str}]")
                    else:
                        print(f"  {input_text} {fmt(output)}")
                    seen.add(task_type)
                if len(seen) == len(generators):
                    break
            print()

        print(f"Tokenizing {split} (NUM input, text output)...")
        tokenized = []
        for input_text, output in tqdm(examples_raw, desc=f"tokenizing {split}"):
            ids, nums = tokenize_task(input_text, output, max_digits=max_output_digits)
            tokenized.append((ids, nums))

        lengths = [len(ids) for ids, _ in tokenized]
        print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}")
        skipped = sum(1 for l in lengths if l > args.block_size)
        if skipped:
            print(f"  WARNING: {skipped} examples exceed block_size={args.block_size}")

        print(f"Packing into blocks of {args.block_size}...")
        all_ids, all_nums, n_blocks, n_packed = pack_into_blocks(
            tokenized, args.block_size)
        print(f"  {n_packed:,} examples packed into {n_blocks:,} blocks")
        print(f"  Task distribution: {task_counts}")

        print(f"Saving {split}...")
        save_split(all_ids, all_nums, split, args.out_dir)
        print()

    # Save meta
    meta = {
        'vocab_size': 50258,  # 50256 real + NUM(50256) + pad -> round up
        'num_token_id': NUM_TOKEN_ID,
        'eot_token_id': EOT_TOKEN_ID,
        'tokenizer': 'luciole_50k',
        'text_output': True,
        'tasks': task_names,
        'n_train': args.n_train,
        'n_val': args.n_val,
        'block_size': args.block_size,
        'number_range': args.number_range,
    }
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    if first_example is not None:
        input_text, output = first_example
        ids, nums = tokenize_task(input_text, output)
        print("Tokenization example:")
        print(f"  Input:  {input_text}")
        print(f"  Output: {output}")
        print(f"  Tokens ({len(ids)}): {ids[:40]}...")
        num_positions = [i for i, t in enumerate(ids) if t == NUM_TOKEN_ID]
        print(f"  <NUM> positions: {num_positions}")
        print(f"  Values at <NUM>: {[nums[i] for i in num_positions]}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
