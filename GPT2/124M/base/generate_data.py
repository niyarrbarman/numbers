"""
Generate synthetic numerical task data for base GPT-2 training (no number embeddings).

This matches the same task mix used for SME data generation, but serializes all
numbers directly in plain text and writes only:
  - train.bin / val.bin (uint16 GPT-2 token ids)
  - meta.pkl
"""

import os
import argparse
import pickle
import random
import math

import numpy as np
from tqdm import tqdm
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc.eot_token

# Keep base data sampling aligned with FE range policy.
MAX_SIG_DIGITS = 15
EXP_MIN = -9
EXP_MAX = 9


# =============================================================================
# Number sampling
# =============================================================================

def _max_exp_for_range(number_range):
    """Highest exponent that can fit in the configured absolute range."""
    nr = max(float(number_range), 10.0 ** EXP_MIN)
    hi = int(math.floor(math.log10(nr)))
    return max(EXP_MIN, min(EXP_MAX, hi))


def _canonicalize_float(val, sig_digits=MAX_SIG_DIGITS):
    """Round to a stable number of significant digits for reproducible text."""
    return float(format(float(val), f".{sig_digits}g"))


def _sample_digits_from_bands(bands, max_digits=MAX_SIG_DIGITS):
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


def build_sig_digits_sampler(progress, use_curriculum=True, max_digits=MAX_SIG_DIGITS):
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
    max_cap = MAX_SIG_DIGITS
    min_digits = min(min_digits, max_cap)
    max_digits = min(max_digits, max_cap)
    return lambda: random.randint(min_digits, max_digits)


def sample_number(number_range, allow_negative, allow_float, sig_digits_sampler=None):
    """Sample one number with broad exponent coverage and mixed precision."""
    max_exp = _max_exp_for_range(number_range)
    max_abs = max(float(number_range), 10.0 ** EXP_MIN)
    return_as_int = False

    if allow_float:
        mode = random.random()

        if mode < 0.65:
            # Main mode: sample exponent explicitly so small/large magnitudes are seen.
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
            # Uniform background coverage.
            val = random.uniform(0.0, max_abs)
            sig_digits = (
                int(sig_digits_sampler()) if sig_digits_sampler is not None
                else random.randint(1, MAX_SIG_DIGITS)
            )
            sig_digits = max(1, min(sig_digits, MAX_SIG_DIGITS))
            val = _canonicalize_float(val, sig_digits=sig_digits)
        else:
            # Integer/round-number bias for short textual outputs.
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
# Task tokenization (plain-text numbers)
# =============================================================================

def tokenize_task(input_text, output):
    """Tokenize a task where numbers stay in normal text form."""
    if isinstance(output, str):
        output_text = output
    elif isinstance(output, list):
        output_text = " ".join(fmt(x) for x in output)
    else:
        output_text = fmt(output)

    text = f"{input_text} {output_text}"
    ids = enc.encode_ordinary(text)
    ids.append(EOT_TOKEN)
    return ids


# =============================================================================
# Task generators — arithmetic operations only
# =============================================================================

def gen_add(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a + b, sig_digits=cfg.get('max_output_digits', MAX_SIG_DIGITS))
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"ADD: {fmt(a)} + {fmt(b)} →", result


def gen_sub(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(a - b, sig_digits=cfg.get('max_output_digits', MAX_SIG_DIGITS))
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"SUB: {fmt(a)} - {fmt(b)} →", result


def gen_sum(cfg):
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'], cfg.get('sig_digits_sampler'))
    result = _canonicalize_float(sum(nums), sig_digits=cfg.get('max_output_digits', MAX_SIG_DIGITS))
    if all(isinstance(x, int) for x in nums):
        result = int(result)
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


# =============================================================================
# Task registry
# =============================================================================

TASK_GENERATORS = [
    gen_add,
    gen_sub,
    gen_sum,
    gen_min,
    gen_max,
]


def build_task_generators():
    return [(g, 1) for g in TASK_GENERATORS]


# =============================================================================
# Block packing
# =============================================================================

def pack_into_blocks(examples, block_size):
    """Pack tokenized examples into fixed-size blocks."""
    all_ids = []
    block_ids = []
    n_blocks = 0
    examples_packed = 0

    for ids in examples:
        example_len = len(ids)

        if example_len > block_size:
            continue

        if len(block_ids) + example_len > block_size:
            pad_len = block_size - len(block_ids)
            block_ids.extend([EOT_TOKEN] * pad_len)
            all_ids.extend(block_ids)
            n_blocks += 1
            block_ids = []

        block_ids.extend(ids)
        examples_packed += 1

    if block_ids:
        pad_len = block_size - len(block_ids)
        block_ids.extend([EOT_TOKEN] * pad_len)
        all_ids.extend(block_ids)
        n_blocks += 1

    return all_ids, n_blocks, examples_packed


def save_split(ids, split, out_dir):
    """Save tokenized ids to {split}.bin."""
    os.makedirs(out_dir, exist_ok=True)
    arr_len = len(ids)

    tok_path = os.path.join(out_dir, f'{split}.bin')
    tok_arr = np.memmap(tok_path, dtype=np.uint16, mode='w+', shape=(arr_len,))
    tok_arr[:] = np.array(ids, dtype=np.uint16)
    tok_arr.flush()

    print(f"  {split}: {arr_len:,} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-task numerical data for base GPT-2 (plain text numbers)")
    parser.add_argument("--n-train", type=int, default=50000000,
                        help="Number of training examples (default: 50000000)")
    parser.add_argument("--n-val", type=int, default=500000,
                        help="Number of validation examples (default: 500000)")
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
    parser.add_argument("--digit-curriculum", action="store_true", default=True,
                        help="Enable mantissa digit curriculum (default: on)")
    parser.add_argument("--no-digit-curriculum", dest="digit_curriculum", action="store_false",
                        help="Disable mantissa digit curriculum")
    parser.add_argument("--sig-digits-min", type=int, default=None,
                        help="Force minimum significant digits for sampled floats")
    parser.add_argument("--sig-digits-max", type=int, default=None,
                        help="Force maximum significant digits for sampled floats")
    parser.add_argument(
        "--output-sig-digits-max",
        type=int,
        default=None,
        help="Override max significant digits used for computed *output* numbers. "
             "If unset, output precision follows --sig-digits-max for backward compatibility.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_negative:
        args.allow_negative = False
    if args.integers_only:
        args.allow_float = False
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
    if args.output_sig_digits_max is not None:
        args.output_sig_digits_max = int(args.output_sig_digits_max)
        if args.output_sig_digits_max < 1:
            raise ValueError("--output-sig-digits-max must be >= 1")
        args.output_sig_digits_max = min(args.output_sig_digits_max, MAX_SIG_DIGITS)
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'numtasks_124M_base')

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_generators = build_task_generators()
    generators = [g for g, _ in task_generators]
    weights = [w for _, w in task_generators]

    # max_output_digits: cap output precision. Historically this tracked input
    # sig-digits; allow decoupling for OOD eval generation.
    if args.output_sig_digits_max is not None:
        max_output_digits = args.output_sig_digits_max
    elif args.sig_digits_max is not None:
        max_output_digits = args.sig_digits_max
    else:
        max_output_digits = MAX_SIG_DIGITS

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
    print("124M BASE DATA GENERATOR (PLAIN TEXT)")
    print("=" * 60)
    print(f"  Tasks:           {', '.join(task_names)}")
    print(f"  Train examples:  {args.n_train:,}")
    print(f"  Val examples:    {args.n_val:,}")
    print(f"  Block size:      {args.block_size}")
    print(f"  Sequence length: {args.min_len}-{args.max_len}")
    print(f"  Number range:    [-{args.number_range:g}, {args.number_range:g}]")
    print(f"  Exponent target: E{EXP_MIN}..E{EXP_MAX}")
    print(f"  Sig digits max:  {MAX_SIG_DIGITS}")
    if args.sig_digits_min is not None:
        print(f"  Digit range:     fixed {args.sig_digits_min}-{args.sig_digits_max}")
        print("  Digit curriculum:off (fixed range overrides curriculum)")
    else:
        print(f"  Digit curriculum:{'on' if args.digit_curriculum else 'off'} "
              f"(phase1 1-4, phase2 5-8, phase3 1-{MAX_SIG_DIGITS})")
    if args.output_sig_digits_max is not None:
        print(f"  Output digits:   capped at {max_output_digits} (override)")
    else:
        print(f"  Output digits:   capped at {max_output_digits}")
    print(f"  Allow float:     {args.allow_float}")
    print(f"  Output dir:      {args.out_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    # Generate, tokenize, and pack each split
    first_example = None
    for split, n_examples in [('train', args.n_train), ('val', args.n_val)]:
        print(f"Generating {split} ({n_examples:,} examples)...")

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
                    max_digits=MAX_SIG_DIGITS,
                )
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
                    if isinstance(output, str):
                        print(f"  {input_text} {output}")
                    elif isinstance(output, list):
                        out_str = " ".join(fmt(x) for x in output)
                        print(f"  {input_text} [{out_str}]")
                    else:
                        print(f"  {input_text} {fmt(output)}")
                    seen.add(task_type)
                if len(seen) == len(generators):
                    break
            print()

        print(f"Tokenizing {split} (plain text numbers)...")
        tokenized = []
        for input_text, output in tqdm(examples_raw, desc=f"tokenizing {split}"):
            ids = tokenize_task(input_text, output)
            tokenized.append(ids)

        lengths = [len(ids) for ids in tokenized]
        print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}")
        skipped = sum(1 for l in lengths if l > args.block_size)
        if skipped:
            print(f"  WARNING: {skipped} examples exceed block_size={args.block_size}, "
                  f"will be skipped")

        print(f"Packing into blocks of {args.block_size}...")
        all_ids, n_blocks, n_packed = pack_into_blocks(tokenized, args.block_size)
        print(f"  {n_packed:,} examples packed into {n_blocks:,} blocks")
        print(f"  Examples per block: {n_packed / n_blocks:.1f} avg")
        print(f"  Task distribution: {task_counts}")

        print(f"Saving {split}...")
        save_split(all_ids, split, args.out_dir)
        print()

    meta = {
        'vocab_size': 50304,
        'dataset': 'numtasks_124M_base',
        'tasks': task_names,
        'n_train': args.n_train,
        'n_val': args.n_val,
        'block_size': args.block_size,
        'number_range': args.number_range,
        'exp_min_target': EXP_MIN,
        'exp_max_target': EXP_MAX,
        'sig_digits_max': MAX_SIG_DIGITS,
        'plain_text_numbers': True,
        'digit_curriculum': bool(args.digit_curriculum),
        'sig_digits_min': args.sig_digits_min,
        'sig_digits_max': args.sig_digits_max,
        'output_sig_digits_max': args.output_sig_digits_max,
        'output_max_digits_effective': max_output_digits,
    }
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    if first_example is not None:
        input_text, output = first_example
        ids = tokenize_task(input_text, output)
        print("Tokenization example:")
        print(f"  Input:  {input_text}")
        print(f"  Output: {output}")
        print(f"  Tokens: {ids}")
        print(f"  Length: {len(ids)} tokens")

    print("\nDone. To train:")
    print(f"  python base/train.py data_dir={args.out_dir}")


if __name__ == '__main__':
    main()
