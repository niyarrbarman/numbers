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

REGRESSION tasks (number output — trains via num_loss):
  SORT: 5 82 -3 → -3 5 82
  ADD:  3 + 5 → 8
  SUB:  10 - 3 → 7
  MIN:  5 82 3 → 3
  MAX:  5 82 3 → 82
  SUM:  10 20 30 → 60
  CMP and COUNT also kept

Reasoning tasks are weighted 2x to force numerical understanding through text_loss.

Examples are packed into fixed-size blocks separated by EOT tokens.
Output is saved in the dual .bin format expected by fe/train.py.

Usage:
    python generate_data.py --out-dir /path/to/output
    python generate_data.py --number-range 100000 --n-train 1000000
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
from fe.prepare import process_text_with_numbers, NUM_TOKEN_ID

EOT_TOKEN = 50256


# =============================================================================
# Number sampling
# =============================================================================

def sample_number(number_range, allow_negative, allow_float):
    """Sample a single random number with controlled variety."""
    if random.random() < 0.3:
        val = random.uniform(0, 10)
    elif random.random() < 0.5:
        val = random.uniform(10, min(1000, number_range))
    else:
        val = random.uniform(0, number_range)

    if allow_float and random.random() < 0.5:
        decimals = random.randint(1, 4)
        val = round(val, decimals)
    else:
        val = int(val)

    if allow_negative and random.random() < 0.3:
        val = -val

    return val


def sample_numbers(n, number_range, allow_negative, allow_float):
    """Sample a list of n random numbers."""
    return [sample_number(number_range, allow_negative, allow_float) for _ in range(n)]


def fmt(val):
    """Format a number for text representation."""
    if isinstance(val, int):
        return str(val)
    return f"{val:g}"


# =============================================================================
# REASONING task generators — text output depends on number values
# =============================================================================

def gen_cmp(cfg):
    """Compare two numbers → GREATER/LESS/EQUAL"""
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    if a > b:
        label = "GREATER"
    elif a < b:
        label = "LESS"
    else:
        label = "EQUAL"
    return f"CMP: {fmt(a)} {fmt(b)} → {label}"


def gen_gt(cfg):
    """Is first > second? → YES/NO"""
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    label = "YES" if a > b else "NO"
    return f"GT: {fmt(a)} {fmt(b)} → {label}"


def gen_is_pos(cfg):
    """Is number positive? → YES/NO"""
    a = sample_number(cfg['range'], cfg['neg'], cfg['flt'])
    label = "YES" if a > 0 else "NO"
    return f"IS_POS: {fmt(a)} → {label}"


def gen_is_sorted(cfg):
    """Is sequence sorted ascending? → YES/NO"""
    n = random.randint(cfg['min_len'], min(10, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    # 50% of the time, actually sort it so we get balanced YES/NO
    if random.random() < 0.5:
        nums = sorted(nums)
    inp = " ".join(fmt(x) for x in nums)
    is_sorted = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))
    label = "YES" if is_sorted else "NO"
    return f"IS_SORTED: {inp} → {label}"


def gen_checksort(cfg):
    """Verify if a proposed sort is correct → YES/NO"""
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
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
    return f"CHECKSORT: {inp} → {out} CHECK → {label}"


def gen_checkadd(cfg):
    """Verify if a + b = c is correct → YES/NO"""
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    correct = round(a + b, 6)
    if isinstance(a, int) and isinstance(b, int):
        correct = int(correct)
    # 50% correct, 50% wrong (add noise to result)
    if random.random() < 0.5:
        c = correct
        label = "YES"
    else:
        # Perturb by a meaningful amount
        noise = sample_number(max(abs(correct) * 0.5, 10), True, cfg['flt'])
        c = round(correct + noise, 4) if cfg['flt'] else int(correct + noise)
        label = "NO" if c != correct else "YES"
    return f"CHECKADD: {fmt(a)} + {fmt(b)} = {fmt(c)} → {label}"


def gen_sum_cmp(cfg):
    """Which pair sums to more? → FIRST/SECOND/EQUAL"""
    a, b, c, d = sample_numbers(4, cfg['range'], cfg['neg'], cfg['flt'])
    sum1 = a + b
    sum2 = c + d
    if sum1 > sum2:
        label = "FIRST"
    elif sum1 < sum2:
        label = "SECOND"
    else:
        label = "EQUAL"
    return f"SUM_CMP: {fmt(a)} + {fmt(b)} vs {fmt(c)} + {fmt(d)} → {label}"


# =============================================================================
# REGRESSION task generators — number output
# =============================================================================

def gen_sort(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    out = " ".join(fmt(x) for x in sorted(nums))
    return f"SORT: {inp} → {out}"


def gen_add(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    result = round(a + b, 6)
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"ADD: {fmt(a)} + {fmt(b)} → {fmt(result)}"


def gen_sub(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    result = round(a - b, 6)
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"SUB: {fmt(a)} - {fmt(b)} → {fmt(result)}"


def gen_min(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"MIN: {inp} → {fmt(min(nums))}"


def gen_max(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"MAX: {inp} → {fmt(max(nums))}"


def gen_sum(cfg):
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    result = round(sum(nums), 6)
    if all(isinstance(x, int) for x in nums):
        result = int(result)
    inp = " ".join(fmt(x) for x in nums)
    return f"SUM: {inp} → {fmt(result)}"


def gen_count(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"COUNT: {inp} → {n}"


# =============================================================================
# Task registry with weights
# =============================================================================

# (generator, weight) — reasoning tasks get 2x weight
TASK_GENERATORS = [
    # Reasoning tasks (text output depends on values) — weight 2
    (gen_cmp, 2),
    (gen_gt, 2),
    (gen_is_pos, 2),
    (gen_is_sorted, 2),
    (gen_checksort, 2),
    (gen_checkadd, 2),
    (gen_sum_cmp, 2),
    # Regression tasks (number output) — weight 1
    (gen_sort, 1),
    (gen_add, 1),
    (gen_sub, 1),
    (gen_min, 1),
    (gen_max, 1),
    (gen_sum, 1),
    (gen_count, 1),
]

# Build weighted list for random.choices
_GENERATORS = [g for g, _ in TASK_GENERATORS]
_WEIGHTS = [w for _, w in TASK_GENERATORS]


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
        example_len = len(ids)  # includes EOT from process_text_with_numbers

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
    print(f"  {split}: {arr_len:,} tokens, {num_count:,} numbers "
          f"({num_count / arr_len * 100:.1f}% of tokens)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-task numerical data for number-aware GPT-2")
    parser.add_argument("--n-train", type=int, default=1000000,
                        help="Number of training examples (default: 1000000)")
    parser.add_argument("--n-val", type=int, default=5000,
                        help="Number of validation examples (default: 5000)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Block size for packing (must match train.py block_size)")
    parser.add_argument("--min-len", type=int, default=2,
                        help="Min sequence length for list tasks (default: 2)")
    parser.add_argument("--max-len", type=int, default=15,
                        help="Max sequence length for list tasks (default: 15)")
    parser.add_argument("--number-range", type=float, default=100000,
                        help="Max absolute value of numbers (default: 100000)")
    parser.add_argument("--allow-negative", action="store_true", default=True)
    parser.add_argument("--no-negative", action="store_true")
    parser.add_argument("--allow-float", action="store_true", default=True)
    parser.add_argument("--integers-only", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_negative:
        args.allow_negative = False
    if args.integers_only:
        args.allow_float = False
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'fe', 'data', 'numtasks_reasoning')

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = {
        'range': args.number_range,
        'neg': args.allow_negative,
        'flt': args.allow_float,
        'min_len': args.min_len,
        'max_len': args.max_len,
    }

    task_names = [g.__name__[4:].upper() for g in _GENERATORS]

    print("=" * 60)
    print("NUMERICAL TASK DATA GENERATOR (REASONING)")
    print("=" * 60)
    print(f"  Tasks:           {', '.join(task_names)}")
    print(f"  Reasoning (2x):  {', '.join(g.__name__[4:].upper() for g, w in TASK_GENERATORS if w == 2)}")
    print(f"  Regression (1x): {', '.join(g.__name__[4:].upper() for g, w in TASK_GENERATORS if w == 1)}")
    print(f"  Train examples:  {args.n_train:,}")
    print(f"  Val examples:    {args.n_val:,}")
    print(f"  Block size:      {args.block_size}")
    print(f"  Sequence length: {args.min_len}-{args.max_len}")
    print(f"  Number range:    [-{args.number_range:g}, {args.number_range:g}]")
    print(f"  Allow float:     {args.allow_float}")
    print(f"  Output dir:      {args.out_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    # --- Generate and tokenize examples ---
    for split, n_examples in [('train', args.n_train), ('val', args.n_val)]:
        print(f"Generating {split} ({n_examples:,} examples)...")

        # Generate text examples
        texts = []
        task_counts = {name: 0 for name in task_names}
        for _ in tqdm(range(n_examples), desc=f"generating {split}"):
            gen = random.choices(_GENERATORS, weights=_WEIGHTS, k=1)[0]
            text = gen(cfg)
            texts.append(text)
            task_counts[gen.__name__[4:].upper()] += 1

        # Show samples
        if split == 'train':
            print("\nSample examples:")
            seen = set()
            for text in texts:
                task_type = text.split(":")[0]
                if task_type not in seen:
                    print(f"  {text}")
                    seen.add(task_type)
                if len(seen) == len(_GENERATORS):
                    break
            print()

        # Tokenize
        print(f"Tokenizing {split}...")
        tokenized = []
        for text in tqdm(texts, desc=f"tokenizing {split}"):
            ids, nums = process_text_with_numbers(text)
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
        'dataset': 'numtasks_reasoning',
        'tasks': task_names,
        'n_train': args.n_train,
        'n_val': args.n_val,
        'block_size': args.block_size,
        'number_range': args.number_range,
    }
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # Show tokenization of one example
    print("Tokenization example:")
    text = texts[0]
    ids, nums = process_text_with_numbers(text)
    print(f"  Text:   {text}")
    print(f"  Tokens: {ids}")
    print(f"  Nums:   {nums}")
    print(f"  Length:  {len(ids)} tokens")

    print(f"\nDone. To train:")
    print(f"  python fe/train.py dataset=numtasks_reasoning data_dir={args.out_dir}")


if __name__ == '__main__':
    main()
