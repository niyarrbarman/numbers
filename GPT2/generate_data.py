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
  SORT: 5 82 -3 → <S->E0D3D0D0 <S+>E0D5D0D0 <S+>E1D8D2D0
  ADD:  3 + 5 → <S+>E0D8D0D0
  SUB:  10 - 3 → <S+>E0D7D0D0
  MIN:  5 82 3 → <S+>E0D3D0D0
  MAX:  5 82 3 → <S+>E1D8D2D0
  SUM:  10 20 30 → <S+>E1D6D0D0
  COUNT also kept

Input numbers use <NUM> token with adapter embeddings.
Output numbers use SME (Sign-Mantissa-Exponent) text tokens.
All loss is now text cross-entropy — no separate num_loss.

Examples are packed into fixed-size blocks separated by EOT tokens.
Output is saved in the dual .bin format expected by fe/train.py.

Usage:
    python generate_data.py --out-dir /path/to/output
    python generate_data.py --number-range 1000 --n-train 1000000
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
    SME_SIGN_POS, SME_DIGIT_BASE,
)
import tiktoken

enc = tiktoken.get_encoding("gpt2")
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
# Task tokenization (SME-aware)
# =============================================================================

def tokenize_task(input_text, output):
    """Tokenize a task with SME encoding for output numbers.

    Args:
        input_text: str — input part (numbers become <NUM> with embeddings)
        output: str | number | list[number] — output part

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
            sme = number_to_sme_tokens(val)
            ids.extend(sme)
            nums.extend([0.0] * len(sme))
    else:
        # Single number output → SME tokens
        ids.append(220)  # space token before number
        nums.append(0.0)
        sme = number_to_sme_tokens(output)
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
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    if a > b:
        label = "GREATER"
    elif a < b:
        label = "LESS"
    else:
        label = "EQUAL"
    return f"CMP: {fmt(a)} {fmt(b)} →", label


def gen_gt(cfg):
    """Is first > second? → YES/NO"""
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    label = "YES" if a > b else "NO"
    return f"GT: {fmt(a)} {fmt(b)} →", label


def gen_is_pos(cfg):
    """Is number positive? → YES/NO"""
    a = sample_number(cfg['range'], cfg['neg'], cfg['flt'])
    label = "YES" if a > 0 else "NO"
    return f"IS_POS: {fmt(a)} →", label


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
    return f"IS_SORTED: {inp} →", label


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
    # All numbers here are INPUT (embedded via <NUM>), output is just text
    return f"CHECKSORT: {inp} → {out} CHECK →", label


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
    # All numbers (a, b, c) are INPUT — output is just YES/NO
    return f"CHECKADD: {fmt(a)} + {fmt(b)} = {fmt(c)} →", label


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
    return f"SUM_CMP: {fmt(a)} + {fmt(b)} vs {fmt(c)} + {fmt(d)} →", label


# =============================================================================
# REGRESSION task generators — number output (encoded as SME tokens)
# Each returns (input_text, number_or_list)
# =============================================================================

def gen_sort(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"SORT: {inp} →", sorted(nums)


def gen_add(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    result = round(a + b, 6)
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"ADD: {fmt(a)} + {fmt(b)} →", result


def gen_sub(cfg):
    a, b = sample_numbers(2, cfg['range'], cfg['neg'], cfg['flt'])
    result = round(a - b, 6)
    if isinstance(a, int) and isinstance(b, int):
        result = int(result)
    return f"SUB: {fmt(a)} - {fmt(b)} →", result


def gen_min(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"MIN: {inp} →", min(nums)


def gen_max(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"MAX: {inp} →", max(nums)


def gen_sum(cfg):
    n = random.randint(cfg['min_len'], min(8, cfg['max_len']))
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    result = round(sum(nums), 6)
    if all(isinstance(x, int) for x in nums):
        result = int(result)
    inp = " ".join(fmt(x) for x in nums)
    return f"SUM: {inp} →", result


def gen_count(cfg):
    n = random.randint(cfg['min_len'], cfg['max_len'])
    nums = sample_numbers(n, cfg['range'], cfg['neg'], cfg['flt'])
    inp = " ".join(fmt(x) for x in nums)
    return f"COUNT: {inp} →", n


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
    # Regression tasks (SME number output) — weight 1
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
    sme_count = int(np.sum(np.isin(np.array(ids, dtype=np.uint32),
                                    list(range(SME_SIGN_POS, SME_DIGIT_BASE + 10)))))
    print(f"  {split}: {arr_len:,} tokens, {num_count:,} <NUM> embeddings, "
          f"{sme_count:,} SME tokens ({sme_count / arr_len * 100:.1f}%)")


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
                                    'fe', 'data', 'numtasks_sme')

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
    print("NUMERICAL TASK DATA GENERATOR (SME OUTPUT)")
    print("=" * 60)
    print(f"  Tasks:           {', '.join(task_names)}")
    print(f"  Reasoning (2x):  {', '.join(g.__name__[4:].upper() for g, w in TASK_GENERATORS if w == 2)}")
    print(f"  SME output (1x): {', '.join(g.__name__[4:].upper() for g, w in TASK_GENERATORS if w == 1)}")
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

        # Generate structured examples
        examples_raw = []
        task_counts = {name: 0 for name in task_names}
        for _ in tqdm(range(n_examples), desc=f"generating {split}"):
            gen = random.choices(_GENERATORS, weights=_WEIGHTS, k=1)[0]
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
                if len(seen) == len(_GENERATORS):
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
        print(f"Tokenizing {split} (with SME for output numbers)...")
        tokenized = []
        for input_text, output in tqdm(examples_raw, desc=f"tokenizing {split}"):
            ids, nums = tokenize_task(input_text, output)
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
        'dataset': 'numtasks_sme',
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
    input_text, output = examples_raw[0]
    ids, nums = tokenize_task(input_text, output)
    print(f"  Input: {input_text}")
    print(f"  Output: {output}")
    print(f"  Tokens: {ids}")
    print(f"  Nums:   {nums}")
    print(f"  Length:  {len(ids)} tokens")

    print(f"\nDone. To train:")
    print(f"  python fe/train.py dataset=numtasks_sme data_dir={args.out_dir}")


if __name__ == '__main__':
    main()
