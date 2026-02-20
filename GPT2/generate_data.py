"""
Generate synthetic sorting data for number-aware GPT-2 training.

Produces examples in the format:
    INPUT: 5 82.26 -3 0.7 42 SORTED: -3 0.7 5 42 82.26

Each example is tokenized using the number-aware tokenizer from fe/prepare.py,
which replaces detected numbers with <NUM> tokens and stores float values in
a parallel array. Output is saved in the dual .bin format expected by fe/train.py.

Usage:
    python generate_data.py                        # defaults
    python generate_data.py --n-train 500000 --n-val 5000
    python generate_data.py --min-len 3 --max-len 20 --out-dir fe/data/sorting
    python generate_data.py --number-range 1e6 --allow-negative --allow-float
"""

import os
import sys
import argparse
import pickle
import random

import numpy as np
from tqdm import tqdm

# Import the tokenizer from fe/prepare.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fe'))
from fe.prepare import process_text_with_numbers, NUM_TOKEN_ID


def sample_number(number_range, allow_negative, allow_float):
    """Sample a single random number with controlled variety."""
    # Decide the magnitude range
    if random.random() < 0.3:
        # Small numbers (0-10)
        val = random.uniform(0, 10)
    elif random.random() < 0.5:
        # Medium numbers (10-1000)
        val = random.uniform(10, min(1000, number_range))
    else:
        # Full range
        val = random.uniform(0, number_range)

    # Decide integer vs float
    if allow_float and random.random() < 0.5:
        # Round to 1-4 decimal places for readable floats
        decimals = random.randint(1, 4)
        val = round(val, decimals)
    else:
        val = int(val)

    # Decide sign
    if allow_negative and random.random() < 0.3:
        val = -val

    return val


def format_number(val):
    """Format a number for the text representation."""
    if isinstance(val, int):
        return str(val)
    # Remove trailing zeros for cleaner floats: 3.10 -> 3.1, but keep 3.0
    s = f"{val:g}"
    return s


def generate_example(min_len, max_len, number_range, allow_negative, allow_float):
    """Generate a single sorting example as text.

    Returns:
        text: str like "INPUT: 5 82.26 -3 SORTED: -3 5 82.26"
    """
    seq_len = random.randint(min_len, max_len)
    numbers = [sample_number(number_range, allow_negative, allow_float)
               for _ in range(seq_len)]

    input_str = " ".join(format_number(n) for n in numbers)
    sorted_str = " ".join(format_number(n) for n in sorted(numbers))

    return f"INPUT: {input_str} SORTED: {sorted_str}"


def tokenize_examples(examples, desc="tokenizing"):
    """Tokenize a list of text examples into parallel token/number arrays."""
    all_ids = []
    all_nums = []
    for text in tqdm(examples, desc=desc):
        ids, nums = process_text_with_numbers(text)
        all_ids.extend(ids)
        all_nums.extend(nums)
    return all_ids, all_nums


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
    return tok_path, num_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sorting data for number-aware GPT-2")
    parser.add_argument("--n-train", type=int, default=100000,
                        help="Number of training examples (default: 100000)")
    parser.add_argument("--n-val", type=int, default=2000,
                        help="Number of validation examples (default: 2000)")
    parser.add_argument("--min-len", type=int, default=2,
                        help="Min sequence length (default: 2)")
    parser.add_argument("--max-len", type=int, default=15,
                        help="Max sequence length (default: 15)")
    parser.add_argument("--number-range", type=float, default=10000,
                        help="Max absolute value of numbers (default: 10000)")
    parser.add_argument("--allow-negative", action="store_true", default=True,
                        help="Include negative numbers (default: True)")
    parser.add_argument("--no-negative", action="store_true",
                        help="Disable negative numbers")
    parser.add_argument("--allow-float", action="store_true", default=True,
                        help="Include floating point numbers (default: True)")
    parser.add_argument("--integers-only", action="store_true",
                        help="Only use integers")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: fe/data/sorting)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.no_negative:
        args.allow_negative = False
    if args.integers_only:
        args.allow_float = False
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'fe', 'data', 'sorting')

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("SORTING DATA GENERATOR")
    print("=" * 60)
    print(f"  Train examples:  {args.n_train:,}")
    print(f"  Val examples:    {args.n_val:,}")
    print(f"  Sequence length: {args.min_len}-{args.max_len}")
    print(f"  Number range:    [{'- ' if args.allow_negative else ''}"
          f"{args.number_range:g}, {args.number_range:g}]")
    print(f"  Allow float:     {args.allow_float}")
    print(f"  Output dir:      {args.out_dir}")
    print(f"  Seed:            {args.seed}")
    print()

    # --- Generate examples ---
    print("Generating training examples...")
    train_texts = [generate_example(args.min_len, args.max_len, args.number_range,
                                    args.allow_negative, args.allow_float)
                   for _ in tqdm(range(args.n_train), desc="generating train")]

    print("Generating validation examples...")
    val_texts = [generate_example(args.min_len, args.max_len, args.number_range,
                                  args.allow_negative, args.allow_float)
                 for _ in tqdm(range(args.n_val), desc="generating val")]

    # Show a few examples
    print("\nSample examples:")
    for i in range(min(5, len(train_texts))):
        print(f"  {train_texts[i]}")
    print()

    # --- Tokenize ---
    print("Tokenizing training data...")
    train_ids, train_nums = tokenize_examples(train_texts, desc="tokenizing train")

    print("Tokenizing validation data...")
    val_ids, val_nums = tokenize_examples(val_texts, desc="tokenizing val")

    # --- Save ---
    print(f"\nSaving to {args.out_dir}/")
    save_split(train_ids, train_nums, 'train', args.out_dir)
    save_split(val_ids, val_nums, 'val', args.out_dir)

    # Save meta
    meta = {
        'vocab_size': 50304,
        'num_token_id': NUM_TOKEN_ID,
        'dataset': 'sorting',
        'n_train': args.n_train,
        'n_val': args.n_val,
        'min_len': args.min_len,
        'max_len': args.max_len,
        'number_range': args.number_range,
        'allow_negative': args.allow_negative,
        'allow_float': args.allow_float,
    }
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # Show tokenization of one example
    print(f"\nTokenization example:")
    example = train_texts[0]
    ids, nums = process_text_with_numbers(example)
    print(f"  Text:   {example}")
    print(f"  Tokens: {ids}")
    print(f"  Nums:   {nums}")
    print(f"  Length:  {len(ids)} tokens")

    print(f"\nDone. To train: python fe/train.py dataset=sorting")


if __name__ == '__main__':
    main()
