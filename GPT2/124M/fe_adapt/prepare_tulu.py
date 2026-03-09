"""Prepare tulu-3 math data for base vs adapted LoRA experiment.

Reads the raw HuggingFace dataset and produces two directories:
  base/     - plain text (no <NUM>), for base model LoRA
  adapted/  - <NUM> in user messages, for NumberEncoder-adapted model

Each directory contains:
  {train,val,test}.bin       - uint16 token IDs
  {train,val,test}_nums.bin  - float32 parallel number values
  test_examples.json         - raw test examples for generation eval

Usage:
  python prepare_tulu.py [--raw_dir PATH] [--out_dir PATH]
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import _get_tokenizer, NUM_TOKEN_ID, EOT_TOKEN_ID, NUMBER_PATTERN
from datasets import load_from_disk

SEED = 42
VAL_FRAC = 0.06
TEST_FRAC = 0.06


def format_conversation(messages):
    """Format messages as a single text string."""
    parts = []
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content'].strip()
        parts.append(f"{role}: {content}")
    return '\n'.join(parts)


def process_content_with_numbers(text):
    """Replace numbers with <NUM> token. Returns (ids, nums) without EOT."""
    tokenizer = _get_tokenizer()
    ids = []
    nums = []
    last_end = 0
    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            seg_ids = tokenizer.encode(text[last_end:start], add_special_tokens=False)
            ids.extend(seg_ids)
            nums.extend([0.0] * len(seg_ids))
        num_str = match.group()
        try:
            value = float(num_str)
        except ValueError:
            seg_ids = tokenizer.encode(num_str, add_special_tokens=False)
            ids.extend(seg_ids)
            nums.extend([0.0] * len(seg_ids))
            last_end = end
            continue
        ids.append(NUM_TOKEN_ID)
        nums.append(value)
        last_end = end
    if last_end < len(text):
        seg_ids = tokenizer.encode(text[last_end:], add_special_tokens=False)
        ids.extend(seg_ids)
        nums.extend([0.0] * len(seg_ids))
    return ids, nums


def tokenize_base(messages):
    """Tokenize conversation as plain text, no <NUM> replacement."""
    tokenizer = _get_tokenizer()
    text = format_conversation(messages)
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids.append(EOT_TOKEN_ID)
    nums = [0.0] * len(ids)
    return ids, nums


def tokenize_adapted(messages):
    """Tokenize with <NUM> in user messages, plain text in assistant."""
    tokenizer = _get_tokenizer()
    ids = []
    nums = []

    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content'].strip()

        # Role prefix (with newline separator between turns)
        if i > 0:
            prefix = f"\n{role.capitalize()}: "
        else:
            prefix = f"{role.capitalize()}: "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        ids.extend(prefix_ids)
        nums.extend([0.0] * len(prefix_ids))

        if role == 'user':
            # Replace numbers with <NUM>
            c_ids, c_nums = process_content_with_numbers(content)
            ids.extend(c_ids)
            nums.extend(c_nums)
        else:
            # Assistant: plain text
            c_ids = tokenizer.encode(content, add_special_tokens=False)
            ids.extend(c_ids)
            nums.extend([0.0] * len(c_ids))

    # End of conversation
    ids.append(EOT_TOKEN_ID)
    nums.append(0.0)
    return ids, nums


def process_split(examples, base_dir, adapted_dir, split_name, save_raw=False):
    """Tokenize examples and save binary files for both base and adapted."""
    all_base_ids, all_base_nums = [], []
    all_adapted_ids, all_adapted_nums = [], []

    for ex in examples:
        messages = ex['messages']

        b_ids, b_nums = tokenize_base(messages)
        all_base_ids.extend(b_ids)
        all_base_nums.extend(b_nums)

        a_ids, a_nums = tokenize_adapted(messages)
        all_adapted_ids.extend(a_ids)
        all_adapted_nums.extend(a_nums)

    # Save base
    np.array(all_base_ids, dtype=np.uint16).tofile(
        os.path.join(base_dir, f'{split_name}.bin'))
    np.array(all_base_nums, dtype=np.float32).tofile(
        os.path.join(base_dir, f'{split_name}_nums.bin'))

    # Save adapted
    np.array(all_adapted_ids, dtype=np.uint16).tofile(
        os.path.join(adapted_dir, f'{split_name}.bin'))
    np.array(all_adapted_nums, dtype=np.float32).tofile(
        os.path.join(adapted_dir, f'{split_name}_nums.bin'))

    n_base_num = sum(1 for x in all_base_ids if x == NUM_TOKEN_ID)
    n_adapted_num = sum(1 for x in all_adapted_ids if x == NUM_TOKEN_ID)

    print(f"  {split_name}: {len(examples)} examples")
    print(f"    base:    {len(all_base_ids):>10,} tokens, {n_base_num:>6} <NUM>")
    print(f"    adapted: {len(all_adapted_ids):>10,} tokens, {n_adapted_num:>6} <NUM>")

    if save_raw:
        raw = [{'messages': ex['messages']} for ex in examples]
        for d in [base_dir, adapted_dir]:
            with open(os.path.join(d, 'test_examples.json'), 'w') as f:
                json.dump(raw, f, indent=2)
        print(f"    saved {len(raw)} raw test examples to test_examples.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir',
                        default='/tmpdir/m24047brmn/numbers/data/tulu3_math_grade/raw')
    parser.add_argument('--out_dir',
                        default='/tmpdir/m24047brmn/numbers/data/tulu3_math_grade')
    args = parser.parse_args()

    base_dir = os.path.join(args.out_dir, 'base')
    adapted_dir = os.path.join(args.out_dir, 'adapted')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(adapted_dir, exist_ok=True)

    print(f"Loading raw dataset from {args.raw_dir}")
    ds = load_from_disk(args.raw_dir)

    # Get all examples (dataset may be a DatasetDict with 'train' key or plain)
    if hasattr(ds, 'keys') and 'train' in ds:
        all_examples = list(ds['train'])
    else:
        all_examples = list(ds)

    print(f"Total examples: {len(all_examples)}")

    # Shuffle and split
    import random
    rng = random.Random(SEED)
    indices = list(range(len(all_examples)))
    rng.shuffle(indices)
    all_examples = [all_examples[i] for i in indices]

    n = len(all_examples)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    n_train = n - n_val - n_test

    train_ex = all_examples[:n_train]
    val_ex = all_examples[n_train:n_train + n_val]
    test_ex = all_examples[n_train + n_val:]

    print(f"Split: train={n_train}, val={n_val}, test={n_test}\n")

    process_split(train_ex, base_dir, adapted_dir, 'train')
    process_split(val_ex, base_dir, adapted_dir, 'val')
    process_split(test_ex, base_dir, adapted_dir, 'test', save_raw=True)

    print(f"\nDone! Data saved to:")
    print(f"  base:    {base_dir}")
    print(f"  adapted: {adapted_dir}")


if __name__ == '__main__':
    main()
