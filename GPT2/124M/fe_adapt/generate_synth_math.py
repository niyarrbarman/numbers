"""Generate synthetic arithmetic training data for base vs adapted LoRA.

Produces two directories (like prepare_tulu.py) with binary files:
  base/     - plain text (no <NUM>)
  adapted/  - <NUM> in user messages, plain text in assistant messages

Format: User/Assistant conversations with simple arithmetic:
  - Addition, subtraction, multiplication, division
  - Comparisons
  - Percentages
  - Multi-step

The SAME problems are used for both base and adapted — only tokenization
differs. This means any performance difference is purely from the
NumberEncoder, not from data differences.

Usage:
  python generate_synth_math.py --out_dir /path/to/synth_data --n_train 50000
"""

import os
import sys
import json
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import (_get_tokenizer, NUM_TOKEN_ID, EOT_TOKEN_ID,
                      NUMBER_PATTERN)


# =============================================================================
# Problem generators
# =============================================================================

def gen_addition(rng):
    """Simple addition."""
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    ans = a + b
    return (f"What is {a} + {b}?",
            f"{a} + {b} = {ans}")

def gen_subtraction(rng):
    """Simple subtraction (non-negative result)."""
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    if a < b:
        a, b = b, a
    ans = a - b
    return (f"What is {a} - {b}?",
            f"{a} - {b} = {ans}")

def gen_multiplication(rng):
    """Multiplication with manageable numbers."""
    a = rng.randint(2, 999)
    b = rng.randint(2, 999)
    ans = a * b
    return (f"What is {a} times {b}?",
            f"{a} times {b} = {ans}")

def gen_division(rng):
    """Division that produces clean results."""
    b = rng.randint(2, 100)
    ans = rng.randint(1, 1000)
    a = b * ans  # ensure clean division
    return (f"What is {a} divided by {b}?",
            f"{a} divided by {b} = {ans}")

def gen_comparison(rng):
    """Number comparison."""
    use_decimal = rng.random() < 0.3
    if use_decimal:
        a = round(rng.uniform(0.1, 99999.99), 2)
        b = round(rng.uniform(0.1, 99999.99), 2)
    else:
        a = rng.randint(1, 99999)
        b = rng.randint(1, 99999)
    while a == b:
        b = rng.randint(1, 99999)
    if a > b:
        return (f"Which is larger, {a} or {b}?",
                f"{a} is larger than {b}")
    else:
        return (f"Which is larger, {a} or {b}?",
                f"{b} is larger than {a}")

def gen_percentage(rng):
    """Percentage calculation."""
    pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100])
    base = rng.randint(10, 10000)
    ans = round(pct / 100.0 * base, 2)
    return (f"What is {pct}% of {base}?",
            f"{pct}% of {base} = {ans}")

def gen_multistep(rng):
    """Two-step arithmetic."""
    a = rng.randint(1, 9999)
    b = rng.randint(1, 9999)
    c = rng.randint(1, 9999)
    step1 = a + b
    result = step1 - c
    return (f"If you have {a} and add {b}, then subtract {c}, what is the result?",
            f"{a} + {b} = {step1}, {step1} - {c} = {result}. The result is {result}")

def gen_add_three(rng):
    """Add three numbers."""
    a = rng.randint(1, 9999)
    b = rng.randint(1, 9999)
    c = rng.randint(1, 9999)
    ans = a + b + c
    return (f"What is {a} + {b} + {c}?",
            f"{a} + {b} + {c} = {ans}")

def gen_difference(rng):
    """Find the difference between two numbers."""
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    diff = abs(a - b)
    return (f"What is the difference between {a} and {b}?",
            f"The difference between {a} and {b} is {diff}")

def gen_double_half(rng):
    """Double or halve a number."""
    if rng.random() < 0.5:
        a = rng.randint(1, 50000)
        return (f"What is double {a}?",
                f"Double {a} = {a * 2}")
    else:
        a = rng.randint(1, 50000) * 2  # ensure even
        return (f"What is half of {a}?",
                f"Half of {a} = {a // 2}")


GENERATORS = [
    (gen_addition, 0.20),
    (gen_subtraction, 0.15),
    (gen_multiplication, 0.12),
    (gen_division, 0.08),
    (gen_comparison, 0.10),
    (gen_percentage, 0.10),
    (gen_multistep, 0.10),
    (gen_add_three, 0.05),
    (gen_difference, 0.05),
    (gen_double_half, 0.05),
]


def generate_problems(n, seed=42):
    """Generate n arithmetic problems as (user_text, assistant_text) pairs."""
    rng = random.Random(seed)
    gen_fns = [g[0] for g in GENERATORS]
    gen_weights = [g[1] for g in GENERATORS]

    problems = []
    for _ in range(n):
        fn = rng.choices(gen_fns, weights=gen_weights, k=1)[0]
        user, assistant = fn(rng)
        problems.append({'user': user, 'assistant': assistant})
    return problems


# =============================================================================
# Tokenization — base vs adapted
# =============================================================================

def process_content_with_numbers(text):
    """Replace numbers with <NUM> tokens. Returns (ids, nums) without EOT."""
    tokenizer = _get_tokenizer()
    ids = []
    nums = []
    last_end = 0
    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            seg_ids = tokenizer.encode(text[last_end:start],
                                       add_special_tokens=False)
            ids.extend(seg_ids)
            nums.extend([0.0] * len(seg_ids))
        try:
            value = float(match.group())
        except ValueError:
            seg_ids = tokenizer.encode(match.group(),
                                       add_special_tokens=False)
            ids.extend(seg_ids)
            nums.extend([0.0] * len(seg_ids))
            last_end = end
            continue
        ids.append(NUM_TOKEN_ID)
        nums.append(value)
        last_end = end
    if last_end < len(text):
        seg_ids = tokenizer.encode(text[last_end:],
                                   add_special_tokens=False)
        ids.extend(seg_ids)
        nums.extend([0.0] * len(seg_ids))
    return ids, nums


def tokenize_base(problem):
    """Tokenize as plain text, no <NUM>."""
    tokenizer = _get_tokenizer()
    text = f"User: {problem['user']}\nAssistant: {problem['assistant']}"
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids.append(EOT_TOKEN_ID)
    nums = [0.0] * len(ids)
    return ids, nums


def tokenize_adapted(problem):
    """Tokenize with <NUM> in user, plain text in assistant."""
    tokenizer = _get_tokenizer()
    ids = []
    nums = []

    # User turn — numbers replaced with <NUM>
    prefix = "User: "
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids.extend(prefix_ids)
    nums.extend([0.0] * len(prefix_ids))

    c_ids, c_nums = process_content_with_numbers(problem['user'])
    ids.extend(c_ids)
    nums.extend(c_nums)

    # Assistant turn — plain text
    prefix = "\nAssistant: "
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids.extend(prefix_ids)
    nums.extend([0.0] * len(prefix_ids))

    c_ids = tokenizer.encode(problem['assistant'], add_special_tokens=False)
    ids.extend(c_ids)
    nums.extend([0.0] * len(c_ids))

    ids.append(EOT_TOKEN_ID)
    nums.append(0.0)
    return ids, nums


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic arithmetic data for LoRA A/B')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_train', type=int, default=50000)
    parser.add_argument('--n_val', type=int, default=3000)
    parser.add_argument('--n_test', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base_dir = os.path.join(args.out_dir, 'base')
    adapted_dir = os.path.join(args.out_dir, 'adapted')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(adapted_dir, exist_ok=True)

    total = args.n_train + args.n_val + args.n_test
    print(f"Generating {total} problems (train={args.n_train}, "
          f"val={args.n_val}, test={args.n_test})")
    all_problems = generate_problems(total, seed=args.seed)

    splits = {
        'train': all_problems[:args.n_train],
        'val': all_problems[args.n_train:args.n_train + args.n_val],
        'test': all_problems[args.n_train + args.n_val:],
    }

    for split_name, problems in splits.items():
        all_base_ids, all_base_nums = [], []
        all_adapted_ids, all_adapted_nums = [], []

        for p in problems:
            b_ids, b_nums = tokenize_base(p)
            all_base_ids.extend(b_ids)
            all_base_nums.extend(b_nums)

            a_ids, a_nums = tokenize_adapted(p)
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

        print(f"  {split_name}: {len(problems)} problems")
        print(f"    base:    {len(all_base_ids):>10,} tokens, "
              f"{n_base_num:>6} <NUM>")
        print(f"    adapted: {len(all_adapted_ids):>10,} tokens, "
              f"{n_adapted_num:>6} <NUM>")

    # Save test examples as JSON for benchmark_arithmetic.py compatibility
    test_json = []
    for p in splits['test']:
        test_json.append({
            'messages': [
                {'role': 'user', 'content': p['user']},
                {'role': 'assistant', 'content': p['assistant']},
            ],
        })
    for d in [base_dir, adapted_dir]:
        with open(os.path.join(d, 'test_examples.json'), 'w') as f:
            json.dump(test_json, f, indent=2)

    print(f"\nDone! Data saved to:")
    print(f"  base:    {base_dir}")
    print(f"  adapted: {adapted_dir}")
    print(f"  test_examples.json: {len(test_json)} examples")


if __name__ == '__main__':
    main()
