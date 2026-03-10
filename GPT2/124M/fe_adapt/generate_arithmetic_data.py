"""Generate a comprehensive arithmetic benchmark with ID and OOD numbers.

Number distribution splits:
  - ID (in-distribution):    matching training range [1, 99999]
  - OOD-small:               negatives [-999, -1] and tiny decimals [0.001, 0.999]
  - OOD-large:               big numbers [100000, 1000000000]

Problem categories:
  Numerical:  addition, subtraction, multiplication, division, sum3, chained
  Reasoning:  comparison, percentage, difference, ordering

Each problem records its category, distribution, and expected answer.

Usage:
  python generate_arithmetic_data.py --out_path /path/to/bench.json
"""

import os
import json
import random
import argparse
from collections import Counter


# =============================================================================
# Number generators per distribution
# =============================================================================

def rand_id(rng):
    """In-distribution: [1, 99999] — matches training range."""
    return rng.randint(1, 99999)

def rand_ood_small(rng):
    """OOD-small: negatives or tiny decimals."""
    if rng.random() < 0.5:
        return -rng.randint(1, 999)
    else:
        return round(rng.uniform(0.001, 0.999), 3)

def rand_ood_large(rng):
    """OOD-large: [100000, 1000000000]."""
    return rng.randint(100000, 1000000000)

DIST_GENERATORS = {
    'id':        rand_id,
    'ood_small': rand_ood_small,
    'ood_large': rand_ood_large,
}


# =============================================================================
# Problem generators
# =============================================================================

def gen_addition(rng, rand_fn):
    a, b = rand_fn(rng), rand_fn(rng)
    ans = a + b
    q = f"What is {a} + {b}?"
    a_text = f"{a} + {b} = {ans}"
    return q, a_text, float(ans), 'numerical'

def gen_subtraction(rng, rand_fn):
    a, b = rand_fn(rng), rand_fn(rng)
    ans = a - b
    q = f"What is {a} - {b}?"
    a_text = f"{a} - {b} = {ans}"
    return q, a_text, float(ans), 'numerical'

def gen_multiplication(rng, rand_fn):
    # Cap to avoid astronomical products
    a, b = rand_fn(rng), rand_fn(rng)
    if abs(a) > 9999:
        a = rng.randint(2, 999) if a > 0 else -rng.randint(2, 999)
    if abs(b) > 9999:
        b = rng.randint(2, 999) if b > 0 else -rng.randint(2, 999)
    ans = a * b
    q = f"What is {a} times {b}?"
    a_text = f"{a} times {b} = {ans}"
    return q, a_text, float(ans), 'numerical'

def gen_division(rng, rand_fn):
    b = rng.randint(2, 200)
    quotient = rand_fn(rng)
    if isinstance(quotient, float) and abs(quotient) < 1:
        quotient = rng.randint(1, 100)
    a = b * quotient
    q = f"What is {a} divided by {b}?"
    a_text = f"{a} divided by {b} = {quotient}"
    return q, a_text, float(quotient), 'numerical'

def gen_sum3(rng, rand_fn):
    a, b, c = rand_fn(rng), rand_fn(rng), rand_fn(rng)
    ans = a + b + c
    q = f"What is {a} + {b} + {c}?"
    a_text = f"{a} + {b} + {c} = {ans}"
    return q, a_text, float(ans), 'numerical'

def gen_chained(rng, rand_fn):
    a, b, c = rand_fn(rng), rand_fn(rng), rand_fn(rng)
    step1 = a + b
    result = step1 - c
    q = f"If you start with {a}, add {b}, then subtract {c}, what do you get?"
    a_text = f"{a} + {b} = {step1}, {step1} - {c} = {result}. The answer is {result}"
    return q, a_text, float(result), 'numerical'

def gen_comparison(rng, rand_fn):
    a, b = rand_fn(rng), rand_fn(rng)
    while a == b:
        b = rand_fn(rng)
    if a > b:
        larger, smaller = a, b
    else:
        larger, smaller = b, a
    q = f"Which is larger, {a} or {b}?"
    a_text = f"{larger} is larger than {smaller}"
    # For comparison, expected is the larger number
    return q, a_text, float(larger), 'reasoning'

def gen_percentage(rng, rand_fn):
    pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100])
    base = abs(rand_fn(rng))
    if base == 0:
        base = rng.randint(10, 1000)
    ans = round(pct / 100.0 * base, 2)
    q = f"What is {pct}% of {base}?"
    a_text = f"{pct}% of {base} = {ans}"
    return q, a_text, float(ans), 'reasoning'

def gen_difference(rng, rand_fn):
    a, b = rand_fn(rng), rand_fn(rng)
    diff = abs(a - b)
    q = f"What is the difference between {a} and {b}?"
    a_text = f"The difference between {a} and {b} is {diff}"
    return q, a_text, float(diff), 'reasoning'

def gen_ordering(rng, rand_fn):
    a, b, c = rand_fn(rng), rand_fn(rng), rand_fn(rng)
    while a == b or b == c or a == c:
        c = rand_fn(rng)
    sorted_nums = sorted([a, b, c])
    q = f"Sort these numbers from smallest to largest: {a}, {b}, {c}"
    a_text = f"{sorted_nums[0]}, {sorted_nums[1]}, {sorted_nums[2]}"
    # expected = smallest (first number in sorted order)
    return q, a_text, float(sorted_nums[0]), 'reasoning'


GENERATORS = {
    'addition':       gen_addition,
    'subtraction':    gen_subtraction,
    'multiplication': gen_multiplication,
    'division':       gen_division,
    'sum3':           gen_sum3,
    'chained':        gen_chained,
    'comparison':     gen_comparison,
    'percentage':     gen_percentage,
    'difference':     gen_difference,
    'ordering':       gen_ordering,
}

# Weighted distribution — more numerical, as requested
CATEGORY_WEIGHTS = {
    'addition':       0.18,
    'subtraction':    0.14,
    'multiplication': 0.10,
    'division':       0.06,
    'sum3':           0.10,
    'chained':        0.10,
    'comparison':     0.08,
    'percentage':     0.08,
    'difference':     0.08,
    'ordering':       0.08,
}


# =============================================================================
# Few-shot examples (used by benchmark_arithmetic.py)
# =============================================================================

FEW_SHOT_EXAMPLES = {
    'addition':       [
        {'user': 'What is 12 + 5?', 'assistant': '12 + 5 = 17'},
        {'user': 'What is 248 + 193?', 'assistant': '248 + 193 = 441'},
    ],
    'subtraction':    [
        {'user': 'What is 50 - 23?', 'assistant': '50 - 23 = 27'},
        {'user': 'What is 1024 - 378?', 'assistant': '1024 - 378 = 646'},
    ],
    'multiplication': [
        {'user': 'What is 7 times 8?', 'assistant': '7 times 8 = 56'},
        {'user': 'What is 23 times 17?', 'assistant': '23 times 17 = 391'},
    ],
    'division':       [
        {'user': 'What is 56 divided by 8?', 'assistant': '56 divided by 8 = 7'},
        {'user': 'What is 144 divided by 12?', 'assistant': '144 divided by 12 = 12'},
    ],
    'sum3':           [
        {'user': 'What is 10 + 20 + 30?', 'assistant': '10 + 20 + 30 = 60'},
        {'user': 'What is 125 + 250 + 375?', 'assistant': '125 + 250 + 375 = 750'},
    ],
    'chained':        [
        {'user': 'If you start with 30, add 15, then subtract 10, what do you get?',
         'assistant': '30 + 15 = 45, 45 - 10 = 35. The answer is 35'},
        {'user': 'If you start with 100, add 250, then subtract 75, what do you get?',
         'assistant': '100 + 250 = 350, 350 - 75 = 275. The answer is 275'},
    ],
    'comparison':     [
        {'user': 'Which is larger, 15 or 9?', 'assistant': '15 is larger than 9'},
        {'user': 'Which is larger, 3200 or 4100?', 'assistant': '4100 is larger than 3200'},
    ],
    'percentage':     [
        {'user': 'What is 10% of 200?', 'assistant': '10% of 200 = 20.0'},
        {'user': 'What is 25% of 480?', 'assistant': '25% of 480 = 120.0'},
    ],
    'difference':     [
        {'user': 'What is the difference between 50 and 30?',
         'assistant': 'The difference between 50 and 30 is 20'},
        {'user': 'What is the difference between 1000 and 750?',
         'assistant': 'The difference between 1000 and 750 is 250'},
    ],
    'ordering':       [
        {'user': 'Sort these numbers from smallest to largest: 5, 2, 8',
         'assistant': '2, 5, 8'},
        {'user': 'Sort these numbers from smallest to largest: 300, 100, 200',
         'assistant': '100, 200, 300'},
    ],
}


# =============================================================================
# Generation
# =============================================================================

def generate_benchmark(n_problems=1500, seed=42):
    rng = random.Random(seed)
    problems = []

    cats = list(CATEGORY_WEIGHTS.keys())
    weights = [CATEGORY_WEIGHTS[c] for c in cats]
    dists = list(DIST_GENERATORS.keys())

    # Distribute evenly across distributions
    per_dist = n_problems // len(dists)

    for dist in dists:
        rand_fn = DIST_GENERATORS[dist]
        for _ in range(per_dist):
            cat = rng.choices(cats, weights=weights, k=1)[0]
            gen_fn = GENERATORS[cat]
            try:
                user, assistant, expected, cat_type = gen_fn(rng, rand_fn)
            except (ValueError, ZeroDivisionError):
                # Retry with a different generator on error
                user, assistant, expected, cat_type = gen_addition(rng, rand_fn)
                cat = 'addition'
                cat_type = 'numerical'

            problems.append({
                'messages': [
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant},
                ],
                'category': cat,
                'category_type': cat_type,
                'distribution': dist,
                'expected_answer': expected,
            })

    rng.shuffle(problems)
    return problems


def main():
    parser = argparse.ArgumentParser(
        description='Generate arithmetic benchmark with ID/OOD splits')
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--n_problems', type=int, default=1500,
                        help='Total problems (split equally across 3 distributions)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    problems = generate_benchmark(n_problems=args.n_problems, seed=args.seed)

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(problems, f, indent=2)

    # Also save few-shot examples alongside
    fewshot_path = args.out_path.replace('.json', '_fewshot.json')
    with open(fewshot_path, 'w') as f:
        json.dump(FEW_SHOT_EXAMPLES, f, indent=2)

    # Summary
    print(f"Generated {len(problems)} problems → {args.out_path}")
    print(f"Few-shot examples → {fewshot_path}")

    print(f"\nBy distribution:")
    dist_counts = Counter(p['distribution'] for p in problems)
    for d, c in sorted(dist_counts.items()):
        print(f"  {d:<12} {c:>5}")

    print(f"\nBy category:")
    cat_counts = Counter(p['category'] for p in problems)
    for cat, c in sorted(cat_counts.items()):
        print(f"  {cat:<20} {c:>5}")

    print(f"\nBy category type:")
    type_counts = Counter(p['category_type'] for p in problems)
    for t, c in sorted(type_counts.items()):
        print(f"  {t:<12} {c:>5}")

    # Show distribution × category matrix
    print(f"\nDistribution × Category:")
    print(f"  {'Category':<20}", end='')
    for d in sorted(dist_counts.keys()):
        print(f" {d:>10}", end='')
    print()
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat:<20}", end='')
        for d in sorted(dist_counts.keys()):
            cnt = sum(1 for p in problems
                      if p['category'] == cat and p['distribution'] == d)
            print(f" {cnt:>10}", end='')
        print()


if __name__ == '__main__':
    main()
