"""Generate a custom arithmetic/numerical benchmark dataset.

Produces a JSON file with 1000 problems across 6 categories:
  1. Addition           (200 problems)
  2. Subtraction        (200 problems)
  3. Multiplication     (150 problems)
  4. Comparison         (150 problems)
  5. Percentage         (150 problems)
  6. Multi-step         (150 problems)

Each problem has 3 difficulty levels (easy/medium/hard) with increasing
number ranges and complexity.

Output format (tulu-style conversations):
  [{"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
    "category": "addition", "difficulty": "easy",
    "expected_answer": 936.0}, ...]

Usage:
  python generate_arithmetic_data.py --out_path /path/to/arithmetic_bench.json
  python generate_arithmetic_data.py --out_path /path/to/test.json --n_problems 20
"""

import os
import json
import random
import argparse


# =============================================================================
# Number range helpers
# =============================================================================

RANGES = {
    'easy':   (1, 99),
    'medium': (100, 9999),
    'hard':   (10000, 99999),
}

def rand_int(difficulty, rng):
    lo, hi = RANGES[difficulty]
    return rng.randint(lo, hi)

def rand_float(difficulty, rng, decimals=2):
    lo, hi = RANGES[difficulty]
    return round(rng.uniform(lo, hi), decimals)

def rand_number(difficulty, rng, allow_float=False):
    """Return int or float depending on difficulty and randomness."""
    if allow_float and difficulty == 'hard':
        return rand_float(difficulty, rng)
    return rand_int(difficulty, rng)


# =============================================================================
# Problem generators (each returns user_prompt, assistant_answer, expected_num)
# =============================================================================

def gen_addition(difficulty, rng):
    a = rand_number(difficulty, rng)
    b = rand_number(difficulty, rng)
    answer = a + b
    prompt = f"What is {a} + {b}?"
    response = f"{a} + {b} = {answer}"
    return prompt, response, float(answer)

def gen_subtraction(difficulty, rng):
    a = rand_number(difficulty, rng)
    b = rand_number(difficulty, rng)
    # Ensure a >= b for non-negative result
    if a < b:
        a, b = b, a
    answer = a - b
    prompt = f"What is {a} - {b}?"
    response = f"{a} - {b} = {answer}"
    return prompt, response, float(answer)

def gen_multiplication(difficulty, rng):
    # Use smaller numbers for multiplication to keep answers reasonable
    mult_ranges = {
        'easy':   (2, 12),
        'medium': (10, 99),
        'hard':   (100, 999),
    }
    lo, hi = mult_ranges[difficulty]
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    answer = a * b
    prompt = f"What is {a} times {b}?"
    response = f"{a} times {b} = {answer}"
    return prompt, response, float(answer)

def gen_comparison(difficulty, rng):
    a = rand_number(difficulty, rng, allow_float=True)
    b = rand_number(difficulty, rng, allow_float=True)
    while a == b:
        b = rand_number(difficulty, rng, allow_float=True)
    if a > b:
        larger = a
        prompt = f"Which is larger, {a} or {b}?"
        response = f"{a} is larger than {b}"
    else:
        larger = b
        prompt = f"Which is larger, {a} or {b}?"
        response = f"{b} is larger than {a}"
    return prompt, response, float(larger)

def gen_percentage(difficulty, rng):
    pct_choices = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    pct = rng.choice(pct_choices)
    base = rand_int(difficulty, rng)
    # Make base divisible by common factors for cleaner answers
    answer = round(pct / 100.0 * base, 2)
    prompt = f"What is {pct}% of {base}?"
    response = f"{pct}% of {base} = {answer}"
    return prompt, response, float(answer)

def gen_multistep(difficulty, rng):
    a = rand_number(difficulty, rng)
    b = rand_number(difficulty, rng)
    c = rand_number(difficulty, rng)
    step1 = a + b
    result = step1 - c
    prompt = (f"If you have {a} and add {b}, then subtract {c}, "
              f"what is the result?")
    response = (f"{a} + {b} = {step1}, {step1} - {c} = {result}. "
                f"The result is {result}")
    return prompt, response, float(result)


GENERATORS = {
    'addition':       gen_addition,
    'subtraction':    gen_subtraction,
    'multiplication': gen_multiplication,
    'comparison':     gen_comparison,
    'percentage':     gen_percentage,
    'multistep':      gen_multistep,
}

# Default distribution
DEFAULT_COUNTS = {
    'addition':       200,
    'subtraction':    200,
    'multiplication': 150,
    'comparison':     150,
    'percentage':     150,
    'multistep':      150,
}

DIFFICULTIES = ['easy', 'medium', 'hard']


# =============================================================================
# Main generation
# =============================================================================

def generate_benchmark(n_problems=None, seed=42):
    """Generate the full benchmark dataset."""
    rng = random.Random(seed)
    problems = []

    if n_problems is not None:
        # Scale counts proportionally
        total_default = sum(DEFAULT_COUNTS.values())
        counts = {k: max(1, int(v * n_problems / total_default))
                  for k, v in DEFAULT_COUNTS.items()}
        # Adjust to hit exact target
        diff = n_problems - sum(counts.values())
        if diff > 0:
            for cat in list(counts.keys())[:diff]:
                counts[cat] += 1
    else:
        counts = DEFAULT_COUNTS.copy()

    for category, count in counts.items():
        gen_fn = GENERATORS[category]
        for i in range(count):
            difficulty = DIFFICULTIES[i % len(DIFFICULTIES)]
            prompt, response, expected = gen_fn(difficulty, rng)

            problems.append({
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response},
                ],
                'category': category,
                'difficulty': difficulty,
                'expected_answer': expected,
            })

    rng.shuffle(problems)
    return problems


def main():
    parser = argparse.ArgumentParser(
        description='Generate custom arithmetic benchmark dataset')
    parser.add_argument('--out_path', required=True,
                        help='Output JSON file path')
    parser.add_argument('--n_problems', type=int, default=None,
                        help='Total number of problems (default: 1000)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    problems = generate_benchmark(n_problems=args.n_problems, seed=args.seed)

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(problems, f, indent=2)

    # Print summary
    from collections import Counter
    cat_counts = Counter(p['category'] for p in problems)
    diff_counts = Counter(p['difficulty'] for p in problems)

    print(f"Generated {len(problems)} arithmetic problems → {args.out_path}")
    print(f"\nBy category:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<20} {cnt:>5}")
    print(f"\nBy difficulty:")
    for diff, cnt in sorted(diff_counts.items()):
        print(f"  {diff:<20} {cnt:>5}")


if __name__ == '__main__':
    main()
