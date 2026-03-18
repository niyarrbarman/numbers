"""Generate synthetic data for Stage 1: adapter + decoder alignment.

Stage 1 trains the numeric encoder→adapter→decoder pathway with the
LLM backbone frozen. Data is simple arithmetic with <NUM> tokens so the
adapter/decoder learn to route numeric information.

Output format (JSONL):
  {
    "prompt": "What is <NUM> + <NUM>?",
    "response": "<NUM>",
    "num_values": [42.0, 58.0, 100.0],
    "num_is_output": [false, false, true]
  }

Usage:
  python3 generate_s1_data.py --out_dir /path/to/s1_data
"""

import os
import json
import random
import argparse
from decimal import Decimal


# =============================================================================
# Problem generators
# =============================================================================

def gen_addition(rng):
    if rng.random() < 0.3:
        a = round(rng.uniform(0.1, 999.99), rng.randint(1, 2))
        b = round(rng.uniform(0.1, 999.99), rng.randint(1, 2))
    else:
        a = rng.randint(1, 99999)
        b = rng.randint(1, 99999)
    ans = a + b
    if isinstance(ans, float):
        ans = round(ans, 2)
    return (f"What is <NUM> + <NUM>?", f"<NUM>",
            [float(a), float(b), float(ans)], [False, False, True])


def gen_subtraction(rng):
    if rng.random() < 0.3:
        a = round(rng.uniform(0.1, 999.99), rng.randint(1, 2))
        b = round(rng.uniform(0.1, a), rng.randint(1, 2))
    else:
        a = rng.randint(1, 99999)
        b = rng.randint(1, a)
    ans = a - b
    if isinstance(ans, float):
        ans = round(ans, 2)
    return (f"What is <NUM> - <NUM>?", f"<NUM>",
            [float(a), float(b), float(ans)], [False, False, True])


def gen_multiplication(rng):
    a = rng.randint(2, 999)
    b = rng.randint(2, 999)
    ans = a * b
    return (f"What is <NUM> times <NUM>?", f"<NUM>",
            [float(a), float(b), float(ans)], [False, False, True])


def gen_division(rng):
    b = rng.randint(2, 100)
    ans = rng.randint(1, 1000)
    a = b * ans
    return (f"What is <NUM> divided by <NUM>?", f"<NUM>",
            [float(a), float(b), float(ans)], [False, False, True])


def gen_percentage(rng):
    pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
    base = rng.randint(10, 10000)
    ans = base * pct / 100
    if ans == int(ans):
        ans = int(ans)
    return (f"What is <NUM>% of <NUM>?", f"<NUM>",
            [float(pct), float(base), float(ans)], [False, False, True])


def gen_comparison(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    while a == b:
        b = rng.randint(1, 99999)
    bigger = max(a, b)
    return (f"Which is larger, <NUM> or <NUM>?", f"<NUM>",
            [float(a), float(b), float(bigger)], [False, False, True])


def gen_multi_step(rng):
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    c = rng.randint(2, 20)
    step1 = a + b
    ans = step1 * c
    return (f"<NUM> + <NUM> = <NUM>. Multiply that by <NUM>.",
            f"<NUM>",
            [float(a), float(b), float(step1), float(c), float(ans)],
            [False, False, True, False, True])


def gen_echo(rng):
    """Identity — model just copies the number through."""
    if rng.random() < 0.5:
        val = rng.randint(0, 99999)
    else:
        val = round(rng.uniform(0.01, 999.99), rng.randint(1, 3))
    return (f"Repeat this number: <NUM>", f"<NUM>",
            [float(val), float(val)], [False, True])


def gen_round_number(rng):
    val = round(rng.uniform(0.001, 999.999), rng.randint(1, 3))
    rounded = round(val)
    return (f"Round <NUM> to the nearest integer.", f"<NUM>",
            [float(val), float(rounded)], [False, True])


GENERATORS = [
    (gen_addition, 0.20),
    (gen_subtraction, 0.15),
    (gen_multiplication, 0.15),
    (gen_division, 0.15),
    (gen_percentage, 0.10),
    (gen_comparison, 0.05),
    (gen_multi_step, 0.10),
    (gen_echo, 0.05),
    (gen_round_number, 0.05),
]


def generate_sample(rng):
    r = rng.random()
    cumulative = 0.0
    for gen_fn, weight in GENERATORS:
        cumulative += weight
        if r < cumulative:
            prompt, response, values, is_output = gen_fn(rng)
            return {
                "prompt": prompt,
                "response": response,
                "num_values": values,
                "num_is_output": is_output,
            }
    # fallback
    return generate_sample(rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(path, "w") as f:
            for _ in range(n):
                sample = generate_sample(rng)
                f.write(json.dumps(sample) + "\n")
        print(f"  {split}: {n} samples -> {path}")

    print("done")


if __name__ == "__main__":
    main()
