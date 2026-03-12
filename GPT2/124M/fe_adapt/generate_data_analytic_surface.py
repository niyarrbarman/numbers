"""Generate synthetic data for Stage 1 analytic number integration.

Produces binary files with <NUM> in BOTH input (user) and output (assistant):
  - {split}.bin          : uint16 token IDs (with NUM_TOKEN_ID at number positions)
  - {split}_nums.bin     : float32 values (float at NUM positions, 0.0 elsewhere)
  - {split}_components.bin : uint8 (N, 34) pre-computed [sign_class, exp_class, d0..d31]

Data mixture:
  70% synthetic numeric expressions (arithmetic, comparison, min/max, etc.)
  20% natural-language quantity sentences
  10% plain non-math text replay (dummy sentences)

Usage:
  python generate_data_analytic.py --out_dir /path/to/data --n_train 100000
"""

import os
import sys
import json
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from prepare import (_get_tokenizer, NUM_TOKEN_ID, EOT_TOKEN_ID,
                      NUMBER_PATTERN)
from num_analytic import AnalyticNumberCodec
from numeric_surface import surface_components_from_value, surface_components_to_row


# =============================================================================
# Problem generators — 70% synthetic numeric
# =============================================================================

def gen_addition(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    ans = a + b
    return (f"What is {a} + {b}?", f"{ans}")


def gen_subtraction(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    if a < b:
        a, b = b, a
    return (f"What is {a} - {b}?", f"{a - b}")


def gen_multiplication(rng):
    a = rng.randint(2, 999)
    b = rng.randint(2, 999)
    return (f"What is {a} times {b}?", f"{a * b}")


def gen_division(rng):
    b = rng.randint(2, 100)
    ans = rng.randint(1, 1000)
    a = b * ans
    return (f"What is {a} divided by {b}?", f"{ans}")


def gen_comparison(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    while a == b:
        b = rng.randint(1, 99999)
    if a > b:
        return (f"Which is larger, {a} or {b}?", f"{a}")
    else:
        return (f"Which is larger, {a} or {b}?", f"{b}")


def gen_min_max(rng):
    nums = [rng.randint(1, 99999) for _ in range(rng.randint(3, 5))]
    nums_str = ", ".join(str(n) for n in nums)
    op = rng.choice(["minimum", "maximum"])
    ans = min(nums) if op == "minimum" else max(nums)
    return (f"What is the {op} of {nums_str}?", f"{ans}")


def gen_absolute(rng):
    a = rng.randint(-99999, 99999)
    return (f"What is the absolute value of {a}?", f"{abs(a)}")


def gen_percentage(rng):
    pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100])
    base = rng.randint(10, 10000)
    ans = round(pct / 100.0 * base, 2)
    return (f"What is {pct}% of {base}?", f"{ans}")


def gen_multistep(rng):
    a = rng.randint(1, 9999)
    b = rng.randint(1, 9999)
    c = rng.randint(1, 9999)
    result = a + b - c
    return (f"Compute {a} + {b} - {c}.", f"{result}")


def gen_add_three(rng):
    a = rng.randint(1, 9999)
    b = rng.randint(1, 9999)
    c = rng.randint(1, 9999)
    return (f"What is {a} + {b} + {c}?", f"{a + b + c}")


def gen_difference(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    return (f"What is the difference between {a} and {b}?", f"{abs(a - b)}")


def gen_double_half(rng):
    if rng.random() < 0.5:
        a = rng.randint(1, 50000)
        return (f"What is double {a}?", f"{a * 2}")
    else:
        a = rng.randint(1, 50000) * 2
        return (f"What is half of {a}?", f"{a // 2}")


def gen_rounding(rng):
    decimals = rng.randint(1, 4)
    val = round(rng.uniform(-9999, 9999), rng.randint(3, 8))
    ans = round(val, decimals)
    return (f"Round {val} to {decimals} decimal places.", f"{ans}")


def gen_decimal_shift(rng):
    a = rng.randint(1, 99999)
    factor = rng.choice([10, 100, 1000])
    op = rng.choice(["multiply", "divide"])
    if op == "multiply":
        ans = a * factor
        return (f"What is {a} times {factor}?", f"{ans}")
    else:
        ans = a / factor
        return (f"What is {a} divided by {factor}?", f"{ans}")


def gen_sorting(rng):
    n = rng.randint(3, 5)
    nums = [rng.randint(1, 99999) for _ in range(n)]
    direction = rng.choice(["ascending", "descending"])
    sorted_nums = sorted(nums, reverse=(direction == "descending"))
    nums_str = ", ".join(str(x) for x in nums)
    ans_str = ", ".join(str(x) for x in sorted_nums)
    return (f"Sort {nums_str} in {direction} order.", f"{ans_str}")


NUMERIC_GENERATORS = [
    (gen_addition, 0.15),
    (gen_subtraction, 0.12),
    (gen_multiplication, 0.10),
    (gen_division, 0.08),
    (gen_comparison, 0.08),
    (gen_min_max, 0.06),
    (gen_absolute, 0.05),
    (gen_percentage, 0.08),
    (gen_multistep, 0.08),
    (gen_add_three, 0.05),
    (gen_difference, 0.05),
    (gen_double_half, 0.04),
    (gen_rounding, 0.03),
    (gen_decimal_shift, 0.03),
]
# gen_sorting excluded from <NUM> replacement (multi-number answer is complex)
# We include it but the answer won't have <NUM> (stays as text)

# =============================================================================
# 20% natural-language quantity sentences
# =============================================================================

NL_TEMPLATES = [
    ("The temperature in Paris is {a} degrees. What is this in fewer words?", "{a} degrees"),
    ("A company reported revenue of {a} million dollars last quarter.", "{a} million"),
    ("The population of the city is {a}.", "{a}"),
    ("The distance from A to B is {a} kilometers.", "{a} km"),
    ("The product costs {a} dollars. What is the price?", "{a}"),
    ("There are {a} students in the class.", "{a}"),
    ("The building is {a} meters tall.", "{a} meters"),
    ("She scored {a} points in the exam.", "{a}"),
    ("The car can travel {a} miles on a full tank.", "{a} miles"),
    ("The weight of the package is {a} kilograms.", "{a} kg"),
]


def gen_natural_language(rng):
    tmpl_q, tmpl_a = rng.choice(NL_TEMPLATES)
    a = rng.randint(1, 99999)
    return (tmpl_q.format(a=a), tmpl_a.format(a=a))


# =============================================================================
# 10% plain text replay (no numbers)
# =============================================================================

PLAIN_TEXTS = [
    ("What is the capital of France?", "Paris"),
    ("Name a primary color.", "Red"),
    ("What day comes after Monday?", "Tuesday"),
    ("Is water wet?", "Yes, water is wet."),
    ("Name a continent.", "Africa"),
    ("What is the opposite of hot?", "Cold"),
    ("What sound does a cat make?", "Meow"),
    ("Name a season.", "Summer"),
    ("What color is the sky?", "Blue"),
    ("How many sides does a triangle have?", "Three"),
    ("What is the largest ocean?", "The Pacific Ocean"),
    ("Name a musical instrument.", "Guitar"),
    ("What language is spoken in Brazil?", "Portuguese"),
    ("What planet is closest to the sun?", "Mercury"),
    ("Name a type of fruit.", "Apple"),
    ("What is the freezing point of water?", "Zero degrees Celsius"),
    ("How many days are in a week?", "Seven"),
    ("What is the opposite of up?", "Down"),
    ("Name a precious metal.", "Gold"),
    ("What shape has four equal sides?", "A square"),
]


def gen_plain_text(rng):
    return rng.choice(PLAIN_TEXTS)


# =============================================================================
# Tokenization — <NUM> in both input and output
# =============================================================================

def process_text_with_numbers(text, return_num_texts=False):
    """Replace numbers with <NUM> tokens.

    Returns:
        (ids, nums) without EOT by default
        (ids, nums, num_texts) when return_num_texts=True, where num_texts is
        parallel to ids and stores the original matched numeric string at each
        <NUM> position.
    """
    tokenizer = _get_tokenizer()
    ids = []
    nums = []
    num_texts = [] if return_num_texts else None
    last_end = 0
    for match in NUMBER_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            seg_ids = tokenizer.encode(text[last_end:start],
                                       add_special_tokens=False)
            ids.extend(seg_ids)
            nums.extend([0.0] * len(seg_ids))
            if return_num_texts:
                num_texts.extend([None] * len(seg_ids))
        try:
            value = float(match.group())
        except ValueError:
            seg_ids = tokenizer.encode(match.group(),
                                       add_special_tokens=False)
            ids.extend(seg_ids)
            nums.extend([0.0] * len(seg_ids))
            if return_num_texts:
                num_texts.extend([None] * len(seg_ids))
            last_end = end
            continue
        ids.append(NUM_TOKEN_ID)
        nums.append(value)
        if return_num_texts:
            num_texts.append(match.group())
        last_end = end
    if last_end < len(text):
        seg_ids = tokenizer.encode(text[last_end:],
                                   add_special_tokens=False)
        ids.extend(seg_ids)
        nums.extend([0.0] * len(seg_ids))
        if return_num_texts:
            num_texts.extend([None] * len(seg_ids))
    if return_num_texts:
        return ids, nums, num_texts
    return ids, nums


def tokenize_problem(problem, use_num=True, return_num_texts=False):
    """Tokenize user/assistant pair with <NUM> in BOTH turns.

    Args:
        problem: dict with 'user' and 'assistant' keys
        use_num: if True, replace numbers with <NUM>; if False, plain text
        return_num_texts: if True, return a third list preserving the original
            matched numeric strings at <NUM> positions

    Returns:
        ids: list[int], nums: list[float]
        optionally num_texts: list[str | None]
    """
    tokenizer = _get_tokenizer()
    ids = []
    nums = []
    num_texts = [] if return_num_texts else None

    # User turn
    prefix = "User: "
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids.extend(prefix_ids)
    nums.extend([0.0] * len(prefix_ids))
    if return_num_texts:
        num_texts.extend([None] * len(prefix_ids))

    if use_num:
        c_ids, c_nums, c_texts = process_text_with_numbers(
            problem['user'], return_num_texts=True)
    else:
        c_ids = tokenizer.encode(problem['user'], add_special_tokens=False)
        c_nums = [0.0] * len(c_ids)
        c_texts = [None] * len(c_ids)
    ids.extend(c_ids)
    nums.extend(c_nums)
    if return_num_texts:
        num_texts.extend(c_texts)

    # Separator
    sep = "\nAssistant: "
    sep_ids = tokenizer.encode(sep, add_special_tokens=False)
    ids.extend(sep_ids)
    nums.extend([0.0] * len(sep_ids))
    if return_num_texts:
        num_texts.extend([None] * len(sep_ids))

    # Assistant turn — <NUM> in output too
    if use_num:
        c_ids, c_nums, c_texts = process_text_with_numbers(
            problem['assistant'], return_num_texts=True)
    else:
        c_ids = tokenizer.encode(problem['assistant'], add_special_tokens=False)
        c_nums = [0.0] * len(c_ids)
        c_texts = [None] * len(c_ids)
    ids.extend(c_ids)
    nums.extend(c_nums)
    if return_num_texts:
        num_texts.extend(c_texts)

    # EOT
    ids.append(EOT_TOKEN_ID)
    nums.append(0.0)
    if return_num_texts:
        num_texts.append(None)

    if return_num_texts:
        return ids, nums, num_texts
    return ids, nums


# =============================================================================
# Component encoding for numeric targets
# =============================================================================

def encode_num_components(codec, value):
    """Encode a number value into [sign_class, exp_class, d0..d31] = 34 uint8.

    sign_class: 0 = positive, 1 = negative
    exp_class: exponent - exp_min (so exp=-32 → class 0, exp=32 → class 64)
    d0..d31: digit values 0..9
    """
    comps = codec.to_components(value)
    sign_class = 0 if comps.sign >= 0 else 1
    exp_class = comps.exponent - codec.exp_min
    exp_class = max(0, min(codec.exp_max - codec.exp_min, exp_class))
    digits = list(comps.digits[:codec.K])
    # Pad if needed
    while len(digits) < codec.K:
        digits.append(0)

    return [sign_class, exp_class] + digits


def build_components_array(ids, num_texts, codec):
    """Build (N, 34) uint8 component array parallel to token IDs.

    Components are meaningful only at NUM_TOKEN_ID positions.
    """
    n = len(ids)
    components = np.zeros((n, 2 + codec.K), dtype=np.uint8)
    for i in range(n):
        if ids[i] != NUM_TOKEN_ID:
            continue
        if num_texts[i] is None:
            raise ValueError(f"Missing numeric text for <NUM> at position {i}")
        components[i] = encode_num_components(codec, num_texts[i])
    return components


def build_surface_array(ids, num_texts, max_digits, scale_min, scale_max):
    """Build (N, 3 + max_digits) uint8 surface labels parallel to token IDs."""
    n = len(ids)
    surface = np.zeros((n, 3 + max_digits), dtype=np.uint8)
    for i in range(n):
        if ids[i] != NUM_TOKEN_ID:
            continue
        if num_texts[i] is None:
            raise ValueError(f"Missing numeric text for <NUM> at position {i}")
        comps = surface_components_from_value(
            num_texts[i],
            max_digits=max_digits,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        surface[i] = surface_components_to_row(
            comps,
            max_digits=max_digits,
            scale_min=scale_min,
            scale_max=scale_max,
        )
    return surface


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data for Stage 1 analytic number integration')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_train', type=int, default=100000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analytic_K', type=int, default=32)
    parser.add_argument('--analytic_exp_min', type=int, default=-32)
    parser.add_argument('--analytic_exp_max', type=int, default=32)
    parser.add_argument('--surface_max_digits', type=int, default=32)
    parser.add_argument('--surface_scale_min', type=int, default=0)
    parser.add_argument('--surface_scale_max', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    codec = AnalyticNumberCodec(
        K=args.analytic_K,
        exp_min=args.analytic_exp_min,
        exp_max=args.analytic_exp_max,
    )
    print(f"AnalyticNumberCodec (K={codec.K}, exp=[{codec.exp_min},{codec.exp_max}], "
          f"dim={codec.total_dim})")

    rng = random.Random(args.seed)

    # Build generators with weights
    numeric_fns = [g[0] for g in NUMERIC_GENERATORS]
    numeric_weights = [g[1] for g in NUMERIC_GENERATORS]

    total = args.n_train + args.n_val + args.n_test
    print(f"Generating {total} problems (train={args.n_train}, "
          f"val={args.n_val}, test={args.n_test})")

    # Generate problems with mixture
    all_problems = []
    all_categories = []  # 'numeric', 'nl', 'plain'
    for _ in range(total):
        r = rng.random()
        if r < 0.70:
            fn = rng.choices(numeric_fns, weights=numeric_weights, k=1)[0]
            user, assistant = fn(rng)
            all_problems.append({'user': user, 'assistant': assistant})
            all_categories.append('numeric')
        elif r < 0.90:
            user, assistant = gen_natural_language(rng)
            all_problems.append({'user': user, 'assistant': assistant})
            all_categories.append('nl')
        else:
            user, assistant = gen_plain_text(rng)
            all_problems.append({'user': user, 'assistant': assistant})
            all_categories.append('plain')

    splits = {
        'train': (all_problems[:args.n_train], all_categories[:args.n_train]),
        'val': (all_problems[args.n_train:args.n_train + args.n_val],
                all_categories[args.n_train:args.n_train + args.n_val]),
        'test': (all_problems[args.n_train + args.n_val:],
                 all_categories[args.n_train + args.n_val:]),
    }

    for split_name, (problems, categories) in splits.items():
        all_ids = []
        all_nums = []
        all_components = []
        all_surface = []

        cat_counts = {'numeric': 0, 'nl': 0, 'plain': 0}
        for p, cat in zip(problems, categories):
            cat_counts[cat] += 1
            # Plain text: no <NUM> replacement
            use_num = (cat != 'plain')
            p_ids, p_nums, p_num_texts = tokenize_problem(
                p, use_num=use_num, return_num_texts=True)

            # Build components
            p_components = build_components_array(p_ids, p_num_texts, codec)
            p_surface = build_surface_array(
                p_ids,
                p_num_texts,
                max_digits=args.surface_max_digits,
                scale_min=args.surface_scale_min,
                scale_max=args.surface_scale_max,
            )

            all_ids.extend(p_ids)
            all_nums.extend(p_nums)
            all_components.append(p_components)
            all_surface.append(p_surface)

        # Flatten components
        components_array = np.concatenate(all_components, axis=0)
        surface_array = np.concatenate(all_surface, axis=0)

        # Save binary files
        np.array(all_ids, dtype=np.uint16).tofile(
            os.path.join(args.out_dir, f'{split_name}.bin'))
        np.array(all_nums, dtype=np.float32).tofile(
            os.path.join(args.out_dir, f'{split_name}_nums.bin'))
        components_array.tofile(
            os.path.join(args.out_dir, f'{split_name}_components.bin'))
        surface_array.tofile(
            os.path.join(args.out_dir, f'{split_name}_surface.bin'))

        n_num_tokens = sum(1 for x in all_ids if x == NUM_TOKEN_ID)
        print(f"  {split_name}: {len(problems)} problems "
              f"(numeric={cat_counts['numeric']}, nl={cat_counts['nl']}, "
              f"plain={cat_counts['plain']})")
        print(f"    {len(all_ids):>10,} tokens, {n_num_tokens:>6} <NUM>, "
              f"components shape: {components_array.shape}, "
              f"surface shape: {surface_array.shape}")

    # Save test examples as JSON for benchmarking
    test_json = []
    for p in splits['test'][0]:
        test_json.append({
            'messages': [
                {'role': 'user', 'content': p['user']},
                {'role': 'assistant', 'content': p['assistant']},
            ],
        })
    with open(os.path.join(args.out_dir, 'test_examples.json'), 'w') as f:
        json.dump(test_json, f, indent=2)

    # Save codec config for reproducibility
    codec_config = {
        'K': args.analytic_K,
        'exp_min': args.analytic_exp_min,
        'exp_max': args.analytic_exp_max,
        'total_dim': codec.total_dim,
        'surface_max_digits': args.surface_max_digits,
        'surface_scale_min': args.surface_scale_min,
        'surface_scale_max': args.surface_scale_max,
    }
    with open(os.path.join(args.out_dir, 'codec_config.json'), 'w') as f:
        json.dump(codec_config, f, indent=2)

    print(f"\nDone! Data saved to: {args.out_dir}")
    print(f"  test_examples.json: {len(test_json)} examples")
    print(f"  codec_config.json: {json.dumps(codec_config)}")


if __name__ == '__main__':
    main()
