"""Generate Stage 2 data from GSM8K + synthetic arithmetic.

Produces TWO directories:
  augmented/  - text with <NUM> tokens + numeric metadata (for NumLM)
  baseline/   - plain text, numbers as tokens (for vanilla Qwen finetune)

The SAME problems appear in both directories — only the representation
differs. Any performance gap is purely from the numeric pathway.

Requires:
  - GSM8K downloaded via download_gsm8k.py

Usage:
  python3 generate_s2_data.py \
    --gsm8k_dir /path/to/gsm8k \
    --out_dir /path/to/s2_data \
    --n_synth 50000
"""

import os
import re
import json
import random
import argparse


NUM_PATTERN = re.compile(r'(?<![.\d])\d+(?:\.\d+)?(?![.\d])')


def find_numbers(text):
    """Find all standalone numbers in text. Returns [(start, end, value), ...]."""
    results = []
    for m in NUM_PATTERN.finditer(text):
        try:
            val = float(m.group())
            results.append((m.start(), m.end(), val))
        except ValueError:
            continue
    return results


def replace_numbers_with_num(text, numbers):
    """Replace number spans with <NUM>, right-to-left to preserve offsets."""
    parts = list(text)
    for start, end, _ in reversed(numbers):
        parts[start:end] = list("<NUM>")
    return "".join(parts)


def strip_gsm8k_annotations(answer_text):
    """Remove <<expression=result>> annotations from GSM8K answers."""
    return re.sub(r'<<[^>]*>>', '', answer_text)


# =============================================================================
# GSM8K processing
# =============================================================================

def process_gsm8k(gsm8k_path):
    """Load and process GSM8K into augmented + baseline format."""
    with open(gsm8k_path) as f:
        data = json.load(f)

    augmented = []
    baseline = []

    for sample in data:
        question = sample["question"]
        raw_answer = sample["answer"]

        # strip annotations and get final answer
        clean_answer = strip_gsm8k_annotations(raw_answer)
        final_answer = clean_answer.split("####")[-1].strip() if "####" in clean_answer else clean_answer.strip()
        solution_steps = clean_answer.split("####")[0].strip() if "####" in clean_answer else clean_answer.strip()

        # baseline: plain text
        baseline.append({
            "prompt": question,
            "response": f"{solution_steps}\nThe answer is {final_answer}.",
        })

        # augmented: replace numbers with <NUM>
        full_prompt = question
        full_response = f"{solution_steps}\nThe answer is {final_answer}."

        prompt_nums = find_numbers(full_prompt)
        response_nums = find_numbers(full_response)

        # replace in prompt and response separately
        aug_prompt = replace_numbers_with_num(full_prompt, prompt_nums)
        aug_response = replace_numbers_with_num(full_response, response_nums)

        num_values = [n[2] for n in prompt_nums] + [n[2] for n in response_nums]
        num_is_output = [False] * len(prompt_nums) + [True] * len(response_nums)

        # skip if no numbers found (rare edge case)
        if not num_values:
            continue

        # skip numbers outside codec range (exp > 32)
        if any(abs(v) > 1e32 for v in num_values if v != 0):
            continue

        augmented.append({
            "prompt": aug_prompt,
            "response": aug_response,
            "num_values": num_values,
            "num_is_output": num_is_output,
        })

    return augmented, baseline


# =============================================================================
# Synthetic arithmetic (same generators as S1 but with response text)
# =============================================================================

def gen_addition(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, 99999)
    ans = a + b
    prompt = f"What is {a} + {b}?"
    response = f"{a} + {b} = {ans}"
    aug_prompt = f"What is <NUM> + <NUM>?"
    aug_response = f"<NUM> + <NUM> = <NUM>"
    return (
        {"prompt": prompt, "response": response},
        {"prompt": aug_prompt, "response": aug_response,
         "num_values": [float(a), float(b), float(a), float(b), float(ans)],
         "num_is_output": [False, False, True, True, True]},
    )


def gen_subtraction(rng):
    a = rng.randint(1, 99999)
    b = rng.randint(1, a)
    ans = a - b
    prompt = f"What is {a} - {b}?"
    response = f"{a} - {b} = {ans}"
    aug_prompt = f"What is <NUM> - <NUM>?"
    aug_response = f"<NUM> - <NUM> = <NUM>"
    return (
        {"prompt": prompt, "response": response},
        {"prompt": aug_prompt, "response": aug_response,
         "num_values": [float(a), float(b), float(a), float(b), float(ans)],
         "num_is_output": [False, False, True, True, True]},
    )


def gen_multiplication(rng):
    a = rng.randint(2, 999)
    b = rng.randint(2, 999)
    ans = a * b
    prompt = f"What is {a} times {b}?"
    response = f"{a} * {b} = {ans}"
    aug_prompt = f"What is <NUM> times <NUM>?"
    aug_response = f"<NUM> * <NUM> = <NUM>"
    return (
        {"prompt": prompt, "response": response},
        {"prompt": aug_prompt, "response": aug_response,
         "num_values": [float(a), float(b), float(a), float(b), float(ans)],
         "num_is_output": [False, False, True, True, True]},
    )


def gen_division(rng):
    b = rng.randint(2, 100)
    ans = rng.randint(1, 1000)
    a = b * ans
    prompt = f"What is {a} divided by {b}?"
    response = f"{a} / {b} = {ans}"
    aug_prompt = f"What is <NUM> divided by <NUM>?"
    aug_response = f"<NUM> / <NUM> = <NUM>"
    return (
        {"prompt": prompt, "response": response},
        {"prompt": aug_prompt, "response": aug_response,
         "num_values": [float(a), float(b), float(a), float(b), float(ans)],
         "num_is_output": [False, False, True, True, True]},
    )


def gen_percentage(rng):
    pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
    base = rng.randint(10, 10000)
    ans = base * pct / 100
    if ans == int(ans):
        ans = int(ans)
    prompt = f"What is {pct}% of {base}?"
    response = f"{pct}% of {base} = {ans}"
    aug_prompt = f"What is <NUM>% of <NUM>?"
    aug_response = f"<NUM>% of <NUM> = <NUM>"
    return (
        {"prompt": prompt, "response": response},
        {"prompt": aug_prompt, "response": aug_response,
         "num_values": [float(pct), float(base), float(pct), float(base), float(ans)],
         "num_is_output": [False, False, True, True, True]},
    )


SYNTH_GENERATORS = [
    (gen_addition, 0.25),
    (gen_subtraction, 0.20),
    (gen_multiplication, 0.20),
    (gen_division, 0.20),
    (gen_percentage, 0.15),
]


def generate_synth_pair(rng):
    r = rng.random()
    cumulative = 0.0
    for gen_fn, weight in SYNTH_GENERATORS:
        cumulative += weight
        if r < cumulative:
            return gen_fn(rng)
    return SYNTH_GENERATORS[0][0](rng)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k_dir", type=str, required=True,
                        help="Dir with train.json/test.json from download_gsm8k.py")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_synth", type=int, default=50000,
                        help="Number of synthetic samples to supplement GSM8K")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    aug_dir = os.path.join(args.out_dir, "augmented")
    base_dir = os.path.join(args.out_dir, "baseline")
    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    # --- process GSM8K train ---
    gsm8k_train_path = os.path.join(args.gsm8k_dir, "train.json")
    print(f"Processing GSM8K train: {gsm8k_train_path}")
    gsm8k_aug, gsm8k_base = process_gsm8k(gsm8k_train_path)
    print(f"  GSM8K: {len(gsm8k_aug)} augmented, {len(gsm8k_base)} baseline")

    # --- generate synthetic ---
    print(f"Generating {args.n_synth} synthetic samples...")
    synth_base = []
    synth_aug = []
    for _ in range(args.n_synth):
        base_sample, aug_sample = generate_synth_pair(rng)
        synth_base.append(base_sample)
        synth_aug.append(aug_sample)

    # --- combine and shuffle ---
    all_aug = gsm8k_aug + synth_aug
    all_base = gsm8k_base + synth_base
    combined = list(zip(all_aug, all_base))
    rng.shuffle(combined)
    all_aug, all_base = zip(*combined)

    # --- split: last 3000 for val, rest for train ---
    val_size = 3000
    train_aug, val_aug = all_aug[:-val_size], all_aug[-val_size:]
    train_base, val_base = all_base[:-val_size], all_base[-val_size:]

    for name, data, out_d in [
        ("train", train_aug, aug_dir), ("val", val_aug, aug_dir),
        ("train", train_base, base_dir), ("val", val_base, base_dir),
    ]:
        path = os.path.join(out_d, f"{name}.jsonl")
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        print(f"  {out_d}/{name}.jsonl: {len(data)} samples")

    # --- process GSM8K test (for benchmark) ---
    gsm8k_test_path = os.path.join(args.gsm8k_dir, "test.json")
    if os.path.exists(gsm8k_test_path):
        print(f"Processing GSM8K test: {gsm8k_test_path}")
        test_aug, test_base = process_gsm8k(gsm8k_test_path)
        for name, data, out_d in [
            ("test", test_aug, aug_dir), ("test", test_base, base_dir),
        ]:
            path = os.path.join(out_d, f"{name}.jsonl")
            with open(path, "w") as f:
                for sample in data:
                    f.write(json.dumps(sample) + "\n")
            print(f"  {out_d}/{name}.jsonl: {len(data)} samples")

    print("done")


if __name__ == "__main__":
    main()
