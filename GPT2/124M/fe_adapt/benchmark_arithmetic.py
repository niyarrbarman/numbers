"""Benchmark base vs adapted LoRA on comprehensive arithmetic problems.

Handles dual tokenization:
  - Base model:    plain text tokenization (no <NUM>)
  - Adapted model: numbers in user prompt → <NUM> token + float values

Breakdowns:
  - By category (addition, subtraction, multiplication, ...)
  - By distribution (id, ood_small, ood_large)
  - By category type (numerical, reasoning)

Uses category-aware answer extraction (fixes comparison evaluation).

Usage:
  python benchmark_arithmetic.py \
    --base_ckpt /path/to/base/ckpt_merged.pt \
    --adapted_ckpt /path/to/adapted/ckpt_merged.pt \
    --data_path /path/to/arithmetic_bench.json \
    --max_samples 1500
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict
from contextlib import nullcontext

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NemotronConfig, Nemotron, NUM_TOKEN_ID
from prepare import _get_tokenizer, EOT_TOKEN_ID, NUMBER_PATTERN

# Import few-shot examples from the data generator
from generate_arithmetic_data import FEW_SHOT_EXAMPLES


# =============================================================================
# Model loading
# =============================================================================

def load_merged_model(ckpt_path, device='cpu'):
    """Load a merged (LoRA-folded) checkpoint."""
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_args = ckpt['model_args']
    model_args['num_emb_checkpoint'] = ''

    nemconf = NemotronConfig(**model_args)
    model = Nemotron(nemconf)

    state_dict = ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    use_adapter = ckpt.get('config', {}).get('use_adapter', True)
    print(f"  Loaded ({len(state_dict)} keys), use_adapter={use_adapter}")
    return model, use_adapter


# =============================================================================
# Tokenization — handles base vs adapted
# =============================================================================

def process_content_with_numbers(text):
    """Replace numbers in text with <NUM> tokens."""
    tokenizer = _get_tokenizer()
    ids, nums = [], []
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


def _tokenize_turn(role, content, use_adapter, is_first, tokenizer):
    """Tokenize a single user/assistant turn."""
    ids, nums = [], []
    prefix = f"{role.capitalize()}: " if is_first else f"\n{role.capitalize()}: "
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids.extend(prefix_ids)
    nums.extend([0.0] * len(prefix_ids))

    if use_adapter and role == 'user':
        c_ids, c_nums = process_content_with_numbers(content)
        ids.extend(c_ids)
        nums.extend(c_nums)
    else:
        c_ids = tokenizer.encode(content, add_special_tokens=False)
        ids.extend(c_ids)
        nums.extend([0.0] * len(c_ids))
    return ids, nums


def format_prompt(messages, use_adapter=False, category=None):
    """Format prompt with few-shot examples for generation."""
    tokenizer = _get_tokenizer()
    ids, nums = [], []
    turn_idx = 0

    # Prepend few-shot examples
    if category and category in FEW_SHOT_EXAMPLES:
        for ex in FEW_SHOT_EXAMPLES[category]:
            t_ids, t_nums = _tokenize_turn(
                'user', ex['user'], use_adapter, turn_idx == 0, tokenizer)
            ids.extend(t_ids)
            nums.extend(t_nums)
            turn_idx += 1

            t_ids, t_nums = _tokenize_turn(
                'assistant', ex['assistant'], False, False, tokenizer)
            ids.extend(t_ids)
            nums.extend(t_nums)
            turn_idx += 1

    # Actual problem
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant' and i == len(messages) - 1:
            prefix = "\nAssistant:"
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            ids.extend(prefix_ids)
            nums.extend([0.0] * len(prefix_ids))
            break

        t_ids, t_nums = _tokenize_turn(
            msg['role'], msg['content'].strip(), use_adapter,
            turn_idx == 0, tokenizer)
        ids.extend(t_ids)
        nums.extend(t_nums)
        turn_idx += 1

    return ids, nums


# =============================================================================
# Category-aware answer extraction
# =============================================================================

def extract_all_numbers(text):
    """Extract all numbers from text."""
    results = []
    for m in NUMBER_PATTERN.finditer(text):
        try:
            results.append(float(m.group()))
        except ValueError:
            pass
    return results


def extract_answer(gen_text, category, expected):
    """Category-aware answer extraction.

    Different categories need to extract the answer differently:
    - comparison:  check if the generated text identifies the correct larger number
                   (first number in "X is larger than Y")
    - ordering:    check if ordering is correct by extracting all numbers
    - everything else: last number in the text is the answer
    """
    nums = extract_all_numbers(gen_text)

    if category == 'comparison':
        # "X is larger than Y" → X is the answer (first number)
        if nums:
            return nums[0]
        return None

    elif category == 'ordering':
        # Check if all sorted numbers are present
        if nums:
            return nums[0]  # first number = smallest
        return None

    else:
        # Standard: last number is the answer
        if nums:
            return nums[-1]
        return None


def check_correct(gen_text, gen_num, expected, category):
    """Category-aware correctness check."""
    if gen_num is None:
        return False

    if category == 'comparison':
        # For comparison: the first number in the output should be the larger
        # Also accept if the text is an exact match to reference
        return abs(gen_num - expected) < max(0.01, abs(expected) * 1e-6)

    elif category == 'ordering':
        # For ordering: check if all numbers appear in sorted order
        nums = extract_all_numbers(gen_text)
        if len(nums) >= 3:
            return nums == sorted(nums)
        return False

    else:
        return abs(gen_num - expected) < max(0.01, abs(expected) * 1e-6)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(model, problems, device, use_adapter=False,
                   max_samples=1500, max_new_tokens=128):
    """Evaluate on arithmetic problems with greedy decoding."""
    tokenizer = _get_tokenizer()
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
               else torch.float16)
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    results = []
    for idx, problem in enumerate(problems[:max_samples]):
        messages = problem['messages']
        category = problem['category']
        distribution = problem.get('distribution', 'id')
        cat_type = problem.get('category_type', 'numerical')
        expected = problem['expected_answer']
        ref_text = messages[-1]['content'].strip()

        # Tokenize prompt with few-shot
        ids, num_vals = format_prompt(messages, use_adapter=use_adapter,
                                      category=category)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        nv = torch.tensor([num_vals], dtype=torch.float32, device=device)
        nm = (x == NUM_TOKEN_ID)

        # Greedy decode
        for _ in range(max_new_tokens):
            T = x.size(1)
            if T > model.config.block_size:
                x_cond = x[:, -model.config.block_size:]
                nv_cond = nv[:, -model.config.block_size:]
                nm_cond = nm[:, -model.config.block_size:]
            else:
                x_cond, nv_cond, nm_cond = x, nv, nm

            with ctx:
                logits, _ = model(x_cond, num_values=nv_cond, num_mask=nm_cond)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == EOT_TOKEN_ID:
                break
            x = torch.cat([x, next_tok], dim=1)
            nv = torch.cat([nv, torch.zeros(1, 1, device=device)], dim=1)
            nm = torch.cat([nm, torch.zeros(1, 1, dtype=torch.bool,
                                            device=device)], dim=1)

        # Decode
        gen_ids = x[0, len(ids):].tolist()
        if EOT_TOKEN_ID in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(EOT_TOKEN_ID)]
        gen_text = tokenizer.decode(
            [t for t in gen_ids if 0 <= t < 50256]).strip()

        # Category-aware extraction
        gen_num = extract_answer(gen_text, category, expected)
        exact_match = gen_text == ref_text
        num_correct = check_correct(gen_text, gen_num, expected, category)

        # Error calculation
        if gen_num is not None and not (category == 'ordering'):
            abs_err = abs(gen_num - expected)
            rel_err = abs_err / max(abs(expected), 1e-9)
        else:
            abs_err = float('inf')
            rel_err = float('inf')

        results.append({
            'category': category,
            'category_type': cat_type,
            'distribution': distribution,
            'expected': expected,
            'gen_num': gen_num,
            'gen_text': gen_text,
            'ref_text': ref_text,
            'exact_match': exact_match,
            'num_correct': num_correct,
            'abs_err': abs_err,
            'rel_err': rel_err,
        })

        if idx < 15:
            status = '✓' if num_correct else '✗'
            print(f"  [{idx}] {status} ({category}/{distribution})")
            print(f"       ref: {ref_text[:120]}")
            print(f"       gen: {gen_text[:120]}")
            print(f"       expected={expected}, got={gen_num}")
            print()

    return results


# =============================================================================
# Metrics
# =============================================================================

def compute_grouped_metrics(results, group_key):
    """Compute metrics grouped by a given key."""
    metrics = {}
    groups = defaultdict(list)
    for r in results:
        groups[r[group_key]].append(r)
    groups['OVERALL'] = results

    for name, group_results in sorted(groups.items()):
        n = len(group_results)
        if n == 0:
            continue
        exact = sum(1 for r in group_results if r['exact_match'])
        correct = sum(1 for r in group_results if r['num_correct'])
        has_num = [r for r in group_results
                   if r['gen_num'] is not None and r['abs_err'] != float('inf')]
        mae = (sum(r['abs_err'] for r in has_num) / len(has_num)
               if has_num else float('inf'))
        parse_rate = len(has_num) / n

        metrics[name] = {
            'n': n,
            'exact_match': exact / n,
            'num_accuracy': correct / n,
            'parse_rate': parse_rate,
            'mae': mae,
        }
    return metrics


def print_metrics_table(metrics, title):
    """Print a formatted metrics table."""
    print(f"\n--- {title} ---")
    print(f"{'Group':<20} {'N':>5} {'ExactM':>8} {'NumAcc':>8} "
          f"{'Parse':>7} {'MAE':>12}")
    print("-" * 62)
    # Print OVERALL last
    keys = sorted(k for k in metrics if k != 'OVERALL')
    keys.append('OVERALL')
    for k in keys:
        if k not in metrics:
            continue
        m = metrics[k]
        mae_s = f"{m['mae']:.2f}" if m['mae'] < 1e10 else "inf"
        print(f"{k:<20} {m['n']:>5} {m['exact_match']:>8.4f} "
              f"{m['num_accuracy']:>8.4f} {m['parse_rate']:>7.3f} "
              f"{mae_s:>12}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark base vs adapted LoRA on arithmetic')
    parser.add_argument('--base_ckpt', required=True)
    parser.add_argument('--adapted_ckpt', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--max_samples', type=int, default=1500)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_path', default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("ARITHMETIC BENCHMARK: Base LoRA vs Adapted LoRA")
    print("  (with ID / OOD-small / OOD-large splits)")
    print("=" * 70)

    with open(args.data_path) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems from {args.data_path}")

    from collections import Counter
    for key in ['distribution', 'category', 'category_type']:
        counts = Counter(p.get(key, '?') for p in problems)
        print(f"\n  By {key}:")
        for k, c in sorted(counts.items()):
            print(f"    {k:<20} {c}")

    all_results = {}
    all_metrics = {}

    for label, ckpt_path, use_adapter in [
        ('Base LoRA', args.base_ckpt, False),
        ('Adapted LoRA', args.adapted_ckpt, True),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  use_adapter={use_adapter}")
        print(f"{'=' * 60}\n")

        model, _ = load_merged_model(ckpt_path, device=args.device)

        results = evaluate_model(
            model, problems, args.device,
            use_adapter=use_adapter,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

        # Compute all breakdowns
        cat_metrics = compute_grouped_metrics(results, 'category')
        dist_metrics = compute_grouped_metrics(results, 'distribution')
        type_metrics = compute_grouped_metrics(results, 'category_type')

        print_metrics_table(cat_metrics, f"{label} — By Category")
        print_metrics_table(dist_metrics, f"{label} — By Distribution")
        print_metrics_table(type_metrics, f"{label} — By Type")

        all_results[label] = results
        all_metrics[label] = {
            'by_category': cat_metrics,
            'by_distribution': dist_metrics,
            'by_type': type_metrics,
        }

        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Comparison tables
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Base LoRA vs Adapted LoRA")
    print("=" * 80)

    for breakdown_name, key in [
        ('By Category', 'by_category'),
        ('By Distribution', 'by_distribution'),
        ('By Type', 'by_type'),
    ]:
        base_m = all_metrics['Base LoRA'][key]
        adapt_m = all_metrics['Adapted LoRA'][key]

        print(f"\n--- {breakdown_name} ---")
        print(f"{'Group':<20} {'Metric':<12} "
              f"{'Base':>10} {'Adapted':>10} {'Delta':>10}")
        print("-" * 64)

        groups = sorted(set(list(base_m.keys()) + list(adapt_m.keys())))
        for group in groups:
            if group not in base_m or group not in adapt_m:
                continue
            bm, am = base_m[group], adapt_m[group]
            for metric, higher_better in [
                ('num_accuracy', True), ('exact_match', True), ('mae', False)
            ]:
                bv, av = bm[metric], am[metric]
                delta = av - bv
                if higher_better:
                    arrow = '↑' if delta > 0 else '↓'
                else:
                    arrow = '↓' if delta < 0 else '↑'
                fmt = '.4f' if metric != 'mae' else '.1f'
                print(f"{group:<20} {metric:<12} "
                      f"{bv:>10{fmt}} {av:>10{fmt}} {delta:>+10{fmt}} {arrow}")
            print()

    print("=" * 80)

    # =========================================================================
    # Save
    # =========================================================================
    if args.out_path is None:
        args.out_path = os.path.join(
            os.path.dirname(args.data_path),
            'arithmetic_benchmark_results.json')

    try:
        save_data = {}
        for label in ['Base LoRA', 'Adapted LoRA']:
            save_data[label] = {
                'metrics': all_metrics[label],
                'examples': [
                    {k: v for k, v in r.items()
                     if k not in ('gen_text', 'ref_text') or len(str(v)) < 200}
                    for r in all_results[label][:100]
                ],
            }
        with open(args.out_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.out_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")


if __name__ == '__main__':
    main()
