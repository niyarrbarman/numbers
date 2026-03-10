"""Benchmark base vs adapted LoRA on custom arithmetic problems.

Handles dual tokenization:
  - Base model:    plain text tokenization (no <NUM>)
  - Adapted model: numbers in user prompt → <NUM> token + float values

Metrics per category + overall:
  - Exact match rate
  - Number accuracy (within ε=0.01)
  - Mean absolute error (MAE)
  - Mean relative error

Usage:
  python benchmark_arithmetic.py \
    --base_ckpt /path/to/base/ckpt_merged.pt \
    --adapted_ckpt /path/to/adapted/ckpt_merged.pt \
    --data_path /path/to/arithmetic_bench.json \
    --max_samples 1000
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


# =============================================================================
# Model loading (same as benchmark_tulu.py)
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
# Few-shot examples (2 per category) to steer output format
# =============================================================================

FEW_SHOT_EXAMPLES = {
    'addition': [
        {'user': 'What is 12 + 5?', 'assistant': '12 + 5 = 17'},
        {'user': 'What is 248 + 193?', 'assistant': '248 + 193 = 441'},
    ],
    'subtraction': [
        {'user': 'What is 50 - 23?', 'assistant': '50 - 23 = 27'},
        {'user': 'What is 1024 - 378?', 'assistant': '1024 - 378 = 646'},
    ],
    'multiplication': [
        {'user': 'What is 7 times 8?', 'assistant': '7 times 8 = 56'},
        {'user': 'What is 23 times 17?', 'assistant': '23 times 17 = 391'},
    ],
    'comparison': [
        {'user': 'Which is larger, 15 or 9?', 'assistant': '15 is larger than 9'},
        {'user': 'Which is larger, 3.8 or 4.2?', 'assistant': '4.2 is larger than 3.8'},
    ],
    'percentage': [
        {'user': 'What is 10% of 200?', 'assistant': '10% of 200 = 20.0'},
        {'user': 'What is 25% of 480?', 'assistant': '25% of 480 = 120.0'},
    ],
    'multistep': [
        {'user': 'If you have 30 and add 15, then subtract 10, what is the result?',
         'assistant': '30 + 15 = 45, 45 - 10 = 35. The result is 35'},
        {'user': 'If you have 100 and add 250, then subtract 75, what is the result?',
         'assistant': '100 + 250 = 350, 350 - 75 = 275. The result is 275'},
    ],
}


# =============================================================================
# Tokenization — handles base vs adapted differently
# =============================================================================

def process_content_with_numbers(text):
    """Replace numbers in text with <NUM> tokens.

    Returns (token_ids, num_values) — no EOT appended.
    """
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


def _tokenize_turn(role, content, use_adapter, is_first, tokenizer):
    """Tokenize a single user/assistant turn.

    Returns (token_ids, num_values) for this turn.
    """
    ids = []
    nums = []

    # Role prefix
    if not is_first:
        prefix = f"\n{role.capitalize()}: "
    else:
        prefix = f"{role.capitalize()}: "
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids.extend(prefix_ids)
    nums.extend([0.0] * len(prefix_ids))

    # Content
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
    """Format conversation prompt for generation, with few-shot examples.

    For the base model:  plain text tokenization.
    For the adapted model: numbers in user content are replaced with <NUM>.

    If category is provided, prepends 2 few-shot examples of the same
    category so the model sees the expected concise answer format.

    Returns (token_ids, num_values) ready for model input.
    The prompt ends after "Assistant:" so the model generates the answer.
    """
    tokenizer = _get_tokenizer()
    ids = []
    nums = []
    turn_idx = 0

    # --- Prepend few-shot examples ---
    if category and category in FEW_SHOT_EXAMPLES:
        for ex in FEW_SHOT_EXAMPLES[category]:
            # User turn
            t_ids, t_nums = _tokenize_turn(
                'user', ex['user'], use_adapter, turn_idx == 0, tokenizer)
            ids.extend(t_ids)
            nums.extend(t_nums)
            turn_idx += 1

            # Assistant turn (always plain text — it's the answer)
            t_ids, t_nums = _tokenize_turn(
                'assistant', ex['assistant'], False, False, tokenizer)
            ids.extend(t_ids)
            nums.extend(t_nums)
            turn_idx += 1

    # --- Actual problem ---
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant' and i == len(messages) - 1:
            # Last assistant message = reference answer → just add prefix
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
# Number extraction
# =============================================================================

def extract_final_number(text):
    """Extract the last number from text (usually the answer)."""
    matches = NUMBER_PATTERN.findall(text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def extract_all_numbers(text):
    """Extract all numbers from text."""
    return [float(x) for x in NUMBER_PATTERN.findall(text)]


# =============================================================================
# Generation-based evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(model, problems, device, use_adapter=False,
                   max_samples=1000, max_new_tokens=128):
    """Evaluate a model on arithmetic problems with greedy decoding.

    Args:
        model: loaded Nemotron model
        problems: list of problem dicts with 'messages', 'category', etc.
        device: 'cuda' or 'cpu'
        use_adapter: if True, tokenize with <NUM> replacement
        max_samples: max problems to evaluate
        max_new_tokens: max tokens to generate per problem
    """
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
        difficulty = problem['difficulty']
        expected = problem['expected_answer']

        # Get reference answer text
        ref_text = messages[-1]['content'].strip()

        # Tokenize prompt (stops before assistant answer)
        ids, num_vals = format_prompt(messages, use_adapter=use_adapter,
                                       category=category)

        x = torch.tensor([ids], dtype=torch.long, device=device)
        nv = torch.tensor([num_vals], dtype=torch.float32, device=device)
        nm = (x == NUM_TOKEN_ID)

        # Greedy decoding
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

        # Decode generated tokens
        gen_ids = x[0, len(ids):].tolist()
        if EOT_TOKEN_ID in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(EOT_TOKEN_ID)]
        gen_text = tokenizer.decode(
            [t for t in gen_ids if 0 <= t < 50256]).strip()

        # Extract answer number
        gen_num = extract_final_number(gen_text)
        exact_match = gen_text == ref_text

        # Compute error
        if gen_num is not None:
            abs_err = abs(gen_num - expected)
            num_correct = abs_err < 0.01
            if abs(expected) > 1e-9:
                rel_err = abs_err / abs(expected)
            else:
                rel_err = abs_err
        else:
            abs_err = float('inf')
            num_correct = False
            rel_err = float('inf')

        results.append({
            'category': category,
            'difficulty': difficulty,
            'expected': expected,
            'gen_num': gen_num,
            'gen_text': gen_text,
            'ref_text': ref_text,
            'exact_match': exact_match,
            'num_correct': num_correct,
            'abs_err': abs_err,
            'rel_err': rel_err,
        })

        # Print first few examples per category
        if idx < 12:
            print(f"  [{idx}] ({category}/{difficulty})")
            print(f"       ref: {ref_text[:120]}")
            print(f"       gen: {gen_text[:120]}")
            print(f"       expected={expected}, got={gen_num}, "
                  f"match={exact_match}")
            print()

    return results


# =============================================================================
# Metrics aggregation
# =============================================================================

def compute_metrics(results):
    """Compute per-category and overall metrics."""
    metrics = {}

    # Group by category
    by_category = defaultdict(list)
    for r in results:
        by_category[r['category']].append(r)
    by_category['OVERALL'] = results

    for cat, cat_results in sorted(by_category.items()):
        n = len(cat_results)
        if n == 0:
            continue

        exact = sum(1 for r in cat_results if r['exact_match'])
        num_correct = sum(1 for r in cat_results if r['num_correct'])
        has_num = [r for r in cat_results if r['gen_num'] is not None]
        mae = (sum(r['abs_err'] for r in has_num) / len(has_num)
               if has_num else float('inf'))
        mre = (sum(r['rel_err'] for r in has_num) / len(has_num)
               if has_num else float('inf'))
        parse_rate = len(has_num) / n

        metrics[cat] = {
            'n': n,
            'exact_match': exact / n,
            'num_accuracy': num_correct / n,
            'parse_rate': parse_rate,
            'mae': mae,
            'mre': mre,
        }

    # Also compute by difficulty
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r['difficulty']].append(r)

    diff_metrics = {}
    for diff, diff_results in sorted(by_diff.items()):
        n = len(diff_results)
        num_correct = sum(1 for r in diff_results if r['num_correct'])
        has_num = [r for r in diff_results if r['gen_num'] is not None]
        mae = (sum(r['abs_err'] for r in has_num) / len(has_num)
               if has_num else float('inf'))
        diff_metrics[diff] = {
            'n': n,
            'num_accuracy': num_correct / n,
            'mae': mae,
        }

    metrics['_by_difficulty'] = diff_metrics
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark base vs adapted LoRA on arithmetic')
    parser.add_argument('--base_ckpt', required=True,
                        help='Path to base (merged) checkpoint')
    parser.add_argument('--adapted_ckpt', required=True,
                        help='Path to adapted (merged) checkpoint')
    parser.add_argument('--data_path', required=True,
                        help='Path to arithmetic benchmark JSON')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_path', default=None,
                        help='Path to save results JSON')
    args = parser.parse_args()

    print("=" * 70)
    print("CUSTOM ARITHMETIC BENCHMARK: Base LoRA vs Adapted LoRA")
    print("=" * 70)

    # Load benchmark data
    with open(args.data_path) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} arithmetic problems from {args.data_path}")

    from collections import Counter
    cat_counts = Counter(p['category'] for p in problems)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<20} {cnt}")

    all_results = {}
    all_metrics = {}

    for label, ckpt_path, use_adapter in [
        ('Base LoRA', args.base_ckpt, False),
        ('Adapted LoRA', args.adapted_ckpt, True),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  use_adapter={use_adapter} "
              f"({'<NUM> tokenization' if use_adapter else 'plain text'})")
        print(f"{'=' * 60}\n")

        model, _ = load_merged_model(ckpt_path, device=args.device)

        results = evaluate_model(
            model, problems, args.device,
            use_adapter=use_adapter,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

        metrics = compute_metrics(results)
        all_results[label] = results
        all_metrics[label] = metrics

        # Print per-category metrics
        print(f"\n--- {label} Results ---")
        print(f"{'Category':<20} {'N':>5} {'ExactM':>8} {'NumAcc':>8} "
              f"{'Parse':>7} {'MAE':>12} {'MRE':>10}")
        print("-" * 72)
        for cat in sorted(metrics.keys()):
            if cat.startswith('_'):
                continue
            m = metrics[cat]
            print(f"{cat:<20} {m['n']:>5} {m['exact_match']:>8.4f} "
                  f"{m['num_accuracy']:>8.4f} {m['parse_rate']:>7.3f} "
                  f"{m['mae']:>12.2f} {m['mre']:>10.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Comparison table
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Base LoRA vs Adapted LoRA")
    print("=" * 80)

    base_m = all_metrics['Base LoRA']
    adapt_m = all_metrics['Adapted LoRA']

    # Per-category comparison
    print(f"\n{'Category':<18} {'Metric':<12} "
          f"{'Base':>10} {'Adapted':>10} {'Delta':>10}")
    print("-" * 62)

    categories = sorted(set(list(base_m.keys()) + list(adapt_m.keys())))
    for cat in categories:
        if cat.startswith('_') or cat not in base_m or cat not in adapt_m:
            continue
        bm = base_m[cat]
        am = adapt_m[cat]
        for metric, higher_better in [('num_accuracy', True), ('mae', False)]:
            bv = bm[metric]
            av = am[metric]
            delta = av - bv
            if higher_better:
                arrow = '↑' if delta > 0 else '↓'
            else:
                arrow = '↓' if delta < 0 else '↑'
            bv_s = f"{bv:.4f}" if metric != 'mae' else f"{bv:.2f}"
            av_s = f"{av:.4f}" if metric != 'mae' else f"{av:.2f}"
            delta_s = f"{delta:+.4f}" if metric != 'mae' else f"{delta:+.2f}"
            print(f"{cat:<18} {metric:<12} {bv_s:>10} {av_s:>10} "
                  f"{delta_s:>10} {arrow}")
        print()

    # By difficulty
    print("\n--- By Difficulty ---")
    base_diff = base_m.get('_by_difficulty', {})
    adapt_diff = adapt_m.get('_by_difficulty', {})
    print(f"{'Difficulty':<12} {'Base NumAcc':>12} {'Adapt NumAcc':>12} "
          f"{'Delta':>10}")
    print("-" * 48)
    for diff in ['easy', 'medium', 'hard']:
        if diff in base_diff and diff in adapt_diff:
            bv = base_diff[diff]['num_accuracy']
            av = adapt_diff[diff]['num_accuracy']
            delta = av - bv
            arrow = '↑' if delta > 0 else '↓'
            print(f"{diff:<12} {bv:>12.4f} {av:>12.4f} "
                  f"{delta:>+10.4f} {arrow}")

    print("=" * 80)

    # =========================================================================
    # Save results
    # =========================================================================
    if args.out_path is None:
        args.out_path = os.path.join(
            os.path.dirname(args.data_path),
            'arithmetic_benchmark_results.json')

    try:
        # Make results JSON-serializable
        save_data = {}
        for label in ['Base LoRA', 'Adapted LoRA']:
            save_data[label] = {
                'metrics': {k: v for k, v in all_metrics[label].items()},
                'examples': [
                    {
                        'category': r['category'],
                        'difficulty': r['difficulty'],
                        'expected': r['expected'],
                        'gen_num': r['gen_num'],
                        'gen_text': r['gen_text'][:200],
                        'ref_text': r['ref_text'][:200],
                        'exact_match': r['exact_match'],
                        'num_correct': r['num_correct'],
                    }
                    for r in all_results[label][:50]  # save first 50
                ],
            }

        with open(args.out_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.out_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")


if __name__ == '__main__':
    main()
