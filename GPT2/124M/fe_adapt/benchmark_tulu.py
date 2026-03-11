"""Benchmark base vs adapted LoRA models on tulu-3 math test set.

Metrics:
  1. Forward-pass: loss, perplexity, token accuracy
  2. Generation: exact match, number accuracy, mean absolute error

Usage:
  python benchmark_tulu.py \
    --base_ckpt /path/to/base/ckpt_merged.pt \
    --adapted_ckpt /path/to/adapted/ckpt_merged.pt \
    --base_data_dir /path/to/base/ \
    --adapted_data_dir /path/to/adapted/
"""

import os
import sys
import json
import math
import re
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NemotronConfig, Nemotron, NUM_TOKEN_ID
from model_analytic import NemotronAnalyticConfig, NemotronAnalytic
from prepare import _get_tokenizer, EOT_TOKEN_ID, NUMBER_PATTERN


def load_merged_model(ckpt_path, device='cpu'):
    """Load a merged (LoRA-folded) checkpoint."""
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

    model_args = ckpt['model_args']
    if 'analytic_K' in model_args:
        from model_analytic import NemotronAnalyticConfig, NemotronAnalytic
        nemconf = NemotronAnalyticConfig(**model_args)
        model = NemotronAnalytic(nemconf)
    else:
        nemconf = NemotronConfig(**model_args)
        model = Nemotron(nemconf)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    use_adapter = ckpt.get('config', {}).get('use_adapter', True)
    print(f"  Loaded {model.__class__.__name__} ({len(state_dict)} keys), use_adapter={use_adapter}")
    return model, use_adapter


# =============================================================================
# Forward-pass evaluation
# =============================================================================

@torch.no_grad()
def evaluate_forward(model, data_dir, device, block_size=512, batch_size=4,
                     n_batches=200, num_norm_match=True):
    """Compute loss, perplexity, token accuracy on test split."""
    data = np.memmap(os.path.join(data_dir, 'test.bin'),
                     dtype=np.uint16, mode='r')
    nums = np.memmap(os.path.join(data_dir, 'test_nums.bin'),
                     dtype=np.float32, mode='r')

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    max_start = len(data) - block_size - 1
    if max_start < 1:
        print(f"  WARNING: test data too small ({len(data)} tokens)")
        return {'loss': float('inf'), 'perplexity': float('inf'), 'accuracy': 0.0}

    for _ in range(n_batches):
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))
                         for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
                         for i in ix]).to(device)
        nv = torch.stack([torch.from_numpy(nums[i:i + block_size].copy())
                          for i in ix]).to(device)
        nm = (x == NUM_TOKEN_ID)

        with ctx:
            if isinstance(model, NemotronAnalytic):
                logits, _, _ = model(x, y, num_values=nv, num_mask=nm)
            else:
                logits, loss = model(x, y, num_values=nv, num_mask=nm,
                                     num_blend_beta=1.0,
                                     num_norm_match=num_norm_match)
        
        if isinstance(model, NemotronAnalytic):
            # Compute cross entropy for perplexity
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        valid = (y >= 0) & (y != EOT_TOKEN_ID)
        total_correct += (preds[valid] == y[valid]).sum().item()
        total_tokens += valid.sum().item()

    avg_loss = total_loss / n_batches
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))  # cap to avoid overflow

    return {'loss': avg_loss, 'perplexity': perplexity, 'accuracy': accuracy}


# =============================================================================
# Generation evaluation
# =============================================================================

def process_content_with_numbers(text):
    """Replace numbers in text with <NUM> tokens (no EOT)."""
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
        try:
            value = float(match.group())
        except ValueError:
            seg_ids = tokenizer.encode(match.group(), add_special_tokens=False)
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


def format_prompt(messages, use_adapter=False):
    """Format conversation prompt up to 'Assistant:' for generation.

    Returns (token_ids, num_values) ready for model input.
    """
    tokenizer = _get_tokenizer()
    ids = []
    nums = []

    # Include all user messages and any assistant messages before the last one
    # The last assistant message is the reference answer
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant' and i == len(messages) - 1:
            # This is the answer — add only the prefix
            prefix = "\nAssistant:"
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            ids.extend(prefix_ids)
            nums.extend([0.0] * len(prefix_ids))
            break

        role = msg['role']
        content = msg['content'].strip()

        if i > 0:
            prefix = f"\n{role.capitalize()}: "
        else:
            prefix = f"{role.capitalize()}: "
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


def extract_numbers(text):
    """Extract all numbers from text."""
    return [float(x) for x in NUMBER_PATTERN.findall(text)]


@torch.no_grad()
def evaluate_generation(model, test_examples, device, use_adapter=False,
                        max_samples=100, max_new_tokens=256, temperature=0.0):
    """Generate answers and compare with references."""
    tokenizer = _get_tokenizer()

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    results = []
    for idx, ex in enumerate(test_examples[:max_samples]):
        messages = ex['messages']

        # Get reference answer
        ref_msg = None
        for msg in messages:
            if msg['role'] == 'assistant':
                ref_msg = msg
        if ref_msg is None:
            continue
        ref_text = ref_msg['content'].strip()

        # Format prompt
        ids, num_vals = format_prompt(messages, use_adapter=use_adapter)

        x = torch.tensor([ids], dtype=torch.long, device=device)
        nv = torch.tensor([num_vals], dtype=torch.float32, device=device)
        nm = (x == NUM_TOKEN_ID)

        # Generate (greedy if temperature=0)
        if temperature <= 0:
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
                    if isinstance(model, NemotronAnalytic):
                        logits, _, _ = model(x_cond, num_values=nv_cond, num_mask=nm_cond)
                    else:
                        logits, _ = model(x_cond, num_values=nv_cond, num_mask=nm_cond)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_tok.item() == EOT_TOKEN_ID:
                    break
                x = torch.cat([x, next_tok], dim=1)
                nv = torch.cat([nv, torch.zeros(1, 1, device=device)], dim=1)
                nm = torch.cat([nm, torch.zeros(1, 1, dtype=torch.bool, device=device)], dim=1)
        else:
            x = model.generate(x, max_new_tokens, temperature=temperature,
                               num_values=nv, num_mask=nm)

        # Decode generated part
        gen_ids = x[0, len(ids):].tolist()
        if EOT_TOKEN_ID in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(EOT_TOKEN_ID)]
        gen_text = tokenizer.decode([t for t in gen_ids if 0 <= t < 50256]).strip()

        # Compare
        exact_match = gen_text == ref_text
        gen_nums = extract_numbers(gen_text)
        ref_nums = extract_numbers(ref_text)

        results.append({
            'exact_match': exact_match,
            'gen_text': gen_text,
            'ref_text': ref_text,
            'gen_nums': gen_nums,
            'ref_nums': ref_nums,
        })

        if idx < 5:
            print(f"  [{idx}] ref: {ref_text[:100]}")
            print(f"       gen: {gen_text[:100]}")
            print(f"       match: {exact_match}")
            print()

    # Aggregate
    n = len(results)
    if n == 0:
        return {'exact_match': 0.0, 'num_accuracy': 0.0, 'mae': float('inf'),
                'n_samples': 0}

    exact_matches = sum(1 for r in results if r['exact_match'])

    total_correct_nums = 0
    total_ref_nums = 0
    total_abs_err = 0.0
    n_matched = 0

    for r in results:
        ref = r['ref_nums']
        gen = r['gen_nums']
        total_ref_nums += len(ref)
        for i in range(min(len(ref), len(gen))):
            err = abs(ref[i] - gen[i])
            total_abs_err += err
            n_matched += 1
            if err < 0.01:
                total_correct_nums += 1

    return {
        'exact_match': exact_matches / n,
        'num_accuracy': total_correct_nums / max(total_ref_nums, 1),
        'mae': total_abs_err / max(n_matched, 1),
        'n_samples': n,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark base vs adapted LoRA')
    parser.add_argument('--base_ckpt', required=True)
    parser.add_argument('--adapted_ckpt', required=True)
    parser.add_argument('--base_data_dir', required=True)
    parser.add_argument('--adapted_data_dir', required=True)
    parser.add_argument('--n_forward_batches', type=int, default=200)
    parser.add_argument('--n_gen_samples', type=int, default=100)
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print("=" * 70)
    print("TULU-3 MATH BENCHMARK: Base LoRA vs Adapted LoRA")
    print("=" * 70)

    # Load test examples for generation
    test_json = os.path.join(args.base_data_dir, 'test_examples.json')
    if os.path.exists(test_json):
        with open(test_json) as f:
            test_examples = json.load(f)
        print(f"Loaded {len(test_examples)} test examples from {test_json}")
    else:
        test_examples = []
        print(f"WARNING: {test_json} not found, skipping generation eval")

    results = {}

    for label, ckpt_path, data_dir, use_adapter in [
        ('Base LoRA', args.base_ckpt, args.base_data_dir, False),
        ('Adapted LoRA', args.adapted_ckpt, args.adapted_data_dir, True),
    ]:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Data: {data_dir}")
        print(f"{'=' * 50}")

        model, _ = load_merged_model(ckpt_path, device=args.device)

        # Forward metrics
        print(f"\n--- Forward-pass metrics ({args.n_forward_batches} batches) ---")
        fwd = evaluate_forward(
            model, data_dir, args.device,
            block_size=args.block_size, batch_size=args.batch_size,
            n_batches=args.n_forward_batches,
        )
        print(f"  Loss:       {fwd['loss']:.4f}")
        print(f"  Perplexity: {fwd['perplexity']:.2f}")
        print(f"  Accuracy:   {fwd['accuracy']:.4f}")

        # Generation metrics
        gen = {'exact_match': 0, 'num_accuracy': 0, 'mae': float('inf'), 'n_samples': 0}
        if test_examples:
            print(f"\n--- Generation metrics ({args.n_gen_samples} samples, greedy) ---")
            gen = evaluate_generation(
                model, test_examples, args.device,
                use_adapter=use_adapter,
                max_samples=args.n_gen_samples,
            )
            print(f"  Exact match:    {gen['exact_match']:.4f}")
            print(f"  Number accuracy:{gen['num_accuracy']:.4f}")
            print(f"  Number MAE:     {gen['mae']:.4f}")

        results[label] = {'forward': fwd, 'generation': gen}

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Base LoRA':>15} {'Adapted LoRA':>15} {'Delta':>10}")
    print("-" * 65)

    base = results['Base LoRA']
    adapt = results['Adapted LoRA']

    rows = [
        ('Loss', base['forward']['loss'], adapt['forward']['loss'], True),
        ('Perplexity', base['forward']['perplexity'], adapt['forward']['perplexity'], True),
        ('Token Accuracy', base['forward']['accuracy'], adapt['forward']['accuracy'], False),
    ]
    if test_examples:
        rows.extend([
            ('Exact Match', base['generation']['exact_match'],
             adapt['generation']['exact_match'], False),
            ('Number Accuracy', base['generation']['num_accuracy'],
             adapt['generation']['num_accuracy'], False),
            ('Number MAE', base['generation']['mae'],
             adapt['generation']['mae'], True),
        ])

    for name, b_val, a_val, lower_is_better in rows:
        delta = a_val - b_val
        if lower_is_better:
            arrow = 'v' if delta < 0 else '^'
        else:
            arrow = '^' if delta > 0 else 'v'
        print(f"{name:<25} {b_val:>15.4f} {a_val:>15.4f} {delta:>+9.4f} {arrow}")

    print("=" * 70)

    # Save results
    out_path = os.path.join(os.path.dirname(args.base_ckpt), '..', 'benchmark_results.json')
    try:
        # Convert to serializable
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                'forward': v['forward'],
                'generation': {kk: vv for kk, vv in v['generation'].items()
                               if kk != 'results'},
            }
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {out_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")


if __name__ == '__main__':
    main()
