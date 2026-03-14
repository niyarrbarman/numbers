"""Zero-shot synthetic arithmetic benchmark for base vs surface-adapted models."""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import NemotronConfig, Nemotron, NUM_TOKEN_ID
from model_analytic_surface import NemotronAnalyticConfig, NemotronAnalytic
from numeric_surface import (
    canonical_decimal_string,
    render_surface_components,
    surface_components_from_value,
)
from prepare import _get_tokenizer, EOT_TOKEN_ID, NUMBER_PATTERN


def load_merged_model(ckpt_path, device='cpu'):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']
    for key in list(state_dict.keys()):
        if key.startswith('_orig_mod.'):
            state_dict[key[len('_orig_mod.'):]] = state_dict.pop(key)

    model_args = ckpt['model_args']
    if 'analytic_K' in model_args:
        nemconf = NemotronAnalyticConfig(**model_args)
        model = NemotronAnalytic(nemconf)
    else:
        nemconf = NemotronConfig(**model_args)
        model = Nemotron(nemconf)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def evaluate_forward(model, data_dir, device, block_size=512, batch_size=4, n_batches=200):
    data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    nums = np.memmap(os.path.join(data_dir, 'test_nums.bin'), dtype=np.float32, mode='r')

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load surface supervision if this is a surface-mode NemotronAnalytic
    surface_comps = None
    if isinstance(model, NemotronAnalytic) and model.config.numeric_output_mode == 'surface':
        target_cols = 3 + model.config.surface_max_digits
        surface_path = os.path.join(data_dir, 'test_surface.bin')
        raw = np.memmap(surface_path, dtype=np.uint8, mode='r')
        surface_comps = raw.reshape(len(data), target_cols)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    eval_block_size = min(block_size, model.config.block_size)
    max_start = len(data) - eval_block_size - 1
    if max_start < 1:
        return {'loss': float('inf'), 'perplexity': float('inf'), 'accuracy': 0.0}

    for _ in range(n_batches):
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + eval_block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + eval_block_size].astype(np.int64)) for i in ix]).to(device)
        nv = torch.stack([torch.from_numpy(nums[i:i + eval_block_size].copy()) for i in ix]).to(device)
        nm = (x == NUM_TOKEN_ID)

        with ctx:
            if isinstance(model, NemotronAnalytic):
                kwargs = {}
                if surface_comps is not None:
                    nc = torch.stack([
                        torch.from_numpy(surface_comps[i + 1:i + 1 + eval_block_size].copy())
                        for i in ix
                    ]).to(device)
                    kwargs['num_target_surface'] = nc
                logits, _, _ = model(x, y, num_values=nv, num_mask=nm, **kwargs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )
            else:
                logits, loss = model(x, y, num_values=nv, num_mask=nm, num_blend_beta=1.0, num_norm_match=True)

        total_loss += float(loss.item())
        preds = logits.argmax(dim=-1)
        valid = (y >= 0) & (y != EOT_TOKEN_ID)
        total_correct += int((preds[valid] == y[valid]).sum().item())
        total_tokens += int(valid.sum().item())

    avg_loss = total_loss / n_batches
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    return {'loss': avg_loss, 'perplexity': perplexity, 'accuracy': accuracy}


def process_content_with_numbers(text):
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


def format_zero_shot_prompt(messages, use_adapter=False):
    tokenizer = _get_tokenizer()
    ids = []
    nums = []

    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant' and i == len(messages) - 1:
            prefix = '\nAssistant: ' if i > 0 else 'Assistant: '
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            ids.extend(prefix_ids)
            nums.extend([0.0] * len(prefix_ids))
            break

        prefix = f"{msg['role'].capitalize()}: " if i == 0 else f"\n{msg['role'].capitalize()}: "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        ids.extend(prefix_ids)
        nums.extend([0.0] * len(prefix_ids))

        content = msg['content'].strip()
        if use_adapter and msg['role'] == 'user':
            content_ids, content_nums = process_content_with_numbers(content)
            ids.extend(content_ids)
            nums.extend(content_nums)
        else:
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            ids.extend(content_ids)
            nums.extend([0.0] * len(content_ids))

    return ids, nums


def render_generated_text(tokenizer, gen_ids, generated_numbers=None, prompt_len=0):
    num_by_pos = {}
    if generated_numbers is not None:
        num_by_pos = {pos: text for pos, text in generated_numbers}

    parts = []
    text_buf = []
    for offset, tok in enumerate(gen_ids):
        abs_pos = prompt_len + offset
        if tok == EOT_TOKEN_ID:
            break
        if tok == NUM_TOKEN_ID:
            if text_buf:
                parts.append(tokenizer.decode(text_buf))
                text_buf = []
            parts.append(num_by_pos.get(abs_pos, '<NUM>'))
            continue
        if 0 <= tok < NUM_TOKEN_ID:
            text_buf.append(tok)

    if text_buf:
        parts.append(tokenizer.decode(text_buf))
    return ''.join(parts).strip()


def extract_number_strings(text):
    return NUMBER_PATTERN.findall(text)


def to_surface(text_number):
    try:
        return surface_components_from_value(text_number)
    except Exception:
        return None


def compare_surface_numbers(ref_numbers, gen_numbers):
    matched = min(len(ref_numbers), len(gen_numbers))
    digit_correct = 0
    digit_total = 0
    scale_correct = 0
    length_correct = 0
    numeric_correct = 0
    abs_err = 0.0
    first_wrong_positions = []

    for i in range(matched):
        ref_comp = to_surface(ref_numbers[i])
        gen_comp = to_surface(gen_numbers[i])
        if ref_comp is None or gen_comp is None:
            continue

        ref_text = render_surface_components(ref_comp)
        gen_text = render_surface_components(gen_comp)
        if ref_text == gen_text:
            numeric_correct += 1
        try:
            abs_err += abs(float(ref_text) - float(gen_text))
        except ValueError:
            pass

        if ref_comp.scale == gen_comp.scale:
            scale_correct += 1
        if ref_comp.length == gen_comp.length:
            length_correct += 1

        ref_digits = ref_comp.active_digits()
        gen_digits = gen_comp.active_digits()
        first_wrong = 0
        for pos in range(ref_comp.length):
            gen_digit = gen_digits[pos] if pos < len(gen_digits) else None
            if gen_digit == ref_digits[pos]:
                digit_correct += 1
            elif first_wrong == 0:
                first_wrong = pos + 1
            digit_total += 1
        if first_wrong == 0 and ref_comp.length != gen_comp.length:
            first_wrong = min(ref_comp.length, gen_comp.length) + 1
        if first_wrong > 0:
            first_wrong_positions.append(first_wrong)

    return {
        'matched': matched,
        'digit_correct': digit_correct,
        'digit_total': digit_total,
        'scale_correct': scale_correct,
        'length_correct': length_correct,
        'numeric_correct': numeric_correct,
        'abs_err': abs_err,
        'first_wrong_positions': first_wrong_positions,
    }


def copied_number_consistency(prompt_text, ref_text, gen_text):
    prompt_numbers = {canonical_decimal_string(x) for x in extract_number_strings(prompt_text)}
    ref_numbers = [canonical_decimal_string(x) for x in extract_number_strings(ref_text)]
    gen_numbers = [canonical_decimal_string(x) for x in extract_number_strings(gen_text)]
    copy_positions = [i for i, num in enumerate(ref_numbers) if num in prompt_numbers]
    if not copy_positions:
        return None
    correct = 0
    for pos in copy_positions:
        if pos < len(gen_numbers) and gen_numbers[pos] == ref_numbers[pos]:
            correct += 1
    return {'correct': correct, 'total': len(copy_positions)}


@torch.no_grad()
def evaluate_generation(model, test_examples, device, use_adapter=False, max_samples=100, max_new_tokens=256):
    tokenizer = _get_tokenizer()
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    results = []
    totals = {
        'exact_match': 0,
        'full_exact_match': 0,
        'num_accuracy_correct': 0,
        'num_accuracy_total': 0,
        'mae_abs_err': 0.0,
        'mae_count': 0,
        'digit_correct': 0,
        'digit_total': 0,
        'scale_correct': 0,
        'scale_total': 0,
        'length_correct': 0,
        'length_total': 0,
        'copy_correct': 0,
        'copy_total': 0,
        'first_wrong_positions': [],
    }

    for ex_idx, ex in enumerate(test_examples[:max_samples]):
        messages = ex['messages']
        ref_msg = next((msg for msg in reversed(messages) if msg['role'] == 'assistant'), None)
        if ref_msg is None:
            continue
        ref_text = ref_msg['content'].strip()
        prompt_text = '\n'.join(msg['content'].strip() for msg in messages if msg['role'] == 'user')
        ids, num_vals = format_zero_shot_prompt(messages, use_adapter=use_adapter and isinstance(model, NemotronAnalytic))

        x = torch.tensor([ids], dtype=torch.long, device=device)
        nv = torch.tensor([num_vals], dtype=torch.float32, device=device)
        nm = (x == NUM_TOKEN_ID)

        generated_numbers = None
        generated_components = None
        if isinstance(model, NemotronAnalytic):
            with ctx:
                x, generated_numbers, generated_components = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_k=1,
                    num_values=nv,
                    num_mask=nm,
                    return_components=True,
                )
        else:
            for _ in range(max_new_tokens):
                if x.size(1) > model.config.block_size:
                    x_cond = x[:, -model.config.block_size:]
                    nv_cond = nv[:, -model.config.block_size:]
                    nm_cond = nm[:, -model.config.block_size:]
                else:
                    x_cond, nv_cond, nm_cond = x, nv, nm
                with ctx:
                    logits, _ = model(x_cond, num_values=nv_cond, num_mask=nm_cond)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if int(next_tok.item()) == EOT_TOKEN_ID:
                    break
                x = torch.cat([x, next_tok], dim=1)
                nv = torch.cat([nv, torch.zeros((1, 1), dtype=torch.float32, device=device)], dim=1)
                nm = torch.cat([nm, torch.zeros((1, 1), dtype=torch.bool, device=device)], dim=1)

        gen_ids = x[0, len(ids):].tolist()
        gen_text = render_generated_text(
            tokenizer,
            gen_ids,
            generated_numbers=generated_numbers,
            prompt_len=len(ids),
        )

        ref_numbers = extract_number_strings(ref_text)
        gen_numbers = extract_number_strings(gen_text)
        comp_metrics = compare_surface_numbers(ref_numbers, gen_numbers)
        copy_metrics = copied_number_consistency(prompt_text, ref_text, gen_text)
        exact_match = gen_text == ref_text

        totals['exact_match'] += int(exact_match)
        totals['full_exact_match'] += int(exact_match)
        totals['num_accuracy_correct'] += comp_metrics['numeric_correct']
        totals['num_accuracy_total'] += len(ref_numbers)
        totals['mae_abs_err'] += comp_metrics['abs_err']
        totals['mae_count'] += comp_metrics['matched']
        totals['digit_correct'] += comp_metrics['digit_correct']
        totals['digit_total'] += comp_metrics['digit_total']
        totals['scale_correct'] += comp_metrics['scale_correct']
        totals['scale_total'] += comp_metrics['matched']
        totals['length_correct'] += comp_metrics['length_correct']
        totals['length_total'] += comp_metrics['matched']
        totals['first_wrong_positions'].extend(comp_metrics['first_wrong_positions'])
        if copy_metrics is not None:
            totals['copy_correct'] += copy_metrics['correct']
            totals['copy_total'] += copy_metrics['total']

        results.append({
            'example_index': ex_idx,
            'prompt_text': prompt_text,
            'ref_text': ref_text,
            'gen_text': gen_text,
            'exact_match': exact_match,
            'ref_numbers': ref_numbers,
            'gen_numbers': gen_numbers,
            'digit_accuracy': (
                comp_metrics['digit_correct'] / comp_metrics['digit_total']
                if comp_metrics['digit_total'] else None
            ),
            'scale_accuracy': (
                comp_metrics['scale_correct'] / comp_metrics['matched']
                if comp_metrics['matched'] else None
            ),
            'length_accuracy': (
                comp_metrics['length_correct'] / comp_metrics['matched']
                if comp_metrics['matched'] else None
            ),
            'copied_number_consistency': (
                copy_metrics['correct'] / copy_metrics['total']
                if copy_metrics is not None and copy_metrics['total'] else None
            ),
            'first_wrong_digit_position': (
                float(np.mean(comp_metrics['first_wrong_positions']))
                if comp_metrics['first_wrong_positions'] else None
            ),
            'generated_numeric_components': [
                {
                    'position': pos,
                    'rendered': render_surface_components(comp) if hasattr(comp, 'scale') else str(comp),
                }
                for pos, comp in (generated_components or [])
            ],
            'messages': messages,
        })

        if ex_idx < 5:
            print(f"  [{ex_idx}] ref: {ref_text[:120]}")
            print(f"       gen: {gen_text[:120]}")
            print(f"       match: {exact_match}")
            print()

    n = len(results)
    return {
        'exact_match': totals['exact_match'] / max(n, 1),
        'full_exact_match': totals['full_exact_match'] / max(n, 1),
        'num_accuracy': totals['num_accuracy_correct'] / max(totals['num_accuracy_total'], 1),
        'mae': totals['mae_abs_err'] / max(totals['mae_count'], 1),
        'digit_accuracy': totals['digit_correct'] / max(totals['digit_total'], 1),
        'scale_accuracy': totals['scale_correct'] / max(totals['scale_total'], 1),
        'length_accuracy': totals['length_correct'] / max(totals['length_total'], 1),
        'copied_number_consistency': totals['copy_correct'] / max(totals['copy_total'], 1),
        'first_wrong_digit_position': (
            float(np.mean(totals['first_wrong_positions']))
            if totals['first_wrong_positions'] else None
        ),
        'n_samples': n,
        'results': results,
    }


def build_generation_comparisons(base_results, adapted_results):
    comparisons = []
    for base_r, adapt_r in zip(base_results, adapted_results):
        comparisons.append({
            'example_index': base_r['example_index'],
            'prompt_text': base_r['prompt_text'],
            'ref_text': base_r['ref_text'],
            'base': {
                'gen_text': base_r['gen_text'],
                'exact_match': base_r['exact_match'],
                'digit_accuracy': base_r['digit_accuracy'],
                'scale_accuracy': base_r['scale_accuracy'],
                'length_accuracy': base_r['length_accuracy'],
                'copied_number_consistency': base_r['copied_number_consistency'],
                'first_wrong_digit_position': base_r['first_wrong_digit_position'],
            },
            'adapted': {
                'gen_text': adapt_r['gen_text'],
                'exact_match': adapt_r['exact_match'],
                'digit_accuracy': adapt_r['digit_accuracy'],
                'scale_accuracy': adapt_r['scale_accuracy'],
                'length_accuracy': adapt_r['length_accuracy'],
                'copied_number_consistency': adapt_r['copied_number_consistency'],
                'first_wrong_digit_position': adapt_r['first_wrong_digit_position'],
                'generated_numeric_components': adapt_r['generated_numeric_components'],
            },
        })
    return comparisons


def print_generation_metrics(gen):
    print(f"  Full exact match:         {gen['full_exact_match']:.4f}")
    print(f"  Number accuracy:          {gen['num_accuracy']:.4f}")
    print(f"  Number MAE:               {gen['mae']:.4f}")
    print(f"  Digit accuracy:           {gen['digit_accuracy']:.4f}")
    print(f"  Scale accuracy:           {gen['scale_accuracy']:.4f}")
    print(f"  Length accuracy:          {gen['length_accuracy']:.4f}")
    print(f"  Copied-number consistency:{gen['copied_number_consistency']:.4f}")
    if gen['first_wrong_digit_position'] is None:
        print("  First wrong digit pos:    n/a")
    else:
        print(f"  First wrong digit pos:    {gen['first_wrong_digit_position']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark base vs surface-adapted models on zero-shot synth arithmetic')
    parser.add_argument('--base_ckpt', required=True)
    parser.add_argument('--adapted_ckpt', required=True)
    parser.add_argument('--base_data_dir', required=True)
    parser.add_argument('--adapted_data_dir', required=True)
    parser.add_argument('--n_forward_batches', type=int, default=200)
    parser.add_argument('--n_gen_samples', type=int, default=3000)
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_path', default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("ZERO-SHOT SYNTH ARITHMETIC BENCHMARK")
    print("=" * 70)

    test_json = os.path.join(args.base_data_dir, 'test_examples.json')
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"Missing synth test examples: {test_json}")
    with open(test_json) as f:
        test_examples = json.load(f)
    print(f"Loaded {len(test_examples)} zero-shot synth test examples from {test_json}")

    results = {}
    configs = [
        ('Base LoRA', args.base_ckpt, args.base_data_dir, False),
        ('Surface Adapted LoRA', args.adapted_ckpt, args.adapted_data_dir, True),
    ]
    for label, ckpt_path, data_dir, use_adapter in configs:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Data: {data_dir}")
        print(f"{'=' * 50}")
        model = load_merged_model(ckpt_path, device=args.device)

        print(f"\n--- Forward-pass metrics ({args.n_forward_batches} batches) ---")
        fwd = evaluate_forward(
            model,
            data_dir,
            args.device,
            block_size=args.block_size,
            batch_size=args.batch_size,
            n_batches=args.n_forward_batches,
        )
        print(f"  Loss:       {fwd['loss']:.4f}")
        print(f"  Perplexity: {fwd['perplexity']:.2f}")
        print(f"  Accuracy:   {fwd['accuracy']:.4f}")

        print(f"\n--- Generation metrics ({args.n_gen_samples} samples, greedy) ---")
        gen = evaluate_generation(
            model,
            test_examples,
            args.device,
            use_adapter=use_adapter,
            max_samples=args.n_gen_samples,
            max_new_tokens=args.max_new_tokens,
        )
        print_generation_metrics(gen)
        results[label] = {'forward': fwd, 'generation': gen}

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    base = results['Base LoRA']['generation']
    adapt = results['Surface Adapted LoRA']['generation']
    summary_rows = [
        ('Full Exact Match', base['full_exact_match'], adapt['full_exact_match']),
        ('Number Accuracy', base['num_accuracy'], adapt['num_accuracy']),
        ('Digit Accuracy', base['digit_accuracy'], adapt['digit_accuracy']),
        ('Scale Accuracy', base['scale_accuracy'], adapt['scale_accuracy']),
        ('Length Accuracy', base['length_accuracy'], adapt['length_accuracy']),
        ('Copy Consistency', base['copied_number_consistency'], adapt['copied_number_consistency']),
    ]
    for name, b_val, a_val in summary_rows:
        print(f"{name:<24} base={b_val:.4f} adapted={a_val:.4f} delta={a_val - b_val:+.4f}")
    print("=" * 70)

    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.adapted_ckpt), 'surface_synth_benchmark.json')
    serializable = {
        'Base LoRA': {
            'forward': results['Base LoRA']['forward'],
            'generation': results['Base LoRA']['generation'],
        },
        'Surface Adapted LoRA': {
            'forward': results['Surface Adapted LoRA']['forward'],
            'generation': results['Surface Adapted LoRA']['generation'],
        },
        'paired_generation_examples': build_generation_comparisons(
            results['Base LoRA']['generation']['results'],
            results['Surface Adapted LoRA']['generation']['results'],
        ),
    }
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.out_path}")


if __name__ == '__main__':
    main()
