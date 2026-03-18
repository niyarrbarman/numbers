"""Benchmark augmented (NumLM) vs baseline (vanilla Qwen) models.

Evaluates on:
  1. GSM8K test set — generate answer, extract final number, compare to gold
  2. Synthetic arithmetic — exact match on numeric outputs

Usage:
  python3 benchmark.py \
    --augmented_ckpt /path/to/s2_augmented/stage2_epoch3.pt \
    --baseline_ckpt /path/to/s2_baseline/baseline_epoch3.pt \
    --model_path /path/to/Qwen2.5-0.5B-Instruct \
    --gsm8k_test /path/to/s2_data/augmented/test.jsonl \
    --gsm8k_test_baseline /path/to/s2_data/baseline/test.jsonl \
    --s1_test /path/to/s1_data/test.jsonl \
    --device cuda
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# project root for num_analytic.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# 124M/fe_adapt for numeric_surface.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '124M', 'fe_adapt'))

from numeric_surface import (
    render_surface_components,
    surface_components_from_value,
    surface_components_to_row,
)

from main_qwen import NumLM, setup_tokenizer, NUM_TOKEN

NUM_PATTERN = re.compile(r'(?<![.\d])\d+(?:\.\d+)?(?![.\d])')


# =============================================================================
# Model loading
# =============================================================================

def load_augmented_model(model_path, checkpoint_path, device):
    """Load NumLM with checkpoint weights."""
    tokenizer = setup_tokenizer(model_path)
    model = NumLM(model_path, dtype=torch.float32)
    model.model.resize_token_embeddings(len(tokenizer))
    model.num_token_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # handle embedding size mismatch
    for key in ["model.model.embed_tokens.weight", "model.lm_head.weight"]:
        if key in state:
            cur_size = getattr(model.model.model, "embed_tokens" if "embed" in key else "ERROR", None)
            if "lm_head" in key:
                cur_vocab = model.model.lm_head.weight.shape[0]
            else:
                cur_vocab = model.model.model.embed_tokens.weight.shape[0]
            ckpt_vocab = state[key].shape[0]
            if ckpt_vocab != cur_vocab:
                if ckpt_vocab < cur_vocab:
                    if "lm_head" in key:
                        padded = model.model.lm_head.weight.data.clone()
                    else:
                        padded = model.model.model.embed_tokens.weight.data.clone()
                    padded[:ckpt_vocab] = state[key]
                    state[key] = padded
                else:
                    state[key] = state[key][:cur_vocab]

    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, tokenizer


def load_baseline_model(model_path, checkpoint_path, device):
    """Load vanilla Qwen with checkpoint weights."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, tokenizer


# =============================================================================
# Augmented model generation
# =============================================================================

def prepare_augmented_prompt(prompt_text, tokenizer, device, max_digits=32):
    """Tokenize a prompt with <NUM> tokens and extract numeric metadata."""
    num_token_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)

    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # find <NUM> positions and extract values from the original sample
    num_positions_list = (input_ids[0] == num_token_id).nonzero(as_tuple=True)[0].tolist()

    return input_ids, num_positions_list


@torch.no_grad()
def generate_augmented(model, tokenizer, sample, device, max_new_tokens=256):
    """Generate with the augmented NumLM model."""
    prompt = sample["prompt"]

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    num_token_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)

    # find <NUM> positions
    num_positions_list = (input_ids[0] == num_token_id).nonzero(as_tuple=True)[0].tolist()

    # extract input number values (only non-output ones go in the prompt)
    num_values = sample.get("num_values", [])
    num_is_output = sample.get("num_is_output", [])

    input_vals = [v for v, is_out in zip(num_values, num_is_output) if not is_out]
    n_found = min(len(num_positions_list), len(input_vals))

    # build tensors (must be same size)
    if n_found > 0:
        nv = torch.tensor([input_vals[:n_found]], dtype=torch.float32, device=device)
        np_tensor = torch.tensor([num_positions_list[:n_found]], dtype=torch.long, device=device)
    else:
        nv = torch.zeros(1, 0, device=device)
        np_tensor = torch.full((1, 0), -1, dtype=torch.long, device=device)

    gen_ids, generated_numbers = model.generate_with_numbers(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        num_values=nv,
        num_positions=np_tensor,
        eos_token_id=tokenizer.eos_token_id,
    )

    # render output text, replacing <NUM> with decoded numbers
    output_ids = gen_ids[0, input_ids.size(1):].tolist()
    num_by_pos = {pos: rendered for pos, rendered, _ in generated_numbers}

    parts = []
    for i, tok in enumerate(output_ids):
        abs_pos = input_ids.size(1) + i
        if tok == tokenizer.eos_token_id:
            break
        if tok == num_token_id:
            parts.append(num_by_pos.get(abs_pos, "<NUM>"))
        else:
            parts.append(tokenizer.decode([tok]))

    return "".join(parts).strip()


# =============================================================================
# Baseline model generation
# =============================================================================

@torch.no_grad()
def generate_baseline(model, tokenizer, sample, device, max_new_tokens=256):
    """Generate with the baseline vanilla Qwen model."""
    prompt = sample["prompt"]

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.size(1)

    generated = []
    past_key_values = None
    cur_input = input_ids

    for _ in range(max_new_tokens):
        outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        if next_tok.item() == tokenizer.eos_token_id:
            break
        generated.append(next_tok.item())
        cur_input = next_tok

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# =============================================================================
# Answer extraction
# =============================================================================

def extract_final_number(text):
    """Extract the final number from generated text."""
    # look for "The answer is X." pattern first
    m = re.search(r'[Tt]he answer is\s+([+-]?\d+(?:\.\d+)?)', text)
    if m:
        return m.group(1)

    # fall back to last number in text
    numbers = NUM_PATTERN.findall(text)
    if numbers:
        return numbers[-1]
    return None


def numbers_match(pred, gold):
    """Check if two number strings represent the same value."""
    if pred is None or gold is None:
        return False
    try:
        p = float(pred.replace(",", ""))
        g = float(gold.replace(",", ""))
        if g == 0:
            return p == 0
        return abs(p - g) / max(abs(g), 1e-10) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


# =============================================================================
# Evaluation routines
# =============================================================================

def evaluate_gsm8k(model, tokenizer, test_data, device, generate_fn,
                   max_samples=None, max_new_tokens=256, label=""):
    """Evaluate on GSM8K-style test data."""
    correct = 0
    total = 0
    results = []

    samples = test_data[:max_samples] if max_samples else test_data

    t0 = time.time()
    for i, sample in enumerate(samples):
        response = sample.get("response", "")

        # extract gold answer
        gold_answer = extract_final_number(response)
        if gold_answer is None:
            continue

        # generate
        gen_text = generate_fn(model, tokenizer, sample, device,
                               max_new_tokens=max_new_tokens)
        pred_answer = extract_final_number(gen_text)
        match = numbers_match(pred_answer, gold_answer)

        if match:
            correct += 1
        total += 1

        results.append({
            "index": i,
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": match,
            "gen_text": gen_text[:300],
        })

        if i < 5:
            print(f"  [{i}] gold={gold_answer} pred={pred_answer} {'OK' if match else 'WRONG'}")
            print(f"       gen: {gen_text[:120]}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {label} {i+1}/{len(samples)}  acc={correct/total:.4f}  ({elapsed:.0f}s)")

    accuracy = correct / max(total, 1)
    elapsed = time.time() - t0
    print(f"  {label} GSM8K: {correct}/{total} = {accuracy:.4f}  ({elapsed:.1f}s)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_synth(model, tokenizer, test_data, device, generate_fn,
                   max_samples=None, max_new_tokens=64, label=""):
    """Evaluate on synthetic arithmetic test data."""
    exact_match = 0
    number_correct = 0
    number_total = 0
    total = 0
    results = []

    samples = test_data[:max_samples] if max_samples else test_data

    t0 = time.time()
    for i, sample in enumerate(samples):
        response = sample.get("response", "")

        # extract gold numbers from response
        gold_numbers = NUM_PATTERN.findall(response) if "num_values" not in sample else None

        # for augmented data, gold answer is the output numbers
        if "num_values" in sample and "num_is_output" in sample:
            gold_values = [
                v for v, is_out in zip(sample["num_values"], sample["num_is_output"])
                if is_out
            ]
        else:
            gold_values = [float(n) for n in NUM_PATTERN.findall(response)] if response else []

        gen_text = generate_fn(model, tokenizer, sample, device,
                               max_new_tokens=max_new_tokens)

        # extract generated numbers
        gen_numbers = NUM_PATTERN.findall(gen_text)

        # check exact text match
        is_exact = gen_text.strip() == response.strip()
        if is_exact:
            exact_match += 1

        # check number accuracy
        if gold_values:
            gen_floats = []
            for n in gen_numbers:
                try:
                    gen_floats.append(float(n))
                except ValueError:
                    pass

            for j, gv in enumerate(gold_values):
                number_total += 1
                if j < len(gen_floats) and numbers_match(str(gen_floats[j]), str(gv)):
                    number_correct += 1

        total += 1

        if i < 5:
            print(f"  [{i}] ref: {response}")
            print(f"       gen: {gen_text}")
            print(f"       exact={is_exact}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {label} {i+1}/{len(samples)}  exact={exact_match/total:.4f}"
                  f"  num_acc={number_correct/max(number_total,1):.4f}  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    acc = exact_match / max(total, 1)
    num_acc = number_correct / max(number_total, 1)
    print(f"  {label} Synth: exact={exact_match}/{total}={acc:.4f}"
          f"  num_acc={number_correct}/{number_total}={num_acc:.4f}  ({elapsed:.1f}s)")
    return {
        "exact_match": acc,
        "number_accuracy": num_acc,
        "exact_correct": exact_match,
        "number_correct": number_correct,
        "number_total": number_total,
        "total": total,
        "results": results,
    }


# =============================================================================
# Main
# =============================================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark augmented vs baseline")
    parser.add_argument("--augmented_ckpt", type=str, required=True,
                        help="Path to S2 augmented checkpoint")
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="Path to S2 baseline checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Qwen model")
    parser.add_argument("--gsm8k_test", type=str, default=None,
                        help="Augmented GSM8K test JSONL")
    parser.add_argument("--gsm8k_test_baseline", type=str, default=None,
                        help="Baseline GSM8K test JSONL")
    parser.add_argument("--s1_test", type=str, default=None,
                        help="S1 synthetic test JSONL")
    parser.add_argument("--max_gsm8k", type=int, default=None,
                        help="Max GSM8K samples (default: all)")
    parser.add_argument("--max_synth", type=int, default=3000,
                        help="Max synthetic samples")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("BENCHMARK: Augmented (NumLM) vs Baseline (Vanilla Qwen)")
    print("=" * 70)

    all_results = {}

    # ---- Augmented model ----
    print(f"\nLoading augmented model: {args.augmented_ckpt}")
    aug_model, aug_tokenizer = load_augmented_model(
        args.model_path, args.augmented_ckpt, args.device)
    total, trainable = aug_model.param_count()
    print(f"  params: {total:,} total, {trainable:,} trainable")

    aug_results = {}

    if args.gsm8k_test:
        print(f"\n--- Augmented: GSM8K ({args.gsm8k_test}) ---")
        gsm8k_aug = load_jsonl(args.gsm8k_test)
        aug_results["gsm8k"] = evaluate_gsm8k(
            aug_model, aug_tokenizer, gsm8k_aug, args.device,
            generate_augmented, max_samples=args.max_gsm8k,
            max_new_tokens=args.max_new_tokens, label="AUG")

    if args.s1_test:
        print(f"\n--- Augmented: Synthetic ({args.s1_test}) ---")
        synth_data = load_jsonl(args.s1_test)
        aug_results["synth"] = evaluate_synth(
            aug_model, aug_tokenizer, synth_data, args.device,
            generate_augmented, max_samples=args.max_synth,
            max_new_tokens=64, label="AUG")

    all_results["augmented"] = aug_results

    # free memory
    del aug_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Baseline model ----
    print(f"\nLoading baseline model: {args.baseline_ckpt}")
    base_model, base_tokenizer = load_baseline_model(
        args.model_path, args.baseline_ckpt, args.device)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  params: {n_params:,}")

    base_results = {}

    if args.gsm8k_test_baseline:
        print(f"\n--- Baseline: GSM8K ({args.gsm8k_test_baseline}) ---")
        gsm8k_base = load_jsonl(args.gsm8k_test_baseline)
        base_results["gsm8k"] = evaluate_gsm8k(
            base_model, base_tokenizer, gsm8k_base, args.device,
            generate_baseline, max_samples=args.max_gsm8k,
            max_new_tokens=args.max_new_tokens, label="BASE")

    if args.s1_test:
        print(f"\n--- Baseline: Synthetic ({args.s1_test}) ---")
        # for baseline, need plain-text version of synth data
        synth_data = load_jsonl(args.s1_test)
        # convert augmented format to baseline format for generation
        synth_baseline = []
        for s in synth_data:
            prompt = s["prompt"]
            response = s["response"]
            values = s.get("num_values", [])
            is_output = s.get("num_is_output", [])

            # replace <NUM> tokens with actual values, respecting input/output split
            plain_prompt = prompt
            plain_response = response
            for val, is_out in zip(values, is_output):
                v_str = str(int(val)) if val == int(val) else str(val)
                if not is_out:
                    plain_prompt = plain_prompt.replace("<NUM>", v_str, 1)
                else:
                    plain_response = plain_response.replace("<NUM>", v_str, 1)

            synth_baseline.append({"prompt": plain_prompt, "response": plain_response})

        base_results["synth"] = evaluate_synth(
            base_model, base_tokenizer, synth_baseline, args.device,
            generate_baseline, max_samples=args.max_synth,
            max_new_tokens=64, label="BASE")

    all_results["baseline"] = base_results

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<28} {'Augmented':>12} {'Baseline':>12} {'Delta':>12}")
    print("-" * 70)

    rows = []
    if "gsm8k" in aug_results and "gsm8k" in base_results:
        a = aug_results["gsm8k"]["accuracy"]
        b = base_results["gsm8k"]["accuracy"]
        rows.append(("GSM8K Accuracy", a, b))

    if "synth" in aug_results and "synth" in base_results:
        a = aug_results["synth"]["exact_match"]
        b = base_results["synth"]["exact_match"]
        rows.append(("Synth Exact Match", a, b))
        a = aug_results["synth"]["number_accuracy"]
        b = base_results["synth"]["number_accuracy"]
        rows.append(("Synth Number Accuracy", a, b))

    for name, a_val, b_val in rows:
        delta = a_val - b_val
        print(f"{name:<28} {a_val:>11.4f} {b_val:>11.4f} {delta:>+11.4f}")
    print("=" * 70)

    # ---- Save results ----
    if args.out_path is None:
        args.out_path = os.path.join(
            os.path.dirname(args.augmented_ckpt), "benchmark_results.json")

    # strip per-sample results to keep file small
    save_results = {}
    for model_name, res in all_results.items():
        save_results[model_name] = {}
        for task_name, task_res in res.items():
            save_copy = {k: v for k, v in task_res.items() if k != "results"}
            save_results[model_name][task_name] = save_copy

    save_results["summary"] = rows

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {args.out_path}")


if __name__ == "__main__":
    main()
