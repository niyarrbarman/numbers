"""
Standalone autoencoder test: number → NumberEncoder → adapter → num_head → number

Tests whether the adapter + num_head pathway can round-trip numbers
WITHOUT any transformer involvement. If this fails, the pathway
itself is broken and no amount of transformer training will help.

Usage:
  python test_autoencoder.py                              # default range 1000
  python test_autoencoder.py --num-range 100000           # larger range
  python test_autoencoder.py --checkpoint path/to/model.pt  # use pretrained encoder
  python test_autoencoder.py --mode bins                  # test bins mode
  python test_autoencoder.py --mode regression            # test regression mode
"""

import sys
import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root for NumberEncoder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from np_emb_torch import NumberEncoder

from model import NumberOutputHead, build_log_bin_edges


def sample_numbers(batch_size, num_range, device):
    """Sample numbers from log-uniform distribution over [-num_range, num_range]."""
    # Log-uniform magnitude
    log_max = math.log(num_range + 1)
    log_vals = torch.rand(batch_size, device=device) * log_max
    magnitude = torch.exp(log_vals) - 1.0
    # Random sign
    signs = torch.sign(torch.randn(batch_size, device=device))
    signs[signs == 0] = 1.0
    return signs * magnitude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', help='NumberEncoder checkpoint')
    parser.add_argument('--num-range', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='both', choices=['bins', 'regression', 'both'])
    parser.add_argument('--n-embd', type=int, default=256, help='Adapter output dim (match GPT config)')
    parser.add_argument('--num-emb-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-bins', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    modes = ['bins', 'regression'] if args.mode == 'both' else [args.mode]

    # Load frozen NumberEncoder
    encoder = NumberEncoder(embedding_dim=args.num_emb_dim).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        enc_state = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('encoder.'):
                enc_state[k[len('encoder.'):]] = v
        encoder.load_state_dict(enc_state)
        print(f"Loaded encoder from {args.checkpoint}")
    else:
        print("Using randomly initialized encoder")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Testing mode: {mode} | range: [-{args.num_range}, {args.num_range}]")
        print(f"{'=' * 60}")

        # Build adapter (same as GPT model)
        adapter = nn.Sequential(
            nn.Linear(args.num_emb_dim, args.n_embd),
            nn.GELU(),
            nn.Linear(args.n_embd, args.n_embd),
        ).to(device)

        # Build num_head (input = adapter output only, no transformer hidden)
        # Testing the pathway without skip connection first: input_dim = n_embd
        num_head = NumberOutputHead(
            input_dim=args.n_embd,
            hidden_dim=args.hidden_dim,
            mode=mode,
            n_bins=args.n_bins,
            num_range=args.num_range,
        ).to(device)

        params = list(adapter.parameters()) + list(num_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        n_params = sum(p.numel() for p in params)
        print(f"Trainable params: {n_params:,}")

        for step in range(args.steps):
            # Sample random numbers
            values = sample_numbers(args.batch_size, args.num_range, device)

            # Forward: number → encoder → adapter → num_head
            with torch.no_grad():
                emb = encoder(values)           # (B, 128)
            projected = adapter(emb)             # (B, n_embd)
            output = num_head(projected)         # (B, n_bins) or (B,)

            # Loss
            if mode == 'bins':
                targets = num_head.value_to_bin(values)  # (B,)
                loss = F.cross_entropy(output, targets)
            else:
                slog_target = NumberOutputHead.to_signed_log(values)
                loss = F.mse_loss(output, slog_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.eval_interval == 0 or step == args.steps - 1:
                # Evaluate accuracy
                with torch.no_grad():
                    eval_vals = sample_numbers(2048, args.num_range, device)
                    eval_emb = encoder(eval_vals)
                    eval_proj = adapter(eval_emb)
                    eval_out = num_head(eval_proj)

                    if mode == 'bins':
                        pred_bins = eval_out.argmax(dim=-1)
                        true_bins = num_head.value_to_bin(eval_vals)
                        exact_acc = (pred_bins == true_bins).float().mean().item()
                        near_acc = ((pred_bins - true_bins).abs() <= 2).float().mean().item()
                        # Also compute value-level error
                        pred_vals = num_head.bin_to_value(pred_bins)
                        abs_err = (pred_vals - eval_vals).abs()
                        rel_err = abs_err / (eval_vals.abs() + 1.0)
                        print(f"  step {step:5d} | loss {loss.item():.4f} | "
                              f"exact_bin {exact_acc:.3f} | near_bin(±2) {near_acc:.3f} | "
                              f"median_abs_err {abs_err.median().item():.2f} | "
                              f"median_rel_err {rel_err.median().item():.3f}")
                    else:
                        slog_pred = eval_out
                        slog_true = NumberOutputHead.to_signed_log(eval_vals)
                        mse = F.mse_loss(slog_pred, slog_true).item()
                        pred_vals = NumberOutputHead.from_signed_log(slog_pred)
                        abs_err = (pred_vals - eval_vals).abs()
                        rel_err = abs_err / (eval_vals.abs() + 1.0)
                        print(f"  step {step:5d} | loss(MSE) {mse:.4f} | "
                              f"median_abs_err {abs_err.median().item():.2f} | "
                              f"median_rel_err {rel_err.median().item():.3f}")

        # Final detailed eval
        print(f"\n--- Final eval ({mode}) ---")
        with torch.no_grad():
            # Test specific values
            test_vals = torch.tensor(
                [0.0, 1.0, -1.0, 5.0, 10.0, 42.0, 100.0, -100.0,
                 500.0, -500.0, 999.0, -999.0, 0.5, -0.1, 3.14],
                device=device)
            test_emb = encoder(test_vals)
            test_proj = adapter(test_emb)
            test_out = num_head(test_proj)

            if mode == 'bins':
                pred_bins = test_out.argmax(dim=-1)
                pred_vals = num_head.bin_to_value(pred_bins)
            else:
                pred_vals = NumberOutputHead.from_signed_log(test_out)

            print(f"  {'True':>10s} | {'Predicted':>10s} | {'Error':>10s}")
            print(f"  {'-' * 36}")
            for i in range(len(test_vals)):
                true = test_vals[i].item()
                pred = pred_vals[i].item()
                err = abs(pred - true)
                print(f"  {true:10.2f} | {pred:10.2f} | {err:10.2f}")


if __name__ == '__main__':
    main()
