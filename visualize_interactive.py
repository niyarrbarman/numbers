"""
Interactive Number Embedding Visualization
============================================
Loads a model checkpoint, encodes numbers, and produces interactive
Plotly plots where you can hover over points to see their values.

Usage:
  python3 visualize_interactive.py --checkpoint /path/to/model.pt
  python3 visualize_interactive.py --range 5000 --n 3000
  python3 visualize_interactive.py  # defaults: U(-1000,1000), 2000 samples

Requires: torch, numpy, plotly, umap-learn
"""

import os
import sys
import argparse
import glob

import torch
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from umap import UMAP
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_emb_torch import NumberEmbeddingSystem

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def find_latest_checkpoint():
    """Find the most recent model .pt file."""
    pattern = os.path.join(CHECKPOINT_DIR, "*_model.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"No checkpoint found in {CHECKPOINT_DIR}. Provide --checkpoint path.")
        sys.exit(1)
    return files[-1]


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    embedding_dim = ckpt.get('embedding_dim', 128)
    system = NumberEmbeddingSystem(embedding_dim=embedding_dim, device=device)
    system.load_state_dict(ckpt['state_dict'])
    system.eval()
    num_steps = ckpt.get('num_steps', '?')
    print(f"Loaded: {checkpoint_path}")
    print(f"  Embedding dim: {embedding_dim}, trained for {num_steps} steps")
    return system


def generate_numbers(range_max, n_samples, seed=42):
    rng = np.random.RandomState(seed)
    numbers = rng.uniform(-range_max, range_max, size=n_samples).astype(np.float32)
    numbers.sort()
    print(f"  Generated {n_samples} numbers from U(-{range_max}, {range_max})")
    return numbers


def encode_numbers(system, numbers):
    with torch.no_grad():
        x = torch.tensor(numbers, device=system.device)
        emb, recon, _ = system.forward(x)
        embeddings = emb.cpu().numpy()
        reconstructions = recon.cpu().numpy()
    abs_err = np.abs(numbers - reconstructions)
    rel_err = abs_err / (np.abs(numbers) + 1e-8)
    print(f"  Encoded {len(numbers)} numbers -> {embeddings.shape}")
    print(f"  Reconstruction: mean rel err {rel_err.mean():.4%}, max {rel_err.max():.4%}")
    return embeddings, reconstructions


def make_hover_text(numbers, reconstructions):
    """Build hover labels showing number, reconstruction, and error."""
    texts = []
    for n, r in zip(numbers, reconstructions):
        err = abs(n - r)
        rel = err / (abs(n) + 1e-8)
        texts.append(
            f"Value: {n:.6g}<br>"
            f"Decoded: {r:.6g}<br>"
            f"Abs err: {err:.4g}<br>"
            f"Rel err: {rel:.4%}"
        )
    return texts


def plot_interactive(coords_2d, numbers, reconstructions, method_name, output_path):
    """Create a multi-tab interactive plot."""
    hover = make_hover_text(numbers, reconstructions)
    signed_log = np.sign(numbers) * np.log1p(np.abs(numbers))
    log_mag = np.log1p(np.abs(numbers))
    sign_val = np.sign(numbers)
    abs_err = np.abs(numbers - reconstructions)
    rel_err = abs_err / (np.abs(numbers) + 1e-8)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{method_name} — Colored by Value",
            f"{method_name} — Colored by Magnitude",
            f"{method_name} — Colored by Sign",
            f"{method_name} — Colored by Reconstruction Error",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Panel 1: Value
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers',
        marker=dict(size=4, color=signed_log, colorscale='RdBu_r',
                    colorbar=dict(title='sign·log(|x|+1)', x=0.45, len=0.4, y=0.8),
                    cmid=0),
        text=hover, hoverinfo='text', name='Value',
    ), row=1, col=1)

    # Panel 2: Magnitude
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers',
        marker=dict(size=4, color=log_mag, colorscale='Viridis',
                    colorbar=dict(title='log(|x|+1)', x=1.0, len=0.4, y=0.8)),
        text=hover, hoverinfo='text', name='Magnitude',
    ), row=1, col=2)

    # Panel 3: Sign
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers',
        marker=dict(size=4, color=sign_val, colorscale='RdBu_r',
                    cmin=-1.5, cmax=1.5,
                    colorbar=dict(title='sign', x=0.45, len=0.4, y=0.2,
                                  tickvals=[-1, 0, 1],
                                  ticktext=['neg', 'zero', 'pos'])),
        text=hover, hoverinfo='text', name='Sign',
    ), row=2, col=1)

    # Panel 4: Error
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers',
        marker=dict(size=4, color=np.clip(rel_err, 0, 0.1), colorscale='Hot_r',
                    cmin=0, cmax=0.1,
                    colorbar=dict(title='Rel Error', x=1.0, len=0.4, y=0.2)),
        text=hover, hoverinfo='text', name='Error',
    ), row=2, col=2)

    fig.update_layout(
        title=f"Number Embeddings — Interactive {method_name}",
        showlegend=False,
        width=1200, height=900,
        template='plotly_white',
    )

    fig.write_html(output_path, include_plotlyjs=True)
    print(f"  Saved: {output_path}")


def plot_pca_interactive(coords_2d, numbers, reconstructions, variance, output_path):
    """Interactive PCA plot."""
    hover = make_hover_text(numbers, reconstructions)
    signed_log = np.sign(numbers) * np.log1p(np.abs(numbers))
    log_mag = np.log1p(np.abs(numbers))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"PCA — Colored by Value (PC1: {variance[0]:.1%}, PC2: {variance[1]:.1%})",
            f"PCA — Colored by Magnitude",
        ],
        horizontal_spacing=0.1,
    )

    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers',
        marker=dict(size=4, color=signed_log, colorscale='RdBu_r',
                    colorbar=dict(title='sign·log(|x|+1)', x=0.45), cmid=0),
        text=hover, hoverinfo='text', name='Value',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers',
        marker=dict(size=4, color=log_mag, colorscale='Viridis',
                    colorbar=dict(title='log(|x|+1)', x=1.0)),
        text=hover, hoverinfo='text', name='Magnitude',
    ), row=1, col=2)

    fig.update_layout(
        title="Number Embeddings — Interactive PCA",
        showlegend=False,
        width=1200, height=500,
        template='plotly_white',
    )
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_yaxes(title_text="PC2", row=1, col=2)

    fig.write_html(output_path, include_plotlyjs=True)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive number embedding visualization")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model .pt checkpoint")
    parser.add_argument("--range", type=float, default=1000.0, dest="range_max",
                        help="Sample from U(-RANGE, RANGE) (default: 1000)")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of samples (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Output directory (default: plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    system = load_model(ckpt_path, device)
    print()

    # Generate and encode
    numbers = generate_numbers(args.range_max, args.n, args.seed)
    embeddings, reconstructions = encode_numbers(system, numbers)
    print()

    # UMAP
    print("  Projecting with UMAP...")
    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                metric='cosine', random_state=42)
    coords_umap = umap.fit_transform(embeddings)
    plot_interactive(coords_umap, numbers, reconstructions, "UMAP",
                     os.path.join(args.output_dir, "interactive_umap.html"))
    print()

    # PCA
    print("  Projecting with PCA...")
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(embeddings)
    variance = pca.explained_variance_ratio_
    print(f"  PCA variance: PC1={variance[0]:.1%}, PC2={variance[1]:.1%}")
    plot_pca_interactive(coords_pca, numbers, reconstructions, variance,
                         os.path.join(args.output_dir, "interactive_pca.html"))
    print()

    print(f"Open in browser:")
    for f in ["interactive_umap.html", "interactive_pca.html"]:
        print(f"  file://{os.path.abspath(os.path.join(args.output_dir, f))}")


if __name__ == "__main__":
    main()
