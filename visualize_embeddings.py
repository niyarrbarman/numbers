"""
Number Embedding Visualization
================================
Loads saved reference embeddings and produces UMAP/t-SNE + PCA plots.

Usage:
  python3 visualize_embeddings.py --embeddings /path/to/np_emb_v8_500k_embeddings.npz
  python3 visualize_embeddings.py  # auto-finds latest checkpoint

Outputs saved to: /tmpdir/m24047brmn/numbers/plots/
"""

import os
import sys
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

# Try UMAP first, fall back to t-SNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from sklearn.manifold import TSNE


PLOT_DIR = "/tmpdir/m24047brmn/numbers/plots"
CHECKPOINT_DIR = "/tmpdir/m24047brmn/numbers/checkpoints"


def find_latest_embeddings():
    """Find the most recent embeddings .npz file."""
    pattern = os.path.join(CHECKPOINT_DIR, "*_embeddings.npz")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"No embedding files found in {CHECKPOINT_DIR}")
        sys.exit(1)
    return files[-1]


def load_embeddings(path):
    """Load saved reference embeddings."""
    data = np.load(path)
    print(f"Loaded: {path}")
    print(f"  Numbers: {data['numbers'].shape}")
    print(f"  Embeddings: {data['embeddings'].shape}")
    print(f"  Reconstructions: {data['reconstructions'].shape}")
    return data['numbers'], data['embeddings'], data['reconstructions']


def project_2d(embeddings, method="umap"):
    """Project embeddings to 2D."""
    if method == "umap" and HAS_UMAP:
        print("  Projecting with UMAP...")
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                       metric='cosine', random_state=42)
        return reducer.fit_transform(embeddings), "UMAP"
    else:
        if method == "umap":
            print("  UMAP not available, using t-SNE...")
        else:
            print("  Projecting with t-SNE...")
        reducer = TSNE(n_components=2, perplexity=50, learning_rate='auto',
                       init='pca', random_state=42, n_iter=1000)
        return reducer.fit_transform(embeddings), "t-SNE"


def project_pca(embeddings):
    """Project embeddings to 2D via PCA."""
    print("  Projecting with PCA...")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}")
    return proj, var


def plot_nonlinear(coords_2d, numbers, reconstructions, method_name, tag, plot_dir):
    """3-panel plot colored by value, magnitude, sign."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Number Embeddings — {method_name} Projection", fontsize=14, y=1.02)

    # --- Panel 1: Color by signed log value ---
    ax = axes[0]
    signed_log = np.sign(numbers) * np.log1p(np.abs(numbers))
    vmax = np.percentile(np.abs(signed_log), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=signed_log,
                    cmap='RdBu_r', norm=norm, s=4, alpha=0.7)
    fig.colorbar(sc, ax=ax, label='sign(x) · log(|x|+1)')
    ax.set_title("Colored by Value")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")

    # --- Panel 2: Color by log magnitude ---
    ax = axes[1]
    log_mag = np.log1p(np.abs(numbers))
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=log_mag,
                    cmap='viridis', s=4, alpha=0.7)
    fig.colorbar(sc, ax=ax, label='log(|x|+1)')
    ax.set_title("Colored by Magnitude")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")

    # --- Panel 3: Color by sign ---
    ax = axes[2]
    sign_color = np.where(numbers > 0, 1.0, np.where(numbers < 0, -1.0, 0.0))
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=sign_color,
                    cmap='RdBu_r', vmin=-1.5, vmax=1.5, s=4, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax, label='sign', ticks=[-1, 0, 1])
    cb.set_ticklabels(['negative', 'zero', 'positive'])
    ax.set_title("Colored by Sign")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"embeddings_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pca(coords_2d, numbers, variance, tag, plot_dir):
    """2-panel PCA plot: colored by value and by magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Number Embeddings — PCA (PC1: {variance[0]:.1%}, PC2: {variance[1]:.1%})",
                 fontsize=14, y=1.02)

    # Panel 1: Color by signed log value
    ax = axes[0]
    signed_log = np.sign(numbers) * np.log1p(np.abs(numbers))
    vmax = np.percentile(np.abs(signed_log), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=signed_log,
                    cmap='RdBu_r', norm=norm, s=4, alpha=0.7)
    fig.colorbar(sc, ax=ax, label='sign(x) · log(|x|+1)')
    ax.set_title("Colored by Value")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Panel 2: Color by log magnitude
    ax = axes[1]
    log_mag = np.log1p(np.abs(numbers))
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=log_mag,
                    cmap='viridis', s=4, alpha=0.7)
    fig.colorbar(sc, ax=ax, label='log(|x|+1)')
    ax.set_title("Colored by Magnitude")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"embeddings_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reconstruction_error(coords_2d, numbers, reconstructions, method_name, tag, plot_dir):
    """Reconstruction error overlay plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    abs_err = np.abs(numbers - reconstructions)
    rel_err = abs_err / (np.abs(numbers) + 1e-8)
    # Cap relative error for visualization
    rel_err_capped = np.clip(rel_err, 0, 1.0)

    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=rel_err_capped,
                    cmap='hot_r', s=4, alpha=0.7, vmin=0, vmax=0.1)
    fig.colorbar(sc, ax=ax, label='Relative Error (capped at 10%)')
    ax.set_title(f"Reconstruction Error — {method_name}")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")

    plt.tight_layout()
    path = os.path.join(plot_dir, f"embeddings_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize number embeddings")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to embeddings .npz file (auto-detects if not given)")
    parser.add_argument("--plot-dir", type=str, default=PLOT_DIR,
                        help=f"Output directory for plots (default: {PLOT_DIR})")
    args = parser.parse_args()

    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    emb_path = args.embeddings or find_latest_embeddings()
    numbers, embeddings, reconstructions = load_embeddings(emb_path)

    print()

    # Non-linear projection (UMAP or t-SNE)
    coords_nl, method_name = project_2d(embeddings, method="umap")
    nl_tag = method_name.lower().replace("-", "")
    plot_nonlinear(coords_nl, numbers, reconstructions, method_name, nl_tag, plot_dir)
    plot_reconstruction_error(coords_nl, numbers, reconstructions, method_name,
                              f"{nl_tag}_error", plot_dir)
    print()

    # PCA projection
    coords_pca, variance = project_pca(embeddings)
    plot_pca(coords_pca, numbers, variance, "pca", plot_dir)
    plot_reconstruction_error(coords_pca, numbers, reconstructions, "PCA",
                              "pca_error", plot_dir)
    print()

    print(f"All plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
