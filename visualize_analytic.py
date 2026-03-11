"""
Analytic Number Embedding Explorer
====================================
Interactive 2D/3D UMAP plot of analytic (non-learned) number embeddings.

Uses AnalyticNumberCodec from num_analytic.py — no trained model weights needed.
Click any two points to see the full-dimensional L2 distance,
cosine similarity, and both numbers' details.

Usage:
  python3 visualize_analytic.py
  python3 visualize_analytic.py --range-min -500 --range-max 500 --n 1000
  python3 visualize_analytic.py --dim 2
"""

import os
import sys
import argparse
import json

import numpy as np

import plotly.graph_objects as go
from umap import UMAP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from num_analytic import AnalyticNumberCodec


# ── helpers ──────────────────────────────────────────────────────────────────

def generate_numbers(range_min: float, range_max: float, n_samples: int, seed: int = 42):
    """Generate uniformly spaced numbers in [range_min, range_max]."""
    rng = np.random.RandomState(seed)
    numbers = rng.uniform(range_min, range_max, size=n_samples).astype(np.float64)
    numbers.sort()
    print(f"  Generated {n_samples} numbers from U({range_min}, {range_max})")
    return numbers


def encode_numbers(codec: AnalyticNumberCodec, numbers: np.ndarray):
    """Encode every number and collect the analytic embeddings and round-trip decoded values."""
    embeddings = []
    reconstructions = []
    for n in numbers:
        emb = codec.encode(n)
        embeddings.append(emb)
        comps = codec.decode(emb)
        decoded = float(codec.components_to_decimal(comps))
        reconstructions.append(decoded)
    embeddings = np.array(embeddings, dtype=np.float64)
    reconstructions = np.array(reconstructions, dtype=np.float64)
    return embeddings, reconstructions


# ── interactive HTML JS (same click-to-measure as visualize_explorer.py) ─────

CLICK_JS = """
<div id="info-panel" style="
    position: fixed; top: 10px; right: 10px; z-index: 1000;
    background: rgba(255,255,255,0.95); border: 2px solid #333;
    border-radius: 8px; padding: 15px; min-width: 320px;
    font-family: 'Courier New', monospace; font-size: 13px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
    <div style="font-weight: bold; font-size: 15px; margin-bottom: 8px;">
        Analytic Embedding Distance Calculator
    </div>
    <div id="instructions" style="color: #666;">
        Click on a point to select it.
    </div>
    <div id="point-a" style="display:none; margin-top: 8px; padding: 6px;
         background: #e8f4fd; border-radius: 4px;">
        <b style="color: #1f77b4;">Point A:</b> <span id="pa-val"></span><br>
        <span style="color: #666; font-size: 11px;">Decoded: <span id="pa-dec"></span>
        | Err: <span id="pa-err"></span></span>
    </div>
    <div id="point-b" style="display:none; margin-top: 6px; padding: 6px;
         background: #fde8e8; border-radius: 4px;">
        <b style="color: #d62728;">Point B:</b> <span id="pb-val"></span><br>
        <span style="color: #666; font-size: 11px;">Decoded: <span id="pb-dec"></span>
        | Err: <span id="pb-err"></span></span>
    </div>
    <div id="distance-box" style="display:none; margin-top: 10px; padding: 10px;
         background: #f0f0f0; border-radius: 4px; border-left: 4px solid #2ca02c;">
        <div><b>__EMB_DIM__d L2 distance:</b> <span id="l2-dist" style="color: #2ca02c; font-size: 16px;"></span></div>
        <div style="margin-top: 4px;"><b>Cosine similarity:</b> <span id="cos-sim"></span></div>
        <div style="margin-top: 4px;"><b>Numerical |a − b|:</b> <span id="num-dist"></span></div>
    </div>
    <button id="reset-btn" onclick="resetSelection()" style="
        display:none; margin-top: 10px; padding: 6px 16px;
        background: #d62728; color: white; border: none;
        border-radius: 4px; cursor: pointer; font-size: 13px;
    ">Reset</button>
</div>

<script>
var selectedA = null;
var selectedB = null;
var allEmbeddings = __EMBEDDINGS_JSON__;
var allNumbers = __NUMBERS_JSON__;
var allRecons = __RECONS_JSON__;
var nPoints = __N_POINTS__;
var baseSize = __BASE_SIZE__;
var plotDiv = document.getElementById('plotly-div');

function computeL2(a, b) {
    var sum = 0;
    for (var i = 0; i < a.length; i++) {
        var d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

function computeCosine(a, b) {
    var dot = 0, na = 0, nb = 0;
    for (var i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function fmtNum(x) {
    if (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0))
        return x.toExponential(4);
    return x.toFixed(4);
}

function fmtPct(x) {
    return (x * 100).toFixed(4) + '%';
}

function updateHighlights() {
    var sizes = new Array(nPoints).fill(baseSize);
    if (selectedA !== null) sizes[selectedA.idx] = baseSize * 4;
    if (selectedB !== null) sizes[selectedB.idx] = baseSize * 4;
    Plotly.restyle(plotDiv, {'marker.size': [sizes]}, [0]);
}

function resetSelection() {
    selectedA = null;
    selectedB = null;
    document.getElementById('instructions').style.display = 'block';
    document.getElementById('instructions').textContent = 'Click on a point to select it.';
    document.getElementById('point-a').style.display = 'none';
    document.getElementById('point-b').style.display = 'none';
    document.getElementById('distance-box').style.display = 'none';
    document.getElementById('reset-btn').style.display = 'none';
    updateHighlights();
}

plotDiv.on('plotly_click', function(data) {
    var pt = data.points[0];
    if (pt.curveNumber !== 0) return;

    var idx = pt.pointIndex;
    var num = allNumbers[idx];
    var rec = allRecons[idx];
    var emb = allEmbeddings[idx];
    var err = Math.abs(num - rec) / (Math.abs(num) + 1e-8);

    if (selectedA === null) {
        selectedA = {idx: idx, num: num, rec: rec, emb: emb, err: err};
        document.getElementById('instructions').textContent = 'Now click a second point.';
        document.getElementById('point-a').style.display = 'block';
        document.getElementById('pa-val').textContent = fmtNum(num);
        document.getElementById('pa-dec').textContent = fmtNum(rec);
        document.getElementById('pa-err').textContent = fmtPct(err);
        document.getElementById('reset-btn').style.display = 'inline-block';
        updateHighlights();

    } else if (selectedB === null) {
        selectedB = {idx: idx, num: num, rec: rec, emb: emb, err: err};
        var l2 = computeL2(selectedA.emb, selectedB.emb);
        var cos = computeCosine(selectedA.emb, selectedB.emb);
        var numDist = Math.abs(selectedA.num - selectedB.num);

        document.getElementById('instructions').style.display = 'none';
        document.getElementById('point-b').style.display = 'block';
        document.getElementById('pb-val').textContent = fmtNum(num);
        document.getElementById('pb-dec').textContent = fmtNum(rec);
        document.getElementById('pb-err').textContent = fmtPct(err);

        document.getElementById('distance-box').style.display = 'block';
        document.getElementById('l2-dist').textContent = l2.toFixed(4);
        document.getElementById('cos-sim').textContent = cos.toFixed(6);
        document.getElementById('num-dist').textContent = fmtNum(numDist);
        updateHighlights();

    } else {
        resetSelection();
    }
});
</script>
"""


# ── plot builder ─────────────────────────────────────────────────────────────

def build_explorer(numbers, embeddings, reconstructions, coords, dim, method,
                   output_path, emb_dim):
    """Build interactive HTML with click-to-measure distance panel."""
    signed_log = np.sign(numbers) * np.log1p(np.abs(numbers))

    if dim == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(size=4, color=signed_log, colorscale='RdBu_r',
                        cmid=0, opacity=0.8,
                        colorbar=dict(title='sign·log(|x|+1)')),
            text=[f"{n:.6g}" for n in numbers],
            hovertemplate="<b>%{text}</b><br>"
                          f"{method} 1: " + "%{x:.2f}<br>"
                          f"{method} 2: " + "%{y:.2f}<br>"
                          f"{method} 3: " + "%{z:.2f}<extra></extra>",
            name='Embeddings',
        )])
        fig.update_layout(scene=dict(
            xaxis_title=f'{method} 1',
            yaxis_title=f'{method} 2',
            zaxis_title=f'{method} 3',
        ))
    else:
        fig = go.Figure(data=[go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode='markers',
            marker=dict(size=5, color=signed_log, colorscale='RdBu_r',
                        cmid=0, opacity=0.8,
                        colorbar=dict(title='sign·log(|x|+1)')),
            text=[f"{n:.6g}" for n in numbers],
            hovertemplate="<b>%{text}</b><br>"
                          f"{method} 1: " + "%{x:.2f}<br>"
                          f"{method} 2: " + "%{y:.2f}<extra></extra>",
            name='Embeddings',
        )])
        fig.update_xaxes(title=f'{method} 1')
        fig.update_yaxes(title=f'{method} 2')

    fig.update_layout(
        title=f"Analytic Number Embedding Explorer — {dim}D {method} "
              f"(click two points to measure {emb_dim}d distance)",
        template='plotly_white',
        width=1000, height=750,
        margin=dict(r=350),  # room for info panel
    )

    # Write HTML with embedded JS
    html = fig.to_html(include_plotlyjs=True, div_id='plotly-div', full_html=True)

    # Inject the embeddings data and click handler before </body>
    base_size = 4 if dim == 3 else 5
    js_block = CLICK_JS
    js_block = js_block.replace('__EMB_DIM__', str(emb_dim))
    js_block = js_block.replace('__EMBEDDINGS_JSON__', json.dumps(embeddings.tolist()))
    js_block = js_block.replace('__NUMBERS_JSON__', json.dumps(numbers.tolist()))
    js_block = js_block.replace('__RECONS_JSON__', json.dumps(reconstructions.tolist()))
    js_block = js_block.replace('__N_POINTS__', str(len(numbers)))
    js_block = js_block.replace('__BASE_SIZE__', str(base_size))

    html = html.replace('</body>', js_block + '\n</body>')

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive explorer for analytic number embeddings "
                    "(no trained model needed)")
    parser.add_argument("--range-min", type=float, default=-100.0,
                        help="Lower bound of number range (default: -100)")
    parser.add_argument("--range-max", type=float, default=100.0,
                        help="Upper bound of number range (default: 100)")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of samples (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3],
                        help="Projection dimensions (default: 3)")
    parser.add_argument("--both", action="store_true",
                        help="Generate both 2D and 3D plots")
    parser.add_argument("--output-dir", type=str, default="plots")
    # AnalyticNumberCodec params
    parser.add_argument("--K", type=int, default=32,
                        help="Number of digit positions (default: 32)")
    parser.add_argument("--exp-min", type=int, default=-32,
                        help="Minimum exponent (default: -32)")
    parser.add_argument("--exp-max", type=int, default=32,
                        help="Maximum exponent (default: 32)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build codec
    codec = AnalyticNumberCodec(K=args.K, exp_min=args.exp_min, exp_max=args.exp_max)
    print(f"AnalyticNumberCodec  (K={codec.K}, exp=[{codec.exp_min},{codec.exp_max}], "
          f"total_dim={codec.total_dim})")
    print()

    # Generate numbers
    numbers = generate_numbers(args.range_min, args.range_max, args.n, args.seed)

    # Encode
    print("  Encoding numbers...")
    embeddings, reconstructions = encode_numbers(codec, numbers)
    print(f"  Embeddings shape: {embeddings.shape}")
    print()

    dims_to_plot = [2, 3] if args.both else [args.dim]

    for dim in dims_to_plot:
        print(f"  UMAP {dim}D projection...")
        reducer = UMAP(n_components=dim, n_neighbors=30, min_dist=0.3,
                       metric='cosine', random_state=42)
        coords = reducer.fit_transform(embeddings)

        output_path = os.path.join(
            args.output_dir,
            f"explorer_analytic_umap_{dim}d.html")
        build_explorer(numbers, embeddings, reconstructions, coords, dim,
                       "UMAP", output_path, emb_dim=codec.total_dim)

    print()
    for dim in dims_to_plot:
        path = os.path.join(args.output_dir, f"explorer_analytic_umap_{dim}d.html")
        print(f"Open in browser:  file://{os.path.abspath(path)}")


if __name__ == "__main__":
    main()
