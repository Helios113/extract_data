"""
Nearest-neighbour distance analysis for hidden states stored in an HDF5 file.

For each (layer, sublayer, token position), we have n_samples points in R^d.
We compute the distance from each point to its nearest neighbour across the
n_samples axis.

Produces two publication-quality figures:

  1. Histogram of all NN distances pooled across the entire transformer
     (all layers × all token positions).

  2. Heatmap of mean ± std of NN distances, with depth on the x-axis
     and token position on the y-axis.  Mean and std are shown as separate
     subplots stacked vertically.

Usage:
  # full extraction from HDF5
  uv run python scripts/plot_knn_distances.py <file.h5> [--out-dir DIR]

  # replot from saved CSV (no extraction)
  uv run python scripts/plot_knn_distances.py --csv results/.../foo_nn_raw.csv \\
                                               [--model-name "gpt2"] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ─── helpers ─────────────────────────────────────────────────────────────────

def depth_key(label: str) -> tuple:
    if label == "embed":
        return (-1, "")
    if label == "final":
        return (100_000, "")
    parts = label.split("/")          # "layer_3/attn"
    return (int(parts[0].split("_")[1]), parts[1])


def collect_depths(f: h5py.File) -> list[str]:
    """Return ordered list of depth labels present in the file."""
    labels = ["embed"]
    n_layers = int(f["meta"].attrs["n_layers"])
    sublayers = []
    # detect sublayer names from whatever is in layer_0
    if "layer_0" in f:
        sublayers = list(f["layer_0"].keys())
        sublayers = sorted(sublayers)   # attn before ffn
    for i in range(n_layers):
        for sub in sublayers:
            key = f"layer_{i}/{sub}"
            if key in f and "hidden_out" in f[key]:
                labels.append(f"layer_{i}/{sub}")
    labels.append("final")
    return labels


def load_hidden(f: h5py.File, label: str) -> np.ndarray:
    """Return (n_samples, seq_len, d) float32 array."""
    if label == "embed":
        return f["embed_out"][:]
    if label == "final":
        return f["final_hidden"][:]
    return f[label]["hidden_out"][:]


def nn_distances_per_position(hidden: np.ndarray) -> np.ndarray:
    """
    hidden: (n_samples, seq_len, d)
    Returns (n_samples, seq_len) array of distances to nearest neighbour
    across the sample axis for each token position.
    """
    n_samples, seq_len, d = hidden.shape
    dists = np.empty((n_samples, seq_len), dtype=np.float32)
    for pos in range(seq_len):
        pts = hidden[:, pos, :].astype(np.float64)  # (n_samples, d)
        tree = cKDTree(pts)
        # k=2: first hit is self (distance 0), second is nearest neighbour
        dd, _ = tree.query(pts, k=2)
        dists[:, pos] = dd[:, 1].astype(np.float32)
    return dists


# ─── plotting ────────────────────────────────────────────────────────────────

def plot_histogram(all_dists: np.ndarray, model_name: str, out_path: Path):
    n_zeros = int((all_dists == 0).sum())
    n_total = len(all_dists)

    pos = all_dists[all_dists > 0]
    bins = np.geomspace(pos.min(), pos.max(), 121)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pos, bins=bins, density=True, color="#4878CF", alpha=0.85, linewidth=0)
    ax.set_xscale("log")

    # mark the minimum non-zero distance with a vertical line
    ax.axvline(pos.min(), color="#CC3333", linewidth=1.2, linestyle="--", zorder=3)
    ax.text(
        pos.min(), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
        f"  min = {pos.min():.3g}\n  (0 excluded: {n_zeros}/{n_total})",
        color="#CC3333", fontsize=8, va="top",
        transform=ax.get_xaxis_transform(),
    )

    zero_note = (
        f"No exact zeros ({n_total} distances total)"
    )
    ax.set_xlabel(f"Nearest-neighbour distance (log scale)\n{zero_note}", fontsize=10)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"NN distance distribution — {model_name}\n(all layers & token positions)", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram → {out_path}")


def _plot_single_heatmap(grid: np.ndarray, depth_labels: list[str],
                          title: str, model_name: str, out_path: Path):
    from matplotlib.colors import LogNorm
    n_depths = len(depth_labels)
    n_zeros = int((grid == 0).sum())
    g = grid.T.copy()
    g = np.clip(g, np.finfo(np.float32).tiny, None)

    fig, ax = plt.subplots(figsize=(max(8, n_depths * 0.35), 4))
    im = ax.imshow(
        g,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=g.min(), vmax=g.max()),
    )
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    cb.set_label(
        f"distance (log)\nmin={g.min():.3g}  —  no zeros"
        if n_zeros == 0
        else f"distance (log)\n{n_zeros} zeros clipped to {np.finfo(np.float32).tiny:.1e}",
        fontsize=8,
    )
    ax.set_ylabel("Token position", fontsize=11)
    ax.set_title(f"{title} — {model_name}", fontsize=11)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.set_xticks(range(n_depths))
    ax.set_xticklabels(depth_labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Depth", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap   → {out_path}")


def plot_heatmap(mean_grid: np.ndarray, std_grid: np.ndarray,
                 depth_labels: list[str],
                 model_name: str, out_path: Path):
    stem = out_path.with_suffix("")
    suffix = out_path.suffix
    _plot_single_heatmap(mean_grid, depth_labels, "Mean NN distance",
                         model_name, Path(f"{stem}_mean{suffix}"))
    _plot_single_heatmap(std_grid,  depth_labels, "Std of NN distance",
                         model_name, Path(f"{stem}_std{suffix}"))


# ─── data loading ────────────────────────────────────────────────────────────

def load_from_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        model_name = f["meta"].attrs.get("model", h5_path.stem)
        depths = collect_depths(f)
        seq_len = int(f["meta"].attrs["seq_len"])

        print(f"Model: {model_name}")
        print(f"Depths found: {len(depths)}  |  seq_len: {seq_len}")

        mean_grid = np.empty((len(depths), seq_len), dtype=np.float32)
        std_grid  = np.empty((len(depths), seq_len), dtype=np.float32)
        all_dists_list = []

        total_zeros = 0
        total_points = 0
        for di, label in enumerate(depths):
            print(f"  [{di+1}/{len(depths)}] {label} ...", end="", flush=True)
            hidden = load_hidden(f, label)
            dists  = nn_distances_per_position(hidden)
            mean_grid[di] = dists.mean(axis=0)
            std_grid[di]  = dists.std(axis=0)
            flat = dists.ravel()
            all_dists_list.append(dists)
            n_zeros = int((flat == 0).sum())
            total_zeros += n_zeros
            total_points += flat.size
            print(f"  mean={flat.mean():.4f}  std={flat.std():.4f}  zeros={n_zeros}/{flat.size}")

        print(f"\nTotal zero distances: {total_zeros} / {total_points} ({100*total_zeros/total_points:.3f}%)")

    all_dists = np.concatenate([d.ravel() for d in all_dists_list])
    return model_name, depths, seq_len, mean_grid, std_grid, all_dists, all_dists_list


def load_from_csv(raw_path: Path):
    raw     = pd.read_csv(raw_path)
    depths  = list(dict.fromkeys(raw["depth"]))   # preserve insertion order
    seq_len = int(raw["pos"].max()) + 1

    mean_grid = np.zeros((len(depths), seq_len), dtype=np.float32)
    std_grid  = np.zeros((len(depths), seq_len), dtype=np.float32)
    for di, label in enumerate(depths):
        sub = raw[raw["depth"] == label].groupby("pos")["nn_dist"]
        mean_grid[di] = sub.mean().loc[range(seq_len)].values
        std_grid[di]  = sub.std().loc[range(seq_len)].values

    all_dists = raw["nn_dist"].values.astype(np.float32)
    return depths, seq_len, mean_grid, std_grid, all_dists


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NN distance plots from hidden-state HDF5 or saved CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("h5_file", nargs="?", default=None,
                    help="Path to the HDF5 file produced by run.py")
    ap.add_argument("--csv", default=None, metavar="CSV",
                    help="Replot from a saved *_nn_raw.csv (skips extraction)")
    ap.add_argument("--model-name", default=None,
                    help="Model label for plot titles when using --summary/--raw")
    ap.add_argument("--out-dir", default=None,
                    help="Directory for output PNGs")
    args = ap.parse_args()

    from_csv = args.csv is not None
    if from_csv and args.h5_file:
        ap.error("Pass either an h5_file or --csv, not both.")
    if not from_csv and not args.h5_file:
        ap.error("Provide an h5_file or use --csv.")

    if from_csv:
        raw_path   = Path(args.csv)
        model_name = args.model_name or raw_path.stem.replace("_nn_raw", "")
        stem       = raw_path.stem.replace("_nn_raw", "")
        out_dir    = Path(args.out_dir) if args.out_dir else raw_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Loading from CSV ...")
        depths, _, mean_grid, std_grid, all_dists = load_from_csv(raw_path)
    else:
        h5_path = Path(args.h5_file).resolve()
        root = Path.cwd()
        try:
            rel = h5_path.parent.relative_to(root)
            out_dir = Path(args.out_dir) if args.out_dir else root / "results" / rel
        except ValueError:
            out_dir = Path(args.out_dir) if args.out_dir else Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = h5_path.stem

        model_name, depths, seq_len, mean_grid, std_grid, all_dists, all_dists_list = load_from_h5(h5_path)

        raw_rows = []
        for di, label in enumerate(depths):
            dists_dp = all_dists_list[di]
            for pos in range(seq_len):
                for v in dists_dp[:, pos]:
                    raw_rows.append({"depth": label, "pos": pos, "nn_dist": float(v)})
        raw_path = out_dir / f"{stem}_nn_raw.csv"
        pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
        print(f"Saved raw CSV → {raw_path}")

    plot_histogram(all_dists, model_name, out_dir / f"{stem}_nn_hist.pdf")
    plot_heatmap(mean_grid, std_grid, depths, model_name, out_dir / f"{stem}_nn_heatmap.pdf")


if __name__ == "__main__":
    main()
