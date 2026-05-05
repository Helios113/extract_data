"""
Plot per-depth statistics from a CSV produced by analyze_id.py.

For each metric (twonn, ess_a, ess_b, pca_curv) and each k value in the CSV,
produces one figure with two subplots:
  - mean across all (sample, position) at each depth
  - 95th-percentile confidence band (mean ± 1.96 * std / sqrt(n))

One PNG per (metric, k) combination is written to the output directory.

Usage:
  python plot_id_sweep.py <csv_file> [--out-dir DIR] [--pos 0 10 49]

  python plot_id_sweep.py results/run_id.csv
  python plot_id_sweep.py results/run_id.csv --out-dir figs/ --pos 10 20
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def depth_order_key(d: str) -> tuple:
    if d == "embed":
        return (-1, "")
    if d == "final":
        return (10_000, "")
    parts = d.split("/")
    return (int(parts[0].split("_")[1]), parts[1])


def sorted_depths(depths) -> list[str]:
    return sorted(set(depths), key=depth_order_key)


def detect_ks(columns: list[str]) -> list[int]:
    """Extract k values from columns named ess_a_k<k>."""
    ks = sorted({int(m.group(1)) for c in columns if (m := re.match(r"ess_a_k(\d+)", c))})
    return ks


def depth_index_labels(depths: list[str]) -> list[str]:
    """Short x-axis labels: embed → 'E', final → 'F', layer_i/sub → 'i/sub'."""
    out = []
    for d in depths:
        if d == "embed":
            out.append("E")
        elif d == "final":
            out.append("F")
        else:
            parts = d.split("/")
            out.append(f"{parts[0].split('_')[1]}/{parts[1]}")
    return out


# ── stats ─────────────────────────────────────────────────────────────────────

def ci95(values: np.ndarray) -> tuple[float, float]:
    """Returns (mean, half-width of 95% CI) using z=1.96."""
    n = np.sum(~np.isnan(values))
    if n < 2:
        return float(np.nanmean(values)), float("nan")
    m  = float(np.nanmean(values))
    se = float(np.nanstd(values, ddof=1)) / np.sqrt(n)
    return m, 1.96 * se


def compute_stats(df: pd.DataFrame, metric_col: str, depths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (means, lo, hi) arrays aligned to depths."""
    means = np.full(len(depths), float("nan"))
    lo    = np.full(len(depths), float("nan"))
    hi    = np.full(len(depths), float("nan"))
    for i, d in enumerate(depths):
        vals = df.loc[df["depth"] == d, metric_col].dropna().values
        if len(vals) == 0:
            continue
        m, hw = ci95(vals)
        means[i] = m
        lo[i]    = m - hw
        hi[i]    = m + hw
    return means, lo, hi


# ── plotting ──────────────────────────────────────────────────────────────────

_METRIC_LABELS = {
    "twonn":    "TwoNN ID",
    "ess_a":    "ESS-a ID",
    "ess_b":    "ESS-b ID",
    "pca_curv": "LocalPCA curvature",
    "entropy":  "Shannon entropy (nats)",
}

_COLORS = {
    "twonn":    "#4878CF",
    "ess_a":    "#E87B23",
    "ess_b":    "#6ACC65",
    "pca_curv": "#D65F5F",
    "entropy":  "#B47CC7",
}


def plot_metric_k(df: pd.DataFrame, metric: str, k: int | None,
                  depths: list[str], x_labels: list[str],
                  model_name: str, out_path: Path):
    col   = f"{metric}_k{k}" if k is not None else metric
    color = _COLORS.get(metric, "#555555")
    label = _METRIC_LABELS.get(metric, metric)
    k_tag = f"  (k={k})" if k is not None else ""

    means, lo, hi = compute_stats(df, col, depths)
    x = np.arange(len(depths))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(depths) * 0.38), 7),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # ── top: mean + CI band ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, means, color=color, linewidth=1.8, marker="o", markersize=4, label="mean")
    ax.fill_between(x, lo, hi, alpha=0.25, color=color, label="95% CI")
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"{label}{k_tag} vs depth — {model_name}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    # ── bottom: CI half-width (shows uncertainty alone) ───────────────────
    ax2 = axes[1]
    hw = hi - means
    ax2.bar(x, hw, color=color, alpha=0.6)
    ax2.set_ylabel("CI half-width", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=55, ha="right", fontsize=7)
    axes[-1].set_xlabel("Depth", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_all_ks_overlay(df: pd.DataFrame, metric: str, ks: list[int],
                        depths: list[str], x_labels: list[str],
                        model_name: str, out_path: Path):
    """Overlay all k values for one metric on the same axes (mean only)."""
    label = _METRIC_LABELS.get(metric, metric)
    x = np.arange(len(depths))

    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(10, len(depths) * 0.38), 5))

    for i, k in enumerate(ks):
        col = f"{metric}_k{k}"
        if col not in df.columns:
            continue
        means, lo, hi = compute_stats(df, col, depths)
        c = palette(i / max(len(ks) - 1, 1))
        ax.plot(x, means, color=c, linewidth=1.6, marker="o", markersize=3.5, label=f"k={k}")
        ax.fill_between(x, lo, hi, alpha=0.15, color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel("Depth", fontsize=11)
    ax.set_title(f"{label} — k-sweep — {model_name}", fontsize=12)
    ax.legend(fontsize=9, title="k (ESS neighbours)")
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("csv_file", help="CSV produced by analyze_id.py")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for PNGs (default: same dir as CSV).")
    ap.add_argument("--pos", type=int, nargs="+", default=None,
                    help="Restrict to these token positions (default: all).")
    args = ap.parse_args()

    csv_path  = Path(args.csv_file)
    out_dir   = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem      = csv_path.stem

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, comment="#")

    if args.pos is not None:
        df = df[df["pos"].isin(args.pos)]
        if df.empty:
            raise SystemExit(f"No rows for positions {args.pos}.")

    # model name from comment header
    model_name = stem
    try:
        with open(csv_path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if line.startswith("# model:"):
                    model_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    depths   = sorted_depths(df["depth"])
    x_labels = depth_index_labels(depths)
    ks       = detect_ks(list(df.columns))

    print(f"Depths: {len(depths)}  |  k-values: {ks}  |  rows: {len(df)}")

    # ── per-metric, per-k individual plots ────────────────────────────────
    kd_metrics = [(m, k) for m in ("ess_a", "ess_b", "pca_curv") for k in ks]
    for metric, k in kd_metrics:
        col = f"{metric}_k{k}"
        if col not in df.columns:
            continue
        fname = out_dir / f"{stem}_{metric}_k{k}.png"
        plot_metric_k(df, metric, k, depths, x_labels, model_name, fname)

    # TwoNN and entropy are k-independent
    if "twonn" in df.columns:
        plot_metric_k(df, "twonn", None, depths, x_labels, model_name,
                      out_dir / f"{stem}_twonn.png")

    if "entropy" in df.columns:
        plot_metric_k(df, "entropy", None, depths, x_labels, model_name,
                      out_dir / f"{stem}_entropy.png")

    # ── k-sweep overlay plots (one per metric) ────────────────────────────
    if len(ks) > 1:
        for metric in ("ess_a", "ess_b", "pca_curv"):
            if not any(f"{metric}_k{k}" in df.columns for k in ks):
                continue
            fname = out_dir / f"{stem}_{metric}_sweep.png"
            plot_all_ks_overlay(df, metric, ks, depths, x_labels, model_name, fname)


if __name__ == "__main__":
    main()
