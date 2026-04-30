"""
Plot ID estimate vs depth from a CSV produced by analyze_id.py.

Usage:
  python plot_id.py <csv_file> [--method twonn|ess_a|ess_b] [--pos 0 10 49] [--out fig.png]

  CSV must have columns: depth, pos, twonn, ess_a, ess_b, n  (produced by analyze_id.py)

  python plot_id.py results.csv                          # all positions, default method
  python plot_id.py results.csv --method ess_a           # choose estimator
  python plot_id.py results.csv --pos 10 49              # specific token positions
  python plot_id.py results.csv --pos 10 --out fig.png   # save instead of show
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def depth_order(depths: pd.Series) -> list[str]:
    """Sort depth labels in residual-stream order."""
    def key(d):
        if d == "embed":
            return (-1, "")
        if d == "final":
            return (10_000, "")
        parts = d.split("/")          # "layer_3/attn"
        return (int(parts[0].split("_")[1]), parts[1])
    return sorted(depths.unique(), key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file")
    ap.add_argument("--method", default=None,
                    help="Column to plot: twonn, ess_a, ess_b. "
                         "Defaults to all available estimator columns.")
    ap.add_argument("--pos", type=int, nargs="+", default=None,
                    help="Token position(s) to plot. Defaults to all in the CSV.")
    ap.add_argument("--out", default="out/figs/",
                    help="Save figure to this path instead of displaying it.")
    args = ap.parse_args()

    df = load_csv(args.csv_file)

    estimator_cols = [c for c in ("twonn", "ess_a", "ess_b") if c in df.columns]
    if not estimator_cols:
        sys.exit("No estimator columns (twonn, ess_a, ess_b) found in CSV.")

    if args.method is not None:
        if args.method not in df.columns:
            sys.exit(f"Method '{args.method}' not in CSV. Available: {estimator_cols}")
        methods = [args.method]
    else:
        methods = estimator_cols

    all_positions = sorted(df["pos"].unique())
    positions = args.pos if args.pos is not None else all_positions
    missing = [p for p in positions if p not in all_positions]
    if missing:
        sys.exit(f"Position(s) {missing} not found in CSV. Available: {all_positions}")

    depths = depth_order(df["depth"])
    x = range(len(depths))

    fig, ax = plt.subplots(figsize=(max(10, len(depths) * 0.4), 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", ":"]

    for pi, pos in enumerate(positions):
        sub = df[df["pos"] == pos].set_index("depth")
        for mi, method in enumerate(methods):
            y = [sub.loc[d, method] if d in sub.index else float("nan") for d in depths]
            label = f"pos={pos}" if len(methods) == 1 else f"pos={pos} / {method}"
            ax.plot(
                x, y,
                marker="o", markersize=4,
                color=colors[pi % len(colors)],
                linestyle=linestyles[mi % len(linestyles)],
                label=label,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(depths, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ID estimate")
    ax.set_xlabel("depth")
    ax.set_title(f"ID vs depth  —  {', '.join(methods)}")
    ax.legend(fontsize=8, ncol=max(1, len(positions) // 8))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
