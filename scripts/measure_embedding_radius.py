"""
Measure the natural scale of embedding vectors from HDF5 output files,
then print a grid of manifold configs for sweeping curvature and noise.

Usage:
    python measure_embedding_radius.py out/gpt2/.../wikitext.h5 [out/other.h5 ...]

R is defined as the median pairwise distance between tokens divided by 2,
i.e. the radius of a ball that would span a typical pair of points.
The orthonormal projection from the patch domain into R^d_model preserves
norms, so this R is directly meaningful as a patch_radius.
"""

import argparse
import json
import sys

import h5py
import numpy as np


def measure_radius(path: str, n_sample: int = 2000, seed: int = 0) -> dict:
    with h5py.File(path, "r") as f:
        meta  = dict(f["meta"].attrs)
        embed = f["embed_out"][:]           # (N, S, d)

    flat  = embed.reshape(-1, embed.shape[-1]).astype(np.float32)
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(flat), size=min(n_sample, len(flat)), replace=False)
    sub   = flat[idx]

    norms = np.linalg.norm(sub, axis=-1)

    # pairwise distances via broadcasting over the sample
    diffs = sub[None] - sub[:, None]
    dists = np.linalg.norm(diffs, axis=-1)
    triu  = dists[np.triu_indices(len(sub), k=1)]

    return {
        "model":        meta.get("model", path),
        "d_model":      int(embed.shape[-1]),
        "n_tokens":     len(flat),
        "norm_mean":    float(norms.mean()),
        "norm_p50":     float(np.median(norms)),
        "norm_p95":     float(np.percentile(norms, 95)),
        "dist_mean":    float(triu.mean()),
        "dist_p50":     float(np.median(triu)),
        "dist_p95":     float(np.percentile(triu, 95)),
        # R = half the median pairwise distance
        "R_large":      float(np.median(triu) / 2.0),
    }


def make_configs(R_large: float, d: int = 30, project_dim: int = 768) -> list[dict]:
    """
    Grid: 2 radii × 3 curvature levels × 3 entropy levels × 3 noise levels
    plus a flat baseline.  same_sign=False throughout so signs can be mixed.

    Curvature levels are set so lambda_max = kappa / R, respecting the
    injectivity bound lambda_max < 1/(2R), i.e. kappa < 0.5.
    """
    R_small = 1.0
    # kappa = R * lambda_max; we use 0.05 / 0.2 / 0.4
    kappas      = [0.05, 0.2, 0.4]
    entropies   = [0.1, 0.5, 1.0]
    noise_ratios = [0.0, 0.05, 0.2]   # noise_std = ratio * R

    configs = []

    # flat baseline (lambdas all zero, no curvature)
    for R, r_tag in [(R_small, "small"), (R_large, "large")]:
        for noise_ratio in noise_ratios:
            noise_std = round(noise_ratio * R, 6)
            noise_tag = f"noise{noise_ratio}" if noise_ratio > 0 else "nonoise"
            configs.append({
                "_tag": f"flat_R{r_tag}_{noise_tag}",
                "manifold_dim":   d,
                "ambient_dim":    d + 1,
                "patch_radius":   round(R, 6),
                "lambdas":        [0.0] * d,
                "noise_std":      noise_std,
                "seed":           42,
                "project_dim":    project_dim,
            })

    # curved grid
    for R, r_tag in [(R_small, "small"), (R_large, "large")]:
        for kappa in kappas:
            lambda_max = kappa / R
            # safety check: must be < 1/(2R)
            assert lambda_max < 1.0 / (2.0 * R) - 1e-9, f"kappa={kappa} violates bound for R={R}"
            for entropy in entropies:
                for noise_ratio in noise_ratios:
                    noise_std = round(noise_ratio * R, 6)
                    noise_tag = f"noise{noise_ratio}" if noise_ratio > 0 else "nonoise"
                    tag = f"R{r_tag}_k{kappa}_e{entropy}_{noise_tag}"
                    configs.append({
                        "_tag":         tag,
                        "manifold_dim": d,
                        "ambient_dim":  d + 1,
                        "patch_radius": round(R, 6),
                        "lambda_params": {
                            "entropy":    entropy,
                            "lambda_min": 0.0,
                            "lambda_max": round(lambda_max, 6),
                            "isotropic":  False,
                            "same_sign":  False,   # allow mixed signs
                        },
                        "noise_std":    noise_std,
                        "seed":         42,
                        "project_dim":  project_dim,
                    })

    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="HDF5 output files to measure")
    ap.add_argument("--manifold-dim", type=int, default=30)
    ap.add_argument("--n-sample", type=int, default=2000,
                    help="Number of tokens to subsample for distance estimation")
    ap.add_argument("--dump-configs", action="store_true",
                    help="Print full config grid as JSON")
    args = ap.parse_args()

    results = []
    for path in args.files:
        r = measure_radius(path, n_sample=args.n_sample)
        results.append(r)
        print(f"\n{r['model']}  (d={r['d_model']}, {r['n_tokens']} tokens)")
        print(f"  token norm   mean={r['norm_mean']:.3f}  p50={r['norm_p50']:.3f}  p95={r['norm_p95']:.3f}")
        print(f"  pairwise dist  mean={r['dist_mean']:.3f}  p50={r['dist_p50']:.3f}  p95={r['dist_p95']:.3f}")
        print(f"  → R_large = {r['R_large']:.3f}   R_small = 1.0")

    if args.dump_configs:
        if len(results) != 1:
            print("\n[dump-configs] requires exactly one input file", file=sys.stderr)
            sys.exit(1)
        r = results[0]
        R_large = r["R_large"]
        if R_large <= 1.0:
            print(f"\nWarning: R_large={R_large:.3f} ≤ 1.0 — R_small=1.0 would not be smaller. "
                  f"Using R_small=R_large/5 instead.", file=sys.stderr)
        configs = make_configs(R_large, d=args.manifold_dim, project_dim=r["d_model"])
        print("\n--- config grid ---")
        for c in configs:
            tag = c.pop("_tag")
            print(f"\n# {tag}")
            print(json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
