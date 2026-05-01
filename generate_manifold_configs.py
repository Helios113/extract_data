"""
Measure embedding scale from H5 files and write the full manifold config sweep.

R is measured once per model family from the best-available H5 (fp32/bf16 preferred),
then the same R is used for all precision variants of that model.

  R_large = median pairwise distance / 2  (embedding scale)
  R_small = R_large / 5                   (tight local patch)

Grid per precision variant:
  kappa  in {0.05, 0.2, 0.4}   lambda_max = kappa / R; kappa < 0.5 enforced
  entropy in {0.1, 0.5, 1.0}   normalised [0,1]
  noise_ratio in {0.0, 0.05, 0.2}  noise_std = ratio * R
  + flat baselines at both R values x 3 noise levels

same_sign=False throughout.

Usage:
    python generate_manifold_configs.py
"""

import json
from pathlib import Path

import h5py
import numpy as np


KAPPAS       = [0.05, 0.2, 0.4]
ENTROPIES    = [0.1, 0.5, 1.0]
NOISE_RATIOS = [0.0, 0.05, 0.2]
MANIFOLD_DIM = 30
N_SAMPLES    = 1024
SEQ_LEN      = 64


# ── model families ────────────────────────────────────────────────────────────
# Each family has one reference H5 to measure R from, then a list of precision
# variants to write configs for.

# (ref_h5, variant) — one entry per model, full precision only
MODELS: list[tuple[str, dict]] = [
    (
        "out/gpt2/0.124b/fp32/dataset/wikitext_n1024_s64.h5",
        dict(model="openai-community/gpt2",  cfg_dir="configs/gpt2/0.124b/fp32/manifolds/real_weights",    out_dir="out/gpt2/0.124b/fp32/manifold",    short="gpt2",        batch=100),
    ),
    (
        "out/pythia/0.160b/fp32/dataset/wikitext_n1024_s64.h5",
        dict(model="EleutherAI/pythia-160m", cfg_dir="configs/pythia/0.160b/fp32/manifolds/real_weights",  out_dir="out/pythia/0.160b/fp32/manifold",  short="pythia_160m", batch=100),
    ),
    (
        "out/pythia/1.4b/fp32/dataset/wikitext_n1024_s64.h5",
        dict(model="EleutherAI/pythia-1.4b", cfg_dir="configs/pythia/1.4b/fp32/manifolds/real_weights",    out_dir="out/pythia/1.4b/fp32/manifold",    short="pythia_1.4b", batch=100),
    ),
    (
        "out/qwen/0.6b/fp8/dataset/wikitext_n1024_s64.h5",
        dict(model="Qwen/Qwen3-0.6B",        cfg_dir="configs/qwen/0.6b/bf16/manifolds/real_weights",      out_dir="out/qwen/0.6b/bf16/manifold",      short="qwen3_0.6b",  batch=100),
    ),
    (
        "out/qwen/1.7b/bf16/dataset/wikitext_n1024_s64.h5",
        dict(model="Qwen/Qwen3-1.7B",        cfg_dir="configs/qwen/1.7b/bf16/manifolds/real_weights",      out_dir="out/qwen/1.7b/bf16/manifold",      short="qwen3_1.7b",  batch=100),
    ),
]


def measure_R_large(h5_path: str, n_sample: int = 2000, seed: int = 0) -> tuple[float, int]:
    with h5py.File(h5_path, "r") as f:
        embed = f["embed_out"][:]
    d    = embed.shape[-1]
    flat = embed.reshape(-1, d).astype(np.float32)
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(flat), size=min(n_sample, len(flat)), replace=False)
    sub  = flat[idx]
    diffs = sub[None] - sub[:, None]
    dists = np.linalg.norm(diffs, axis=-1)
    triu  = dists[np.triu_indices(len(sub), k=1)]
    return float(np.median(triu) / 2.0), d


def configs_for_variant(v: dict, R_large: float, d_model: int) -> list[tuple[str, dict]]:
    R_small = R_large / 5.0
    out = []

    for R, r_tag in [(R_small, "Rsmall"), (R_large, "Rlarge")]:

        # flat baselines
        for noise_ratio in NOISE_RATIOS:
            noise_std = noise_ratio * R
            n_tag = f"noise{noise_ratio}".replace(".", "p") if noise_ratio > 0 else "nonoise"
            stem  = f"{v['short']}_flat_{r_tag}_{n_tag}_d{MANIFOLD_DIM}_n{N_SAMPLES}_s{SEQ_LEN}"
            src   = {
                "type":         "manifold",
                "manifold_dim": MANIFOLD_DIM,
                "ambient_dim":  MANIFOLD_DIM + 1,
                "patch_radius": round(R, 6),
                "lambdas":      [0.0] * MANIFOLD_DIM,
                "noise_std":    round(noise_std, 6),
                "seed":         42,
                "project_dim":  d_model,
            }
            out_path = f"{v['out_dir']}/flat_{r_tag}_{n_tag}_d{MANIFOLD_DIM}_n{N_SAMPLES}_s{SEQ_LEN}.h5"
            out.append((stem, _cfg(v, out_path, src)))

        # curved grid
        for kappa in KAPPAS:
            lambda_max = kappa / R
            k_tag = f"k{kappa}".replace(".", "p")
            for entropy in ENTROPIES:
                e_tag = f"e{entropy}".replace(".", "p")
                for noise_ratio in NOISE_RATIOS:
                    noise_std = noise_ratio * R
                    n_tag = f"noise{noise_ratio}".replace(".", "p") if noise_ratio > 0 else "nonoise"
                    stem  = f"{v['short']}_{r_tag}_{k_tag}_{e_tag}_{n_tag}_d{MANIFOLD_DIM}_n{N_SAMPLES}_s{SEQ_LEN}"
                    isotropic = (entropy == 1.0)
                    src   = {
                        "type":         "manifold",
                        "manifold_dim": MANIFOLD_DIM,
                        "ambient_dim":  MANIFOLD_DIM + 1,
                        "patch_radius": round(R, 6),
                        "lambda_params": {
                            "entropy":    entropy,
                            "lambda_min": 0.0,
                            "lambda_max": round(lambda_max, 6),
                            "isotropic":  isotropic,
                            "same_sign":  isotropic,  # isotropic paraboloid → all same sign
                        },
                        "noise_std":    round(noise_std, 6),
                        "seed":         42,
                        "project_dim":  d_model,
                    }
                    out_path = f"{v['out_dir']}/{r_tag}_{k_tag}_{e_tag}_{n_tag}_d{MANIFOLD_DIM}_n{N_SAMPLES}_s{SEQ_LEN}.h5"
                    out.append((stem, _cfg(v, out_path, src)))

    return out


def _cfg(v, out_path, source):
    return {
        "model":             v["model"],
        "device":            "cuda",
        "weights":           "real",
        "output":            out_path,
        "compute_jacobians": False,
        "wandb":             False,
        "sampling": {
            "n_samples":  N_SAMPLES,
            "seq_len":    SEQ_LEN,
            "batch_size": v["batch"],
        },
        "source": source,
    }


def main():
    total = 0
    for ref_h5, v in MODELS:
        R_large, d_model = measure_R_large(ref_h5)
        R_small = R_large / 5.0
        print(f"{v['short']:15s}  d={d_model}  R_large={R_large:.4f}  R_small={R_small:.4f}")

        cfg_dir = Path(v["cfg_dir"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        pairs = configs_for_variant(v, R_large, d_model)
        for stem, cfg in pairs:
            (cfg_dir / f"{stem}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"  wrote {len(pairs)} configs → {cfg_dir}/")
        total += len(pairs)

    print(f"\nTotal: {total} configs")


if __name__ == "__main__":
    main()
