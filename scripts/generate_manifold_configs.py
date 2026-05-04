"""
Read existing manifold configs to get R values and write the full sweep.

R values are read from the current real-weights configs per model family, then
reused to write the full grid for that model.

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



KAPPAS       = [0.05, 0.2, 0.4]
ENTROPIES    = [0.1, 0.5, 1.0]
NOISE_RATIOS = [0.0, 0.05, 0.2]
MANIFOLD_DIM = 30
N_SAMPLES    = 1024
SEQ_LEN      = 64
EXTRA_N_SAMPLES = 6000
EXTRA_SEQ_LEN   = 256


# ── model families ────────────────────────────────────────────────────────────
# Each family has one reference H5 to measure R from, then a list of precision
# variants to write configs for.

# (variant) — one entry per model, full precision only
MODELS: list[dict] = [
    dict(model="openai-community/gpt2",  cfg_dir="configs/gpt2/0.124b/fp32/manifolds/real_weights",    out_dir="out/gpt2/0.124b/fp32/manifold",    short="gpt2",        batch=100),
    dict(model="EleutherAI/pythia-160m", cfg_dir="configs/pythia/0.160b/fp32/manifolds/real_weights",  out_dir="out/pythia/0.160b/fp32/manifold",  short="pythia_160m", batch=100),
    dict(model="EleutherAI/pythia-1.4b", cfg_dir="configs/pythia/1.4b/fp32/manifolds/real_weights",    out_dir="out/pythia/1.4b/fp32/manifold",    short="pythia_1.4b", batch=100),
    dict(model="Qwen/Qwen3-0.6B",        cfg_dir="configs/qwen/0.6b/bf16/manifolds/real_weights",      out_dir="out/qwen/0.6b/bf16/manifold",      short="qwen3_0.6b",  batch=100),
    dict(model="Qwen/Qwen3-1.7B",        cfg_dir="configs/qwen/1.7b/bf16/manifolds/real_weights",      out_dir="out/qwen/1.7b/bf16/manifold",      short="qwen3_1.7b",  batch=100),
]


def _load_first_cfg(cfg_dir: Path, patterns: list[str]) -> dict:
    for pattern in patterns:
        matches = sorted(cfg_dir.glob(pattern))
        if matches:
            return json.loads(matches[0].read_text())
    patterns_text = ", ".join(patterns)
    raise FileNotFoundError(f"No existing configs found in {cfg_dir} for patterns: {patterns_text}")


def read_R_values(cfg_dir: Path) -> tuple[float, float, int]:
    r_large_cfg = _load_first_cfg(cfg_dir, [f"*_flat_Rlarge_nonoise_d{MANIFOLD_DIM}_*.json", f"*_Rlarge_*_d{MANIFOLD_DIM}_*.json"])
    r_small_cfg = _load_first_cfg(cfg_dir, [f"*_flat_Rsmall_nonoise_d{MANIFOLD_DIM}_*.json", f"*_Rsmall_*_d{MANIFOLD_DIM}_*.json"])
    r_large = float(r_large_cfg["source"]["patch_radius"])
    r_small = float(r_small_cfg["source"]["patch_radius"])
    d_model = int(r_large_cfg["source"]["project_dim"])
    return r_large, r_small, d_model


def configs_for_variant(v: dict, R_large: float, R_small: float, d_model: int) -> list[tuple[str, dict]]:
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

    stem = f"{v['short']}_flat_Rlarge_nonoise_d{MANIFOLD_DIM}_n{EXTRA_N_SAMPLES}_s{EXTRA_SEQ_LEN}_extra"
    src = {
        "type":         "manifold",
        "manifold_dim": MANIFOLD_DIM,
        "ambient_dim":  MANIFOLD_DIM + 1,
        "patch_radius": round(R_large, 6),
        "lambdas":      [0.0] * MANIFOLD_DIM,
        "noise_std":    0.0,
        "seed":         42,
        "project_dim":  d_model,
    }
    out_path = f"{v['out_dir']}/flat_Rlarge_nonoise_d{MANIFOLD_DIM}_n{EXTRA_N_SAMPLES}_s{EXTRA_SEQ_LEN}_extra.h5"
    out.append((stem, _cfg(v, out_path, src, n_samples=EXTRA_N_SAMPLES, seq_len=EXTRA_SEQ_LEN)))

    return out


def _cfg(v, out_path, source, n_samples=N_SAMPLES, seq_len=SEQ_LEN):
    return {
        "model":             v["model"],
        "device":            "cuda",
        "weights":           "real",
        "output":            out_path,
        "compute_jacobians": False,
        "wandb":             False,
        "sampling": {
            "n_samples":  n_samples,
            "seq_len":    seq_len,
            "batch_size": v["batch"],
        },
        "source": source,
    }


def main():
    total = 0
    for v in MODELS:
        cfg_dir = Path(v["cfg_dir"])
        R_large, R_small, d_model = read_R_values(cfg_dir)
        print(f"{v['short']:15s}  d={d_model}  R_large={R_large:.4f}  R_small={R_small:.4f}")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        pairs = configs_for_variant(v, R_large, R_small, d_model)
        for stem, cfg in pairs:
            (cfg_dir / f"{stem}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"  wrote {len(pairs)} configs → {cfg_dir}/")
        total += len(pairs)

    print(f"\nTotal: {total} configs")


if __name__ == "__main__":
    main()
