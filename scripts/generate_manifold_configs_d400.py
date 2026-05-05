"""
Generate a second set of manifold configs for fp32 GPT-2 with manifold_dim=400.

Same grid as the d30 set (same R values, kappas, noise levels) but with
manifold_dim=400 and ambient_dim=401.
"""

import json
from pathlib import Path

MANIFOLD_DIM = 768
N_SAMPLES    = 1024
SEQ_LEN      = 64

# R values measured from the GPT-2 wikitext dataset (same as d30 set)
R_LARGE = 2.269174
R_SMALL = 0.453835

# kappa -> lambda_max = kappa / R  (rounded to 6 dp, matching d30 set)
KAPPAS = [0.05, 0.2, 0.4]
NOISE_RATIOS = [0.0, 0.05, 0.2]

MODEL   = "openai-community/gpt2"
D_MODEL = 768
BATCH   = 100

BASE_CFG_DIR = Path("configs/gpt2/0.124b/fp32/manifolds")
BASE_OUT_DIR = "out/gpt2/0.124b/fp32/manifold"
SHORT        = "gpt2"


def _cfg(weights, out_path, source):
    return {
        "model":             MODEL,
        "device":            "cuda",
        "weights":           weights,
        "output":            out_path,
        "compute_jacobians": False,
        "wandb":             False,
        "sampling": {
            "n_samples":  N_SAMPLES,
            "seq_len":    SEQ_LEN,
            "batch_size": BATCH,
        },
        "source": source,
    }


def build_configs(weights_tag, weights_val):
    pairs = []
    cfg_dir  = BASE_CFG_DIR / f"{weights_tag}_weights"
    out_dir  = f"{BASE_OUT_DIR}/{weights_tag}_weights"
    d        = MANIFOLD_DIM
    stem_d   = f"d{d}"

    for R, r_tag in [(R_SMALL, "Rsmall"), (R_LARGE, "Rlarge")]:

        # flat baselines
        for noise_ratio in NOISE_RATIOS:
            noise_std = round(noise_ratio * R, 6)
            n_tag = f"noise{noise_ratio}".replace(".", "p") if noise_ratio > 0 else "nonoise"
            stem  = f"{SHORT}_flat_{r_tag}_{n_tag}_{stem_d}_n{N_SAMPLES}_s{SEQ_LEN}"
            src = {
                "type":         "manifold",
                "manifold_dim": d,
                "ambient_dim":  d + 1,
                "patch_radius": R,
                "lambdas":      [0.0] * d,
                "noise_std":    noise_std,
                "seed":         42,
                "project_dim":  D_MODEL,
            }
            out_path = f"{out_dir}/flat_{r_tag}_{n_tag}_{stem_d}_n{N_SAMPLES}_s{SEQ_LEN}.h5"
            pairs.append((cfg_dir / f"{stem}.json", _cfg(weights_val, out_path, src)))

        # curved grid (entropy=1.0 only, matching the existing d30 set)
        for kappa in KAPPAS:
            lambda_max = round(kappa / R, 6)
            k_tag = f"k{kappa}".replace(".", "p")
            for noise_ratio in NOISE_RATIOS:
                noise_std = round(noise_ratio * R, 6)
                n_tag = f"noise{noise_ratio}".replace(".", "p") if noise_ratio > 0 else "nonoise"
                stem  = f"{SHORT}_{r_tag}_{k_tag}_e1p0_{n_tag}_{stem_d}_n{N_SAMPLES}_s{SEQ_LEN}"
                src = {
                    "type":         "manifold",
                    "manifold_dim": d,
                    "ambient_dim":  d + 1,
                    "patch_radius": R,
                    "lambda_params": {
                        "entropy":    1.0,
                        "lambda_min": 0.0,
                        "lambda_max": lambda_max,
                        "isotropic":  True,
                        "same_sign":  True,
                    },
                    "noise_std":    noise_std,
                    "seed":         42,
                    "project_dim":  D_MODEL,
                }
                out_path = f"{out_dir}/{r_tag}_{k_tag}_e1p0_{n_tag}_{stem_d}_n{N_SAMPLES}_s{SEQ_LEN}.h5"
                pairs.append((cfg_dir / f"{stem}.json", _cfg(weights_val, out_path, src)))

    return pairs


def main():
    total = 0
    for weights_tag, weights_val in [("real", "real"), ("synthetic", "random")]:
        cfg_dir = BASE_CFG_DIR / f"{weights_tag}_weights"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        pairs = build_configs(weights_tag, weights_val)
        for path, cfg in pairs:
            path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{weights_tag}_weights: wrote {len(pairs)} configs → {cfg_dir}/")
        total += len(pairs)
    print(f"Total: {total} configs")


if __name__ == "__main__":
    main()
