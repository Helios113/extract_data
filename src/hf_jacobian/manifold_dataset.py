"""
Manifold dataset backed by the Stan Monge patch sampler.

Every manifold is a Monge patch  f: R^d → R,  f(x) = Σ λᵢ xᵢ²,
embedded in R^D via a random orthonormal frame.

Special cases via lambdas:
  all zeros  → hyperplane (flat patch, exact uniform on d-ball)
  all equal  → isotropic paraboloid (local approximation to a sphere)

Curvatures and the validity condition R < 1/(2 max|λ|) are enforced
at config resolution time; a ValueError stops the run immediately.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data

from hf_jacobian.stan_samples import (
    check_patch_radius,
    lambdas_from_params,
    sample_monge_patch,
    sample_monge_patch_neighbourhood,
)


@dataclass
class ManifoldConfig:
    manifold_dim: int            # intrinsic dimension d
    ambient_dim:  int            # embedding space R^D
    n_samples:    int            # number of sequences
    seq_len:      int            # points per sequence
    noise_std:    float = 0.0
    seed:         int | None = None
    patch_radius: float = 1.0   # base domain ball radius in R^d
    neighbourhood_radius: float | None = None  # if set, each sequence is a
                                               # neighbourhood of this radius
    # curvature — provide exactly one:
    #   lambdas       : explicit list of d coefficients (takes priority)
    #   lambda_params : dict with keys entropy, lambda_min, lambda_max,
    #                   isotropic (bool), same_sign (bool)
    #   (neither)     : all-zeros → hyperplane
    lambdas:       list[float] | None = None
    lambda_params: dict | None = None


def _ortho_frame(source_dim: int, target_dim: int, gen: torch.Generator) -> torch.Tensor:
    """(target_dim, source_dim) matrix with orthonormal columns."""
    Q, _ = torch.linalg.qr(torch.randn(target_dim, target_dim, generator=gen))
    return Q[:, :source_dim]


def _resolve_lambdas(cfg: ManifoldConfig) -> list[float]:
    """Resolve lambdas from config and enforce the injectivity-radius bound."""
    d, R = cfg.manifold_dim, cfg.patch_radius
    if cfg.lambdas is not None:
        lambdas = cfg.lambdas
    elif cfg.lambda_params is not None:
        p = cfg.lambda_params
        lambdas = lambdas_from_params(
            d=d,
            R=R,
            entropy=p["entropy"],
            lambda_min=p["lambda_min"],
            lambda_max=p["lambda_max"],
            isotropic=p.get("isotropic", False),
            same_sign=p.get("same_sign", True),
            rng=np.random.default_rng(cfg.seed),
        )
    else:
        lambdas = [0.0] * d
    check_patch_radius(lambdas, R)
    return lambdas


def sample_manifold(n: int, cfg: ManifoldConfig, gen: torch.Generator) -> torch.Tensor:
    """Sample n points from the Monge patch. Returns (n, ambient_dim)."""
    lambdas = _resolve_lambdas(cfg)
    if cfg.neighbourhood_radius is not None:
        return sample_monge_patch_neighbourhood(
            cfg.n_samples, cfg.seq_len,
            cfg.manifold_dim, cfg.ambient_dim,
            lambdas,
            radius=cfg.neighbourhood_radius,
            R=cfg.patch_radius,
            noise_std=cfg.noise_std,
            seed=cfg.seed,
        )
    return sample_monge_patch(
        n, cfg.manifold_dim, cfg.ambient_dim, lambdas,
        R=cfg.patch_radius, noise_std=cfg.noise_std, seed=cfg.seed,
    )


class ManifoldDataset(torch.utils.data.Dataset):
    """
    Each item is a (seq_len, ambient_dim) tensor of Monge patch points.
    Batching via DataLoader gives (B, seq_len, ambient_dim).

    If project_dim is set, a fixed random orthonormal projection
    R^ambient_dim → R^project_dim is applied after sampling.
    """

    def __init__(self, cfg: ManifoldConfig, project_dim: int | None = None):
        gen = torch.Generator()
        if cfg.seed is not None:
            gen.manual_seed(cfg.seed)

        pts = sample_manifold(cfg.n_samples * cfg.seq_len, cfg, gen)  # (N, D)

        if project_dim is not None and project_dim != cfg.ambient_dim:
            proj = _ortho_frame(
                min(cfg.ambient_dim, project_dim),
                max(cfg.ambient_dim, project_dim),
                gen,
            )
            pts = pts @ proj if project_dim < cfg.ambient_dim else pts @ proj.T

        self.data = pts.reshape(cfg.n_samples, cfg.seq_len, -1)
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.data[i]
