"""
Geometric manifold dataset for feeding into transformers as latent inputs.

Each manifold is parameterized by intrinsic dimension d and ambient dimension D.
Points live in R^D via a random orthonormal embedding of the manifold's native space.

Manifolds supported:
  plane       — d-flat, uniform on [-1, 1]^d
  sphere      — d-sphere S^d (surface of (d+1)-ball), uniform
  ellipsoid   — d-ellipsoid with configurable axis scales, near-uniform
  hyperboloid — upper sheet of H^d embedded in R^{d+1}, Gaussian pushforward

If ambient_dim != model's d_model, pass project_dim to ManifoldDataset and a
fixed random projection R^D → R^{project_dim} is applied.
"""

from dataclasses import dataclass, field

import torch
import torch.utils.data


@dataclass
class ManifoldConfig:
    manifold:     str            # 'plane' | 'sphere' | 'ellipsoid' | 'hyperboloid'
    manifold_dim: int            # intrinsic dimension d
    ambient_dim:  int            # embedding space dimension D
    n_samples:    int            # number of sequences (analogous to n_documents)
    seq_len:      int            # manifold points per sequence
    noise_std:    float = 0.0   # isotropic Gaussian noise added after embedding
    seed:         int | None = None
    scales:       list[float] | None = None  # ellipsoid only: d+1 axis lengths


# ─── random embedding helpers ────────────────────────────────────────────────

def _ortho_frame(source_dim: int, target_dim: int, gen: torch.Generator) -> torch.Tensor:
    """(target_dim, source_dim) matrix with orthonormal columns: embeds R^source into R^target."""
    Q, _ = torch.linalg.qr(torch.randn(target_dim, target_dim, generator=gen))
    return Q[:, :source_dim]


def _embed(pts: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """pts: (N, s)  frame: (D, s)  →  (N, D)"""
    return pts @ frame.T


# ─── sampling functions ──────────────────────────────────────────────────────

def sample_plane(n: int, d: int, D: int, gen: torch.Generator) -> torch.Tensor:
    """Uniform on the d-flat, coefficients in [-1, 1]^d."""
    assert D >= d, f"ambient_dim ({D}) must be >= manifold_dim ({d}) for plane"
    frame = _ortho_frame(d, D, gen)
    coeffs = torch.rand(n, d, generator=gen) * 2 - 1
    return _embed(coeffs, frame)


def sample_sphere(n: int, d: int, D: int, gen: torch.Generator) -> torch.Tensor:
    """Uniform on the d-sphere S^d ⊂ R^{d+1}, embedded in R^D."""
    assert D >= d + 1, f"ambient_dim ({D}) must be >= manifold_dim+1 ({d+1}) for sphere"
    frame = _ortho_frame(d + 1, D, gen)
    raw = torch.randn(n, d + 1, generator=gen)
    pts = raw / raw.norm(dim=1, keepdim=True)
    return _embed(pts, frame)


def sample_ellipsoid(
    n: int, d: int, D: int, scales: list[float] | None, gen: torch.Generator
) -> torch.Tensor:
    """Near-uniform on the d-ellipsoid (sphere scaled per axis), embedded in R^D.

    scales: d+1 axis lengths (defaults to [1, 1.5, 2, ..., 1 + 0.5*d]).
    """
    assert D >= d + 1, f"ambient_dim ({D}) must be >= manifold_dim+1 ({d+1}) for ellipsoid"
    frame = _ortho_frame(d + 1, D, gen)
    raw = torch.randn(n, d + 1, generator=gen)
    unit = raw / raw.norm(dim=1, keepdim=True)

    if scales is None:
        scales = [1.0 + 0.5 * i for i in range(d + 1)]
    scale_t = torch.tensor(scales[: d + 1], dtype=torch.float32)
    pts = unit * scale_t
    return _embed(pts, frame)


def sample_hyperboloid(n: int, d: int, D: int, gen: torch.Generator) -> torch.Tensor:
    """Upper sheet of H^d: x_{d+1}^2 - ||x_{1:d}||^2 = 1, x_{d+1} > 0.

    Gaussian pushforward: v ~ N(0, I_d), x = (v, sqrt(1+||v||^2)).
    This is not the uniform Riemannian measure but is geometrically correct.
    """
    assert D >= d + 1, f"ambient_dim ({D}) must be >= manifold_dim+1 ({d+1}) for hyperboloid"
    frame = _ortho_frame(d + 1, D, gen)
    v = torch.randn(n, d, generator=gen)
    t = (1 + v.pow(2).sum(dim=1, keepdim=True)).sqrt()
    pts = torch.cat([v, t], dim=1)
    return _embed(pts, frame)


_SAMPLERS = {
    "plane":       sample_plane,
    "sphere":      sample_sphere,
    "hyperboloid": sample_hyperboloid,
}


def sample_manifold(n: int, cfg: ManifoldConfig, gen: torch.Generator) -> torch.Tensor:
    """Dispatch to the right sampler, apply noise. Returns (n, ambient_dim)."""
    if cfg.manifold == "ellipsoid":
        pts = sample_ellipsoid(n, cfg.manifold_dim, cfg.ambient_dim, cfg.scales, gen)
    elif cfg.manifold in _SAMPLERS:
        pts = _SAMPLERS[cfg.manifold](n, cfg.manifold_dim, cfg.ambient_dim, gen)
    else:
        raise ValueError(f"Unknown manifold {cfg.manifold!r}. Choose from: plane, sphere, ellipsoid, hyperboloid")

    if cfg.noise_std > 0:
        pts = pts + torch.randn_like(pts, generator=gen) * cfg.noise_std

    return pts


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ManifoldDataset(torch.utils.data.Dataset):
    """
    Each item is a (seq_len, ambient_dim) tensor of manifold points.
    Batching via DataLoader produces (B, seq_len, ambient_dim) — directly usable
    as inputs_embeds in a transformer.

    If project_dim is set, a fixed random projection is applied so items are
    (seq_len, project_dim). Use this when ambient_dim != model's d_model.
    """

    def __init__(self, cfg: ManifoldConfig, project_dim: int | None = None):
        gen = torch.Generator()
        if cfg.seed is not None:
            gen.manual_seed(cfg.seed)

        total = cfg.n_samples * cfg.seq_len
        pts = sample_manifold(total, cfg, gen)  # (total, D)

        if project_dim is not None and project_dim != cfg.ambient_dim:
            # Fixed random orthonormal projection R^D → R^{project_dim}
            proj = _ortho_frame(
                min(cfg.ambient_dim, project_dim),
                max(cfg.ambient_dim, project_dim),
                gen,
            )
            if project_dim < cfg.ambient_dim:
                # project down: (total, D) @ (D, project_dim)
                pts = pts @ proj
            else:
                # embed up: (total, D) @ (D, project_dim)
                pts = pts @ proj.T

        self.data = pts.reshape(cfg.n_samples, cfg.seq_len, -1)
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.data[i]  # (seq_len, out_dim)
