"""
Stan-based Monge patch sampler for curved hypersurfaces.

The surface is f: R^d → R defined by f(x) = sum_i(lambda_i * x_i^2), embedded
in R^D via a random orthonormal frame.  Special cases:
  - all lambdas == 0  → hyperplane (flat; area element = 1, reduces to uniform on d-ball)
  - all lambdas equal → isotropic paraboloid (local approximation to a sphere)

Sampling is exact (up to MCMC) with the surface-area-element Jacobian correction
computed in the Stan model.

Validity condition
------------------
The Monge patch is a graph over R^d so its metric is always positive definite —
it never self-intersects.  The meaningful constraint is that the base domain ball
stays within the **injectivity radius** of the surface, i.e., that neighbouring
surface normals do not cross inside the patch.  For f(x) = Σ λᵢ xᵢ², the
principal curvatures at the origin are κᵢ = 2λᵢ (the second fundamental form
divided by the metric factor, which equals 1 at x=0).  The injectivity radius is
bounded below by 1/κ_max = 1/(2 |λ_max|).  Therefore the patch is well-posed iff

    R  <  1 / (2 * max_i |λᵢ|)

When all λ = 0 (hyperplane) the bound is +∞, i.e., always valid.
"""

from pathlib import Path

import numpy as np
import torch

_STAN_DIR = Path(__file__).parent / "stan_codes"
_MODEL_PATH = _STAN_DIR / "sampler.stan"


# ─── curvature helpers ────────────────────────────────────────────────────────

def patch_curvatures(lambdas: list[float]) -> dict:
    """
    Curvature invariants of the Monge patch f(x) = Σ λᵢ xᵢ² at the origin.

    At x=0 the metric is the identity so the second fundamental form entries are
    simply 2λᵢ, giving principal curvatures κᵢ = 2λᵢ.

    Returns
    -------
    mean_curvature   : H = (1/d) Σ κᵢ = (2/d) Σ λᵢ
    scalar_curvature : R = Σ_{i≠j} κᵢ κⱼ = (Σκᵢ)² − Σκᵢ²  (sectional sum)
    max_radius       : injectivity-radius bound = 1/(2 max|λᵢ|), or inf if all zero
    lambda_entropy   : Shannon entropy of the normalised |λ| distribution (nats)
    """
    lam = np.asarray(lambdas, dtype=float)
    kappa = 2.0 * lam                        # principal curvatures at origin

    d = len(lam)
    mean_curvature = kappa.mean()            # H = (1/d) Σ κᵢ

    # scalar curvature = sum of all pairwise products κᵢ κⱼ (i≠j)
    # = ((Σκᵢ)² − Σκᵢ²) / 2  — this is the Gauss–Bonnet / Riemann scalar
    sum_k  = kappa.sum()
    sum_k2 = (kappa ** 2).sum()
    scalar_curvature = (sum_k ** 2 - sum_k2) / 2.0

    lam_abs = np.abs(lam)
    max_lam = lam_abs.max() if d > 0 else 0.0
    max_radius = 1.0 / (2.0 * max_lam) if max_lam > 0 else float("inf")

    # Shannon entropy over the normalised |λ| distribution
    total = lam_abs.sum()
    if total == 0.0 or d == 0:
        lambda_entropy = 0.0
    else:
        p = lam_abs / total
        # avoid log(0)
        lambda_entropy = -float(np.sum(p * np.log(p, where=p > 0, out=np.zeros_like(p))))

    return {
        "mean_curvature":   float(mean_curvature),
        "scalar_curvature": float(scalar_curvature),
        "max_radius":       float(max_radius),
        "lambda_entropy":   float(lambda_entropy),
    }


def check_patch_radius(lambdas: list[float], R: float) -> None:
    """Raise ValueError if R violates the injectivity-radius bound."""
    info = patch_curvatures(lambdas)
    if R >= info["max_radius"]:
        raise ValueError(
            f"Patch radius R={R} exceeds injectivity bound "
            f"1/(2*|λ_max|) = {info['max_radius']:.4g}. "
            f"Reduce R or lower the curvature."
        )


def lambdas_from_params(
    d: int,
    R: float,
    entropy: float,
    lambda_min: float,
    lambda_max: float,
    isotropic: bool = False,
    same_sign: bool = True,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """
    Generate a lambdas vector of length d consistent with the given parameters.

    Parameters
    ----------
    d           : intrinsic dimension (length of output vector)
    R           : patch radius; enforces max|λ| < 1/(2R)
    entropy     : target Shannon entropy of the |λ| distribution (nats).
                  Clamped to [0, log(d)].  Ignored when isotropic=True.
    lambda_min  : minimum magnitude for each |λᵢ| (≥ 0)
    lambda_max  : maximum magnitude for each |λᵢ|; also bounded by 1/(2R)
    isotropic   : if True all |λᵢ| are equal (maximum entropy); entropy arg ignored
    same_sign   : if True all λᵢ share the same sign (positive by default);
                  if False signs are drawn i.i.d. uniformly ±1
    rng         : numpy random generator (seeded externally or None for default)

    The validity condition R < 1/(2 max|λᵢ|) is always enforced: lambda_max is
    silently clamped to min(lambda_max, 1/(2R) - ε) before generating.
    """
    if d == 0:
        return []

    rng = rng or np.random.default_rng()

    # hard upper bound from the injectivity radius
    hard_max = (1.0 / (2.0 * R)) * (1.0 - 1e-6) if R > 0 else float("inf")
    lambda_max = min(lambda_max, hard_max)

    if lambda_max < lambda_min:
        raise ValueError(
            f"lambda_max={lambda_max:.4g} < lambda_min={lambda_min:.4g} "
            f"after applying the injectivity bound 1/(2R)={1/(2*R):.4g}. "
            f"Reduce lambda_min or R."
        )

    if isotropic or d == 1:
        # all magnitudes equal — maximum entropy (or d=1, entropy is trivially 0)
        mag = np.full(d, (lambda_min + lambda_max) / 2.0)
    else:
        H_target = float(np.clip(entropy, 0.0, np.log(d)))
        mag = _sample_magnitudes_at_entropy(d, H_target, lambda_min, lambda_max, rng)

    if same_sign:
        signs = np.ones(d)
    else:
        signs = rng.choice([-1.0, 1.0], size=d)

    return (mag * signs).tolist()


def _sample_magnitudes_at_entropy(
    d: int,
    H_target: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a d-vector of magnitudes in [lo, hi] whose normalised |λ| distribution
    has Shannon entropy ≈ H_target (nats).

    Strategy: build a deterministic "temperature" profile p_i ∝ exp(-t * rank_i)
    over randomly permuted ranks to hit H_target exactly, then scale to [lo, hi].
    This gives exact entropy matching regardless of d, lo, hi.
    """
    H_max = np.log(d)

    if H_target >= H_max - 1e-9:
        return np.full(d, hi)
    if H_target <= 1e-9:
        mag = np.full(d, lo if lo > 0 else hi * 1e-6)
        mag[rng.integers(d)] = hi
        return mag

    # Bisect temperature t ≥ 0 of the geometric profile p_i ∝ exp(-t*i).
    # At t=0: uniform (H=log d). As t→∞: one-hot (H→0).
    def profile_entropy(t: float) -> float:
        ranks = np.arange(d, dtype=float)
        log_p = -t * ranks
        log_p -= log_p.max()                     # numerical stability
        p = np.exp(log_p)
        p /= p.sum()
        return -float(np.sum(p * np.log(p + 1e-300)))

    t_lo, t_hi = 0.0, 1e4
    for _ in range(80):
        t_mid = (t_lo + t_hi) / 2.0
        if profile_entropy(t_mid) > H_target:
            t_lo = t_mid
        else:
            t_hi = t_mid
    t = (t_lo + t_hi) / 2.0

    ranks = np.arange(d, dtype=float)
    log_p = -t * ranks
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()                                 # sorted descending proportions

    # shuffle so the dominant component lands on a random index
    rng.shuffle(p)

    # scale: max proportion → hi, preserving ratios; clip to [lo, hi]
    mag = p * (hi / p.max())
    return np.clip(mag, lo, hi)


def _ortho_frame(source_dim: int, target_dim: int, gen: torch.Generator) -> torch.Tensor:
    """(target_dim, source_dim) orthonormal column matrix embedding R^source into R^target."""
    Q, _ = torch.linalg.qr(torch.randn(target_dim, target_dim, generator=gen))
    return Q[:, :source_dim]


def _get_compiled_model():
    """Compile (and cache) the Stan model. Compilation is done once per process."""
    import cmdstanpy
    # suppress cmdstanpy noise
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    return cmdstanpy.CmdStanModel(stan_file=str(_MODEL_PATH))


_compiled_model = None


def _model():
    global _compiled_model
    if _compiled_model is None:
        _compiled_model = _get_compiled_model()
    return _compiled_model


def sample_monge_patch(
    n_samples: int,
    d: int,
    D: int,
    lambdas: list[float],
    R: float = 1.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample n_samples points from the Monge patch surface z = sum_i(lambda_i * x_i^2)
    embedded in R^D via a random orthonormal frame, with uniform surface measure.

    lambdas: list of d curvature coefficients.
      - all zeros  → hyperplane (flat patch, exact uniform on d-ball projected to R^D)
      - all equal  → isotropic paraboloid (symmetric curvature)
    R: radius of the base domain ball in R^d.
    noise_std: isotropic Gaussian noise added to each point in R^D.
    """
    assert len(lambdas) == d, f"len(lambdas)={len(lambdas)} must equal d={d}"
    assert D >= d + 1, f"ambient_dim D={D} must be >= d+1={d+1}"
    check_patch_radius(lambdas, R)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # frame embeds the local (x, z) coordinates into R^D
    frame = _ortho_frame(d + 1, D, gen)

    lam = np.array(lambdas, dtype=float)
    all_zero = np.all(lam == 0.0)

    if all_zero:
        # Hyperplane: z=0, area element=1 → uniform on d-ball, no Stan needed.
        pts = _sample_flat_patch(n_samples, d, frame, R, noise_std, gen)
        return pts

    data = {
        "d": d,
        "D": D,
        "frame": frame.numpy(),     # shape (D, d+1) matches Stan's matrix[D, d+1]
        "R": float(R),
        "lambdas": lam,
        "noise_std": float(noise_std),
    }

    chains = 4
    iter_per_chain = max(1, (n_samples // chains) + 1)

    fit = _model().sample(
        data=data,
        iter_warmup=500,
        iter_sampling=iter_per_chain,
        chains=chains,
        seed=seed,
        show_progress=False,
        show_console=False,
    )

    pts_np = fit.stan_variable("pt")          # (chains * iter_per_chain, D)
    pts_np = pts_np[:n_samples]               # trim to exactly n_samples
    return torch.from_numpy(pts_np).float()


def _sample_flat_patch(
    n: int, d: int, frame: torch.Tensor, R: float, noise_std: float, gen: torch.Generator
) -> torch.Tensor:
    """Uniform on d-ball of radius R, embedded via frame as a flat patch (z=0)."""
    # direction on d-sphere, then radius with volume-correct CDF
    dirs = torch.randn(n, d, generator=gen)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u = torch.rand(n, 1, generator=gen).pow(1.0 / d)
    x = dirs * (u * R)                          # (n, d)

    # local coords (x, z=0) in R^{d+1}
    local = torch.cat([x, torch.zeros(n, 1)], dim=-1)  # (n, d+1)
    pts = local @ frame.T                               # (n, D)

    if noise_std > 0:
        pts = pts + torch.randn_like(pts, generator=gen) * noise_std
    return pts


# ─── neighbourhood sampler ────────────────────────────────────────────────────

def sample_monge_patch_neighbourhood(
    n_samples: int,
    seq_len: int,
    d: int,
    D: int,
    lambdas: list[float],
    radius: float,
    R: float = 1.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample n_samples sequences of seq_len points each, where each sequence is
    drawn from a neighbourhood of radius `radius` around a random anchor on the
    Monge patch surface.  Returns (n_samples * seq_len, D).

    Strategy: sample n_samples anchors from the full surface, then for each
    anchor draw seq_len points uniformly from the ambient L2 ball of given radius
    and project them back onto the surface via the Monge patch equation.
    """
    assert len(lambdas) == d
    check_patch_radius(lambdas, R)
    lam = torch.tensor(lambdas, dtype=torch.float32)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    frame = _ortho_frame(d + 1, D, gen)

    # sample anchors in base domain
    anchors_x = _sample_ball(n_samples, d, R, gen)         # (n_samples, d)
    anchors_z = (lam * anchors_x.pow(2)).sum(-1, keepdim=True)  # (n_samples, 1)
    anchors = torch.cat([anchors_x, anchors_z], dim=-1)    # (n_samples, d+1)

    # for each anchor, sample seq_len offsets in R^{d+1} and project to surface
    dirs = torch.randn(n_samples, seq_len, d + 1, generator=gen)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u = torch.rand(n_samples, seq_len, 1, generator=gen).pow(1.0 / (d + 1))
    offsets = dirs * (u * radius)                           # (n_samples, seq_len, d+1)

    local = anchors.unsqueeze(1) + offsets                  # (n_samples, seq_len, d+1)
    # project: keep x-coords, recompute z from Monge patch
    x_part = local[..., :d]                                 # (n_samples, seq_len, d)
    z_part = (lam * x_part.pow(2)).sum(-1, keepdim=True)   # (n_samples, seq_len, 1)
    local = torch.cat([x_part, z_part], dim=-1)

    pts = local.reshape(n_samples * seq_len, d + 1) @ frame.T   # (N, D)

    if noise_std > 0:
        pts = pts + torch.randn_like(pts, generator=gen) * noise_std
    return pts


def _sample_ball(n: int, d: int, R: float, gen: torch.Generator) -> torch.Tensor:
    dirs = torch.randn(n, d, generator=gen)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u = torch.rand(n, 1, generator=gen).pow(1.0 / d)
    return dirs * (u * R)
