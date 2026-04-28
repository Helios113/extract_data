"""
Intrinsic dimension estimators: TwoNN and ESS, implemented in PyTorch.

TwoNN: Facco et al., 2017 — global estimator based on nearest-neighbor distance ratios.
ESS:   Johnsson et al., 2015 — local estimator based on expected simplex skewness.

Both match the skdim reference implementations numerically.
"""

import bisect
import math
from functools import lru_cache
from itertools import combinations

import numpy as np
import torch


# ─── TwoNN ───────────────────────────────────────────────────────────────────

def twonn(X: torch.Tensor, discard_fraction: float = 0.1) -> float:
    """
    TwoNN intrinsic dimension estimator (Facco et al., 2017).

    X: (N, D) float tensor
    Returns scalar ID estimate.
    """
    N = X.shape[0]
    X = X.double()

    dists = torch.cdist(X, X)
    dists.fill_diagonal_(float("inf"))
    r, _ = dists.topk(2, dim=1, largest=False)   # (N, 2)
    mu = r[:, 1] / r[:, 0]                        # r2 / r1

    keep = int(N * (1 - discard_fraction))
    mu_sorted = mu.sort().values[:keep]

    F_emp = torch.arange(keep, dtype=torch.float64, device=X.device) / N

    log_mu = mu_sorted.log()
    neg_log_surv = -(1 - F_emp).log()

    # OLS slope through origin: d = (x'y) / (x'x)
    d = (log_mu @ neg_log_surv) / (log_mu @ log_mu)
    return d.item()


# ─── ESS reference table ──────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _ess_ref_cached(maxdim: int, mindim: int, d: int, ver: str) -> tuple:
    """ESS reference values for integer dimensions mindim..maxdim (as a tuple for caching)."""
    if ver == "a":
        J1 = np.array([1 + i for i in range(0, maxdim + 2, 2) if 1 + i <= maxdim])
        J2 = np.array([2 + i for i in range(0, maxdim + 2, 2) if 2 + i <= maxdim])

        f1 = np.full(maxdim, np.nan)
        f1[J1 - 1] = math.gamma(0.5) / math.gamma(1.0) * np.concatenate(([1.0], np.cumprod(J1 / (J1 + 1))[:-1]))
        f1[J2 - 1] = math.gamma(1.0) / math.gamma(1.5) * np.concatenate(([1.0], np.cumprod(J2 / (J2 + 1))[:-1]))

        K1 = np.array([d + 1 + i for i in range(0, maxdim + 2, 2) if d + 1 + i <= maxdim])
        K2 = np.array([d + 2 + i for i in range(0, maxdim + 2, 2) if d + 2 + i <= maxdim])
        f2 = np.zeros(maxdim)
        if len(K1):
            f2[K1 - 1] = math.gamma((d + 1) / 2) / math.gamma(0.5) * np.concatenate(([1.0], np.cumprod(K1 / (K1 - d))[:-1]))
        if len(K2):
            f2[K2 - 1] = math.gamma((d + 2) / 2) / math.gamma(1.0) * np.concatenate(([1.0], np.cumprod(K2 / (K2 - d))[:-1]))

        return tuple((f1 ** d * f2)[mindim - 1: maxdim].tolist())

    if ver == "b" and d == 1:
        J1 = np.array([1 + i for i in range(0, maxdim + 2, 2) if 1 + i <= maxdim])
        J2 = np.array([2 + i for i in range(0, maxdim + 2, 2) if 2 + i <= maxdim])
        ID = np.full(maxdim, np.nan)
        ID[J1 - 1] = math.gamma(1.5) / math.gamma(1.0) * np.concatenate(([1.0], np.cumprod((J1 + 2) / (J1 + 1))[:-1]))
        ID[J2 - 1] = math.gamma(2.0) / math.gamma(1.5) * np.concatenate(([1.0], np.cumprod((J2 + 2) / (J2 + 1))[:-1]))
        ns = np.arange(mindim, maxdim + 1, dtype=float)
        return tuple((ID[mindim - 1: maxdim] * 2 / math.sqrt(math.pi) / ns).tolist())

    raise ValueError(f"ver={ver!r} with d={d} is not supported")


def _ess_ref(maxdim: int, mindim: int, d: int, ver: str) -> list:
    return list(_ess_ref_cached(maxdim, mindim, d, ver))


# ─── ESS value computation ────────────────────────────────────────────────────

def _ess_values_batch(
    neighborhoods: torch.Tensor,
    d: int = 1,
    ver: str = "a",
    n_groups: int = 5000,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Batch ESS value computation.

    neighborhoods: (N, k, D) — k neighbors per point (NOT yet centered)
    Returns: (N,) ESS values (float64)
    """
    N, k, D = neighborhoods.shape
    p = d + 1

    if p > D:
        fill = 0.0 if ver == "a" else 1.0
        return torch.full((N,), fill, dtype=torch.float64, device=neighborhoods.device)

    vecs = neighborhoods.double() - neighborhoods.double().mean(dim=1, keepdim=True)  # (N, k, D)

    all_combs = list(combinations(range(k), p))
    if len(all_combs) > n_groups:
        if rng is None:
            rng = np.random.default_rng()
        sel = rng.choice(len(all_combs), size=n_groups, replace=False)
        all_combs = [all_combs[int(i)] for i in np.sort(sel)]

    G = len(all_combs)
    comb_idx = torch.tensor(all_combs, dtype=torch.long, device=vecs.device)  # (G, p)

    # (N, G, p, D)
    grouped = vecs[:, comb_idx, :]

    norms = grouped.norm(dim=-1)         # (N, G, p)
    weight_sums = norms.prod(dim=-1).sum(dim=-1)  # (N,)

    if ver == "a":
        gram = grouped @ grouped.transpose(-1, -2)  # (N, G, p, p)
        vol = gram.det().abs().sqrt()               # (N, G)
        ess = vol.sum(dim=-1) / weight_sums
    else:  # ver == "b", d == 1
        proj = (grouped[:, :, 0, :] * grouped[:, :, 1, :]).sum(dim=-1).abs()  # (N, G)
        ess = proj.sum(dim=-1) / weight_sums

    return torch.where(weight_sums > 0, ess, torch.full_like(ess, float("nan")))


# ─── ESS value → ID lookup ────────────────────────────────────────────────────

def _ess_to_id(essval: float, d: int, ver: str) -> float:
    """Map a scalar ESS value to a fractional ID via the reference table."""
    if math.isnan(essval):
        return float("nan")

    mindim, maxdim = 1, 20
    # dimvals[i] = ESS reference for dimension i+1 (0-indexed), cumulative across expansions
    dimvals = _ess_ref(maxdim, 1, d, ver)

    while (ver == "a" and essval > dimvals[maxdim - 1]) or (ver == "b" and essval < dimvals[maxdim - 1]):
        mindim = maxdim + 1
        maxdim = 2 * (maxdim - 1)
        dimvals = dimvals + _ess_ref(maxdim, mindim, d, ver)

    if ver == "a":
        i = bisect.bisect(dimvals[mindim - 1: maxdim], essval)
    else:
        i = (maxdim - mindim + 1) - bisect.bisect(dimvals[mindim - 1: maxdim][::-1], essval)

    de_int = mindim + i - 1
    if de_int < 1 or de_int >= len(dimvals):
        return float("nan")

    de_frac = (essval - dimvals[de_int - 1]) / (dimvals[de_int] - dimvals[de_int - 1])
    return de_int + de_frac


# ─── ESS public API ───────────────────────────────────────────────────────────

def ess(
    X: torch.Tensor,
    k: int = 100,
    d: int = 1,
    ver: str = "a",
    n_groups: int = 5000,
    seed: int | None = None,
) -> dict:
    """
    ESS intrinsic dimension estimator (Johnsson et al., 2015).

    X       : (N, D) float tensor
    k       : number of nearest neighbors defining each local patch
    d       : simplex dimension (ver='a': any d≥1; ver='b': d=1 only)
    ver     : 'a' or 'b' (see paper)
    n_groups: max simplex groups to sample per point

    Returns dict:
      dimension_pw  np.ndarray (N,) — per-point fractional ID
      essval        np.ndarray (N,) — per-point ESS statistic
      dimension     float — nanmean of dimension_pw
    """
    rng = np.random.default_rng(seed)

    X_d = X.double()
    dists = torch.cdist(X_d, X_d)
    dists.fill_diagonal_(float("inf"))
    _, knn_idx = dists.topk(k, dim=1, largest=False)  # (N, k)

    neighborhoods = X[knn_idx]  # (N, k, D)
    ess_vals = _ess_values_batch(neighborhoods, d=d, ver=ver, n_groups=n_groups, rng=rng)
    ess_np = ess_vals.cpu().numpy()

    ids = np.array([_ess_to_id(float(ev), d, ver) for ev in ess_np])

    return {
        "dimension_pw": ids,
        "essval": ess_np,
        "dimension": float(np.nanmean(ids)),
    }
