"""
Randomized invertibility certificate for a matrix J accessed only via
matrix-vector products v -> J @ v.

Based on: Halko, Martinsson & Tropp (2011), "Finding Structure with Randomness"
"""

import torch
from dataclasses import dataclass


@dataclass
class InvertibilityResult:
    is_invertible: bool
    sigma_min_estimate: float
    sigma_max_estimate: float
    condition_estimate: float
    rank_estimate: int
    singular_values: torch.Tensor


def check_invertibility(
    Jv_fn,
    n: int,
    m: int = 40,
    tau: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: int = None,
) -> InvertibilityResult:
    """
    Probabilistic invertibility test for an n×n matrix J accessed via Jv_fn.

    Draws m Gaussian probe vectors, applies J to each, and inspects the singular
    values of the resulting matrix V = [Jv_1 | ... | Jv_m].

    If rank(J) = r < m, rank(V) <= r — detectable via SVD.
    sigma_min(V) / sqrt(n - m) gives a conservative lower bound on sigma_min(J).

    Failure probability decreases exponentially in m when a singular value gap exists:
        P(miss) <= (sigma_{r+1} / sigma_r)^m
    """
    if seed is not None:
        torch.manual_seed(seed)

    Omega = torch.randn(n, m, dtype=dtype, device=device)
    V = torch.empty(n, m, dtype=dtype, device=device)
    for k in range(m):
        V[:, k] = Jv_fn(Omega[:, k])

    singular_values = torch.linalg.svdvals(V)
    sigma_1 = singular_values[0].item()
    sigma_m = singular_values[-1].item()

    scale_correction = (n - m) ** 0.5 if n > m else 1.0
    sigma_min_J = sigma_m / scale_correction if scale_correction > 0 else sigma_m
    sigma_max_J = sigma_1 / (n ** 0.5)

    numerical_rank = int((singular_values / sigma_1 > tau).sum().item())
    condition_estimate = sigma_max_J / sigma_min_J if sigma_min_J > 0 else float("inf")
    is_invertible = (numerical_rank == m) and (sigma_min_J > tau)

    return InvertibilityResult(
        is_invertible=is_invertible,
        sigma_min_estimate=sigma_min_J,
        sigma_max_estimate=sigma_max_J,
        condition_estimate=condition_estimate,
        rank_estimate=numerical_rank,
        singular_values=singular_values,
    )
