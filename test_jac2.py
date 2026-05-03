"""
Randomized invertibility certificate for a matrix J accessed only via
matrix-vector products v -> J @ v (and optionally v -> J^T @ v).

Based on: Halko, Martinsson & Tropp (2011), "Finding Structure with Randomness"

The idea: draw m Gaussian random vectors, apply J to each, collect the outputs
into a matrix V = [Jv_1 | ... | Jv_m], then inspect the singular values of V.

Key facts:
  - If rank(J) = r < n, then all columns of V lie in the same r-dim subspace,
    so rank(V) <= r. This is detectable via SVD of V.
  - If J is full rank (det J != 0), then V has rank m (a.s. over Gaussian draws).
  - The smallest singular value of V gives a proxy for sigma_min(J), scaled by
    the norms of the probe vectors. This detects near-singularity.
"""

import torch
from dataclasses import dataclass


@dataclass
class InvertibilityResult:
    is_invertible: bool          # True if J is numerically full-rank
    sigma_min_estimate: float    # Estimate of the smallest singular value of J
    sigma_max_estimate: float    # Estimate of the largest singular value of J
    condition_estimate: float    # sigma_max / sigma_min (condition number estimate)
    rank_estimate: int           # Estimated numerical rank of J
    singular_values: torch.Tensor  # Full singular value spectrum of the probe matrix V


def check_invertibility(
    Jv_fn: callable,
    n: int,
    m: int = 40,
    tau: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    seed: int = None,
) -> InvertibilityResult:
    """
    Probabilistic invertibility test for an n x n matrix J accessed via Jv_fn.

    Args:
        Jv_fn   : callable, takes a vector v of shape (n,) and returns J @ v of shape (n,).
                  This is the ONLY access to J required.
        n       : dimension of J (J is n x n).
        m       : number of random probe vectors. More probes -> higher confidence
                  and better sigma_min estimate. m=40 is conservative for most problems.
        tau     : threshold for numerical rank determination.
                  A singular value sigma_i is considered zero if sigma_i / sigma_1 < tau.
        dtype   : floating point type. float64 recommended for numerical stability.
        device  : torch device string.
        seed    : optional RNG seed for reproducibility.

    Returns:
        InvertibilityResult with invertibility flag, sigma_min estimate, condition number,
        rank estimate, and the full singular value array of the probe matrix V.

    Failure probability:
        If J has a singular value gap (sigma_r >> sigma_{r+1}), the probability of
        missing a rank deficiency decreases exponentially in m:
            P(miss) <= (sigma_{r+1} / sigma_r)^m
        For near-singular J (no clean gap), sigma_min_estimate will be small,
        flagging the problem via the condition number even if is_invertible is True.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # --- Step 1: Draw m Gaussian probe vectors ---
    # Each column of Omega is one probe vector v_k ~ N(0, I_n).
    # Gaussian distribution is rotationally invariant, which is key: it ensures
    # the probes are not aligned with any particular subspace of J.
    Omega = torch.randn(n, m, dtype=dtype, device=device)  # shape (n, m)

    # --- Step 2: Apply J to each probe vector ---
    # V[:, k] = J @ Omega[:, k]
    # We build V column by column using the provided matrix-vector product function.
    V = torch.empty(n, m, dtype=dtype, device=device)
    for k in range(m):
        v_k = Omega[:, k]            # k-th probe vector, shape (n,)
        V[:, k] = Jv_fn(v_k)        # J applied to v_k, shape (n,)

    # --- Step 3: Compute SVD of V ---
    # V = U S W^T where S contains the singular values of V.
    # If J is full rank, rank(V) = m (almost surely).
    # If J has rank r < m, then rank(V) = r, and sigma_{r+1}, ..., sigma_m ~ 0.
    # We only need the singular values (compute_uv=False is faster).
    singular_values = torch.linalg.svdvals(V)  # shape (m,), sorted descending

    sigma_1 = singular_values[0].item()   # largest singular value of V
    sigma_m = singular_values[-1].item()  # smallest singular value of V

    # --- Step 4: Estimate sigma_min(J) ---
    # The probe vectors Omega have expected squared norm n (since each entry ~ N(0,1)).
    # So the expected scaling factor per probe is sqrt(n).
    # sigma_min(V) approx sigma_min(J) * sigma_min(Omega).
    # For a Gaussian matrix Omega of shape (n, m) with m << n,
    # sigma_min(Omega) concentrates around sqrt(n) - sqrt(m) (Marchenko-Pastur law).
    # We use sqrt(n - m) as a conservative correction factor.
    scale_correction = (n - m) ** 0.5 if n > m else 1.0
    sigma_min_J = sigma_m / scale_correction if scale_correction > 0 else sigma_m
    sigma_max_J = sigma_1 / (n ** 0.5)  # rough estimate for sigma_max(J)

    # --- Step 5: Determine numerical rank ---
    # A singular value is considered numerically zero if it is smaller than
    # tau * sigma_1 (the largest singular value), i.e. relatively small.
    numerical_rank = int((singular_values / sigma_1 > tau).sum().item())

    # --- Step 6: Compute condition number estimate ---
    if sigma_min_J > 0:
        condition_estimate = sigma_max_J / sigma_min_J
    else:
        condition_estimate = float("inf")

    # --- Step 7: Invertibility decision ---
    # J is declared invertible if the estimated numerical rank equals m,
    # meaning no singular value of V was below the threshold.
    # Note: if m < n, this certifies rank(J) >= m, not rank(J) = n exactly.
    # For a full certificate set m = n (costly but exact).
    is_invertible = (numerical_rank == m) and (sigma_min_J > tau)

    return InvertibilityResult(
        is_invertible=is_invertible,
        sigma_min_estimate=sigma_min_J,
        sigma_max_estimate=sigma_max_J,
        condition_estimate=condition_estimate,
        rank_estimate=numerical_rank,
        singular_values=singular_values,
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 100

    # -----------------------------------------------------------------------
    # NOTE on choosing m:
    #   With m probes you certify rank(J) >= m, not rank(J) = n.
    #   - For a *full* deterministic certificate, set m = n (costs n Jv calls).
    #   - For a probabilistic certificate against rank deficiency of dimension k,
    #     m = n - k + oversampling (e.g. +10) is sufficient with exponentially
    #     small failure probability. If J has rank r, use m > r.
    #   The examples below use m = n for the deterministic guarantee.
    # -----------------------------------------------------------------------

    # --- Case 1: Full-rank J, m = n (deterministic certificate) ---
    J_full = torch.randn(n, n, dtype=torch.float64)
    Jv_full = lambda v: J_full @ v

    result = check_invertibility(Jv_full, n=n, m=n, seed=42)
    print("=== Full-rank J (m = n, deterministic) ===")
    print(f"  Invertible:        {result.is_invertible}")
    print(f"  Rank estimate:     {result.rank_estimate} / {n}")
    print(f"  sigma_min(J) est:  {result.sigma_min_estimate:.4f}")
    print(f"  sigma_max(J) est:  {result.sigma_max_estimate:.4f}")
    print(f"  Condition number:  {result.condition_estimate:.4f}")

    # --- Case 2: Rank-deficient J (rank n-5), m = n ---
    # Zero out the 5 smallest singular directions to create exact rank deficiency.
    # With m = n probes, this is detectable with probability 1.
    J_rank_def = torch.randn(n, n, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(J_rank_def)
    S[-5:] = 0.0
    J_rank_def = U @ torch.diag(S) @ Vh
    Jv_rank_def = lambda v: J_rank_def @ v

    result2 = check_invertibility(Jv_rank_def, n=n, m=n, seed=42)
    print("\n=== Rank-deficient J, rank = n-5 (m = n, deterministic) ===")
    print(f"  Invertible:        {result2.is_invertible}")
    print(f"  Rank estimate:     {result2.rank_estimate} / {n}")
    print(f"  sigma_min(J) est:  {result2.sigma_min_estimate:.2e}")
    print(f"  Condition number:  {result2.condition_estimate:.2e}")

    # --- Case 3: Near-singular J (sigma_min ~ 1e-10), m = n ---
    # J is technically invertible (det != 0), but extremely ill-conditioned.
    # The sigma_min estimate and condition number expose this.
    J_near_sing = torch.randn(n, n, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(J_near_sing)
    S[-1] = 1e-10   # plant one very small singular value
    J_near_sing = U @ torch.diag(S) @ Vh
    Jv_near_sing = lambda v: J_near_sing @ v

    result3 = check_invertibility(Jv_near_sing, n=n, m=n, seed=42)
    print("\n=== Near-singular J (sigma_min ~ 1e-10, m = n) ===")
    print(f"  Invertible:        {result3.is_invertible}")
    print(f"  Rank estimate:     {result3.rank_estimate} / {n}")
    print(f"  sigma_min(J) est:  {result3.sigma_min_estimate:.2e}  <-- near-singularity detected")
    print(f"  Condition number:  {result3.condition_estimate:.2e}  <-- huge, J is numerically unusable")

    # --- Case 4: Probabilistic certificate (m << n), low-rank deficiency ---
    # If the deficiency is large (rank r << n), you can detect it with m slightly
    # above r, without spending n matrix-vector products.
    # Here J has rank 20 (deficiency 80). We use m = 30 probes (> 20).
    J_low_rank = torch.randn(n, 20, dtype=torch.float64) @ torch.randn(20, n, dtype=torch.float64)
    Jv_low_rank = lambda v: J_low_rank @ v

    result4 = check_invertibility(Jv_low_rank, n=n, m=30, seed=42)
    print("\n=== Low-rank J (rank 20), probabilistic (m = 30) ===")
    print(f"  Invertible:        {result4.is_invertible}")
    print(f"  Rank estimate:     {result4.rank_estimate} / 30 probes")
    print(f"  sigma_min(J) est:  {result4.sigma_min_estimate:.2e}")
    print(f"  Condition number:  {result4.condition_estimate:.2e}")