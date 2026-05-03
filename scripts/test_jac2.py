import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_jacobian.invertibility import InvertibilityResult, check_invertibility

__all__ = ["InvertibilityResult", "check_invertibility"]


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    n = 100

    J_full = torch.randn(n, n, dtype=torch.float64)
    result = check_invertibility(lambda v: J_full @ v, n=n, m=n, seed=42)
    print("=== Full-rank J (m = n, deterministic) ===")
    print(f"  Invertible:        {result.is_invertible}")
    print(f"  Rank estimate:     {result.rank_estimate} / {n}")
    print(f"  sigma_min(J) est:  {result.sigma_min_estimate:.4f}")
    print(f"  sigma_max(J) est:  {result.sigma_max_estimate:.4f}")
    print(f"  Condition number:  {result.condition_estimate:.4f}")

    J_rank_def = torch.randn(n, n, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(J_rank_def)
    S[-5:] = 0.0
    J_rank_def = U @ torch.diag(S) @ Vh
    result2 = check_invertibility(lambda v: J_rank_def @ v, n=n, m=n, seed=42)
    print("\n=== Rank-deficient J, rank = n-5 (m = n, deterministic) ===")
    print(f"  Invertible:        {result2.is_invertible}")
    print(f"  Rank estimate:     {result2.rank_estimate} / {n}")
    print(f"  sigma_min(J) est:  {result2.sigma_min_estimate:.2e}")
    print(f"  Condition number:  {result2.condition_estimate:.2e}")

    J_near_sing = torch.randn(n, n, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(J_near_sing)
    S[-1] = 1e-10
    J_near_sing = U @ torch.diag(S) @ Vh
    result3 = check_invertibility(lambda v: J_near_sing @ v, n=n, m=n, seed=42)
    print("\n=== Near-singular J (sigma_min ~ 1e-10, m = n) ===")
    print(f"  Invertible:        {result3.is_invertible}")
    print(f"  Rank estimate:     {result3.rank_estimate} / {n}")
    print(f"  sigma_min(J) est:  {result3.sigma_min_estimate:.2e}")
    print(f"  Condition number:  {result3.condition_estimate:.2e}")

    J_low_rank = torch.randn(n, 20, dtype=torch.float64) @ torch.randn(20, n, dtype=torch.float64)
    result4 = check_invertibility(lambda v: J_low_rank @ v, n=n, m=30, seed=42)
    print("\n=== Low-rank J (rank 20), probabilistic (m = 30) ===")
    print(f"  Invertible:        {result4.is_invertible}")
    print(f"  Rank estimate:     {result4.rank_estimate} / 30 probes")
    print(f"  sigma_min(J) est:  {result4.sigma_min_estimate:.2e}")
    print(f"  Condition number:  {result4.condition_estimate:.2e}")
