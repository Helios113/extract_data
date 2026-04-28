"""
Comparison of PyTorch TwoNN/ESS implementations against skdim references.

Datasets from skdim.datasets with known intrinsic dimension:
  - hyperBall(d=3)  : ID = 3
  - hyperBall(d=7)  : ID = 7
  - hyperSphere(d=4): ID = 4  (surface of S^4 embedded in R^5)
  - hyperSphere(d=8): ID = 8
"""

import numpy as np
import torch
import skdim

from hf_jacobian.id_estimators import twonn, ess

SEED = 42
N = 800
K = 50  # ESS neighborhood size

datasets = [
    ("hyperBall d=3",  skdim.datasets.hyperBall(N, d=3,  random_state=SEED), 3),
    ("hyperBall d=7",  skdim.datasets.hyperBall(N, d=7,  random_state=SEED), 7),
    ("hyperSphere d=4", skdim.datasets.hyperSphere(N, d=4, random_state=SEED), 4),
    ("hyperSphere d=8", skdim.datasets.hyperSphere(N, d=8, random_state=SEED), 8),
]


def _run_twonn(X_np, X_t):
    ref = skdim.id.TwoNN().fit(X_np).dimension_
    ours = twonn(X_t)
    return ref, ours


def _run_ess(X_np, X_t, ver):
    ref_est = skdim.id.ESS(ver=ver).fit(X_np, n_neighbors=K)
    ref = ref_est.dimension_

    ours_out = ess(X_t, k=K, ver=ver, seed=SEED)
    ours = ours_out["dimension"]
    return ref, ours


def _row(label, true_id, ref, ours):
    delta = abs(ref - ours)
    ref_err = abs(ref - true_id)
    ours_err = abs(ours - true_id)
    return f"  {label:<22} true={true_id}  skdim={ref:.3f} (err={ref_err:.3f})  torch={ours:.3f} (err={ours_err:.3f})  |Δ|={delta:.4f}"


print("=" * 80)
print("TwoNN")
print("=" * 80)
for name, X_np, true_id in datasets:
    X_t = torch.tensor(X_np, dtype=torch.float32)
    ref, ours = _run_twonn(X_np, X_t)
    print(_row(name, true_id, ref, ours))

print()
print("=" * 80)
print("ESS  ver='a'")
print("=" * 80)
for name, X_np, true_id in datasets:
    X_t = torch.tensor(X_np, dtype=torch.float32)
    ref, ours = _run_ess(X_np, X_t, ver="a")
    print(_row(name, true_id, ref, ours))

print()
print("=" * 80)
print("ESS  ver='b'")
print("=" * 80)
for name, X_np, true_id in datasets:
    X_t = torch.tensor(X_np, dtype=torch.float32)
    ref, ours = _run_ess(X_np, X_t, ver="b")
    print(_row(name, true_id, ref, ours))
