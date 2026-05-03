import sys; sys.path.insert(0, 'src')
import torch, numpy as np
from hf_jacobian.id_estimators import ess

X = torch.rand(500, 768)

for k in [25, 50, 100]:
    for ver in ['a', 'b']:
        r = ess(X, k=k, ver=ver, d=1)
        print(f'k={k:3d} ver={ver}  ESS={r["dimension"]:.2f}')