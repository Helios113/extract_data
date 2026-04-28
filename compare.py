"""
Compare hook-based extraction (jacobian.py dispatch machinery, via extract.py)
against direct extraction (calling AttnResidual / FFNResidual on a leaf directly).

Both paths should produce identical hidden states and Jacobian stats.
"""

import torch
import hf_jacobian as hj
from hf_jacobian.custom_model import CustomModel, Config, extract_direct
from extract import extract_target

torch.manual_seed(0)
cfg   = Config(d_model=32, n_heads=4, n_layers=2, vocab_size=64)
model = CustomModel(cfg).eval()
ids   = torch.randint(0, cfg.vocab_size, (1, 5))

hj.print_model(model)

all_pass = True

for layer_idx in range(cfg.n_layers):
    for sublayer in ("attn", "ffn"):
        tag = f"layer {layer_idx} / {sublayer}"
        print(f"\n{'─' * 55}")
        print(f"  {tag}")

        h_hook, s_hook = extract_target(model, ids, layer_idx, sublayer)
        h_dir,  s_dir  = extract_direct(model, ids, layer_idx, sublayer)
        # both return (B=1, seq, d) / (B=1, seq) — compare at batch index 0
        hidden_diff = (h_hook[0] - h_dir[0]).abs().max().item()
        det_diff    = (s_hook["det"][0]         - s_dir["det"][0]).abs().max().item()
        ratio_diff  = (s_hook["sigma_ratio"][0] - s_dir["sigma_ratio"][0]).abs().max().item()

        ok = hidden_diff < 1e-5 and det_diff < 1e-5 and ratio_diff < 1e-5
        all_pass = all_pass and ok

        print(f"  hidden    max|Δ| : {hidden_diff:.2e}")
        print(f"  det       max|Δ| : {det_diff:.2e}")
        print(f"  σ_ratio   max|Δ| : {ratio_diff:.2e}")
        print(f"  {'✓ match' if ok else '✗ MISMATCH'}")

print(f"\n{'=' * 55}")
print(f"Overall: {'✓ all match' if all_pass else '✗ MISMATCHES FOUND'}")
