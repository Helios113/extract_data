"""Print approx_sigma_min from a run.py HDF5 output file."""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "out/smoke/gpt2_approx_sigma.h5"

with h5py.File(path, "r") as f:
    m        = f["meta"].attrs
    n_layers = int(m["n_layers"])
    n_samples= int(m["n_samples"])
    seq_len  = int(m["seq_len"])
    sublayers= ("block",) if m["source_type"] == "GPTNeoXModel" else ("attn", "ffn")

    # detect sublayers from actual keys
    sublayers = []
    for sub in ("attn", "ffn", "block"):
        if f"layer_0/{sub}/approx_sigma_min" in f:
            sublayers.append(sub)

    print(f"model={m['model']}  n_layers={n_layers}  n_samples={n_samples}  seq_len={seq_len}")
    print()

    # header: sample mean across positions and samples
    hdr = f"{'layer':>5} {'sub':>5}  {'mean_σ_min':>10}  {'min_σ_min':>10}  {'max_σ_min':>10}  {'%invertible':>12}"
    print(hdr)
    print("-" * len(hdr))

    for layer_idx in range(n_layers):
        for sub in sublayers:
            ds  = f[f"layer_{layer_idx}/{sub}/approx_sigma_min"][:]   # (n_samples, seq_len)
            inv = f[f"layer_{layer_idx}/{sub}/is_invertible"][:]       # (n_samples, seq_len)
            print(f"  {layer_idx:>3}  {sub:>5}  "
                  f"{ds.mean():>10.4f}  {ds.min():>10.4f}  {ds.max():>10.4f}  "
                  f"{100*inv.mean():>11.1f}%")
