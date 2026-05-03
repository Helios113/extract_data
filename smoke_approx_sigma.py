"""
Smoke test: run compute_approx_sigma on GPT-2 and Qwen3-0.6B, save to HDF5, print results.
"""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import h5py
import numpy as np
import torch

import hf_jacobian as hj
from hf_jacobian.jacobian import _layers, _sublayer_fn, capture_all_hidden
from hf_jacobian.jacobian import _layers
from hf_jacobian.invertibility import check_invertibility
from torch.func import jvp as _jvp

MODELS = [
    ("openai-community/gpt2",  "gpt2"),
    ("Qwen/Qwen3-0.6B",        "qwen3-0.6b"),
]
N_PROBES  = 32
SEQ_LEN   = 8
N_SAMPLES = 2   # batches of 1 each
DEVICE    = "cuda"
TEXT      = "The quick brown fox jumps over the lazy dog"

OUT_DIR = Path(__file__).parent.parent / "out" / "smoke_approx_sigma"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_model(model_name, tag):
    print(f"\n{'='*60}")
    print(f"  {tag}  ({model_name})")
    print(f"{'='*60}")

    model, tok = hj.load(model_name, device=DEVICE)
    d_model    = model.config.hidden_size
    n_layers   = len(_layers(model))
    arch       = type(model).__name__
    sublayers  = ("block",) if arch == "GPTNeoXModel" else ("attn", "ffn")

    ids = tok(TEXT, return_tensors="pt").input_ids.to(DEVICE)
    # trim or repeat to SEQ_LEN
    if ids.shape[1] >= SEQ_LEN:
        ids = ids[:, :SEQ_LEN]
    else:
        ids = ids.repeat(1, SEQ_LEN // ids.shape[1] + 1)[:, :SEQ_LEN]

    out_path = OUT_DIR / f"{tag}.h5"
    results  = {}   # (layer, sub) -> list of sigma_min per position

    store = capture_all_hidden(model, ids)

    with h5py.File(out_path, "w") as f:
        f.attrs["model"]    = model_name
        f.attrs["d_model"]  = d_model
        f.attrs["n_layers"] = n_layers
        f.attrs["seq_len"]  = ids.shape[1]

        for layer_idx in range(n_layers):
            layer = _layers(model)[layer_idx]
            for sub in sublayers:
                key = (layer_idx, sub)
                if key not in store:
                    continue
                h_B = store[key].to(DEVICE)       # (1, seq, d)
                fn  = _sublayer_fn(layer, sub, model=model)

                from hf_jacobian.custom_model import CustomModel as _CM
                _model_dtype = next(model.parameters()).dtype
                _needs_upcast = _model_dtype in (torch.float16, torch.bfloat16)
                if _needs_upcast:
                    layer.float()
                if not isinstance(model, _CM):
                    _fn = lambda x, _f=fn: _f(x.unsqueeze(0)).squeeze(0)
                else:
                    _fn = fn

                smin_seq = []
                for p in range(h_B.shape[1]):
                    ctx = h_B[0].detach().clone().float()
                    def f_loc(x, _ctx=ctx, _p=p, _f=_fn):
                        full = _ctx.clone()
                        full[_p] = x
                        return _f(full)[_p]
                    x_p   = ctx[p].clone()
                    Jv_fn = lambda v, _fl=f_loc, _x=x_p: _jvp(_fl, (_x,), (v,))[1]
                    res   = check_invertibility(Jv_fn, n=d_model, m=N_PROBES,
                                               device=DEVICE, dtype=torch.float32)
                    smin_seq.append(res.sigma_min_estimate)

                if _needs_upcast:
                    layer.to(_model_dtype)

                arr = np.array(smin_seq, dtype=np.float32)
                grp = f.require_group(f"layer_{layer_idx}/{sub}")
                grp.create_dataset("approx_sigma_min", data=arr)
                results[key] = arr

    print(f"  Saved → {out_path}")
    return out_path, results, sublayers, n_layers


def print_results(path, results, sublayers, n_layers):
    print(f"\n--- Reading back {path} ---")
    with h5py.File(path, "r") as f:
        dm = f.attrs["d_model"]
        nl = f.attrs["n_layers"]
        sl = f.attrs["seq_len"]
        print(f"  d_model={dm}  n_layers={nl}  seq_len={sl}\n")

        hdr = f"{'layer':>5} {'sub':>5}  {'sigma_min (per position)':}"
        print(hdr)
        print("-" * 60)
        for layer_idx in range(nl):
            for sub in sublayers:
                ds_path = f"layer_{layer_idx}/{sub}/approx_sigma_min"
                if ds_path not in f:
                    continue
                vals = f[ds_path][:]
                vals_str = "  ".join(f"{v:.4f}" for v in vals)
                print(f"  {layer_idx:>3}  {sub:>5}  [{vals_str}]")


for model_name, tag in MODELS:
    path, results, sublayers, n_layers = run_model(model_name, tag)
    print_results(path, results, sublayers, n_layers)

print("\nDone.")
