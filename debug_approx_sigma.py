"""
Quick debug script for compute_approx_sigma on GPU.
Runs a single batch (B=1, seq=4, gpt2) through the approx-sigma path.
"""
import torch
from torch.func import jvp as _jvp

import hf_jacobian as hj
from hf_jacobian.jacobian import _layers, _sublayer_fn, capture_all_hidden
from hf_jacobian.custom_model import CustomModel as _CM
from test_jac2 import check_invertibility

device = "cuda"
model_name = "openai-community/gpt2"
n_samples = 1
seq_len = 4
approx_sigma_probes = 16

print(f"Loading {model_name!r} on {device} ...")
model, tok = hj.load(model_name, device=device)
d_model = model.config.hidden_size
print(f"d_model={d_model}")

# Build a tiny random-token batch
emb_weight = model.get_input_embeddings().weight
ids = torch.randint(0, emb_weight.shape[0], (seq_len,))
batch = emb_weight[ids].detach().unsqueeze(0).to(device)  # (1, seq, d)
print(f"batch shape: {batch.shape}, dtype: {batch.dtype}, device: {batch.device}")

# Run full forward to capture hidden states
store = capture_all_hidden(model, batch)
print(f"store keys: {list(store.keys())[:6]} ...")

layers = _layers(model)
n_layers = len(layers)
sublayers = ("attn", "ffn")

for layer_idx in range(min(2, n_layers)):
    for sub in sublayers:
        if (layer_idx, sub) not in store:
            print(f"  layer {layer_idx} {sub}: not in store, skipping")
            continue

        h_B = store[(layer_idx, sub)].to(device)  # (B, seq, d)
        layer = layers[layer_idx]

        fn = _sublayer_fn(layer, sub, model=model)
        _model_dtype = next(model.parameters()).dtype
        _probe_dtype = torch.float32
        _needs_upcast = _model_dtype in (torch.float16, torch.bfloat16)
        if _needs_upcast:
            layer.float()
        if not isinstance(model, _CM):
            _fn = lambda x, _f=fn: _f(x.unsqueeze(0)).squeeze(0)
        else:
            _fn = fn

        B = h_B.shape[0]
        smin_batch = []
        inv_batch  = []
        for b in range(B):
            h = h_B[b].to(_probe_dtype)  # (seq, d)
            smin_seq = []
            inv_seq  = []
            for p in range(h.shape[0]):
                ctx = h.detach().clone()
                def f_loc(x, _ctx=ctx, _p=p, _f=_fn):
                    full = _ctx.clone()
                    full[_p] = x
                    return _f(full)[_p]
                x_p   = h[p].detach().clone()
                Jv_fn = lambda v, _fl=f_loc, _x=x_p: _jvp(_fl, (_x,), (v,))[1]
                res   = check_invertibility(Jv_fn, n=d_model, m=approx_sigma_probes,
                                            device=device, dtype=_probe_dtype)
                smin_seq.append(res.sigma_min_estimate)
                inv_seq.append(float(res.is_invertible))
                print(f"  layer {layer_idx} {sub} b={b} p={p}: "
                      f"sigma_min={res.sigma_min_estimate:.4f}  "
                      f"invertible={res.is_invertible}")
            smin_batch.append(smin_seq)
            inv_batch.append(inv_seq)

        if _needs_upcast:
            layer.to(_model_dtype)

print("Done.")
