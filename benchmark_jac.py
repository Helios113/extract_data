"""
Benchmark: time-per-token for approx_sigma and full Jacobian across models/seq lengths.

approx_sigma: times a single token (position 0), reports sec/token and extrapolated
              total for the full sequence. The cost per token is independent of seq_len
              for FFN (pointwise), but grows with seq_len for attention (full context).

full Jacobian: times the entire sequence at once via _jac_batched, then divides.

Runs approx_sigma first, then full Jacobian. seq_len capped at 64.
OOM is caught and reported rather than crashing.

Usage:
    uv run python benchmark_jac.py
"""

import time
import traceback

import torch

import hf_jacobian as hj
from hf_jacobian.jacobian import _layers, _sublayer_fn, capture_all_hidden, _jac_batched
from hf_jacobian.custom_model import CustomModel as _CM
from test_jac2 import check_invertibility
from torch.func import jvp as _jvp

MODELS = [
    "openai-community/gpt2",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen3-0.6B",
    "EleutherAI/pythia-160m",
]

SEQ_LENS      = [4, 8, 16, 32, 64]
DEVICE        = "cuda"
APPROX_PROBES = 64
B             = 1


def _random_batch(model, seq_len):
    d = model.config.hidden_size
    return torch.randn(B, seq_len, d, device=DEVICE, dtype=next(model.parameters()).dtype)


def _sublayers_for(model):
    return ("block",) if type(model).__name__ == "GPTNeoXModel" else ("attn", "ffn")


def _sync_time(fn):
    """Run fn(), return (result, elapsed_s) with CUDA sync."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0


# ─── approx sigma ─────────────────────────────────────────────────────────────

def bench_approx_sigma(model, model_name):
    """
    Times one token (position 0) per (seq_len, sublayer) combination.
    sec/token is the measured cost; est_total = sec/token * seq_len.
    For attention, sec/token grows with seq_len because the full context is used.
    For FFN, it should be roughly constant.
    """
    print(f"\n{'='*66}")
    print(f"  approx_sigma  |  {model_name}")
    print(f"{'='*66}")
    print(f"  {'seq':>4}  {'sub':<6}  {'sec/tok':>8}  {'est_total':>10}  {'sigma_min':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}")

    layer   = _layers(model)[0]
    d_model = model.config.hidden_size

    for seq_len in SEQ_LENS:
        batch = _random_batch(model, seq_len)
        store = capture_all_hidden(model, batch)

        for sub in _sublayers_for(model):
            if (0, sub) not in store:
                continue

            h_B = store[(0, sub)].float().to(DEVICE)  # (B, seq, d)
            fn  = _sublayer_fn(layer, sub, model=model)
            _fn = (lambda x, _f=fn: _f(x.unsqueeze(0)).squeeze(0)
                   if not isinstance(model, _CM) else fn)

            h   = h_B[0]                   # (seq, d)
            ctx = h.detach().clone()
            x_p = h[0].detach().clone()

            def f_loc(x, _ctx=ctx, _f=_fn):
                full = _ctx.clone()
                full[0] = x
                return _f(full)[0]

            Jv_fn = lambda v, _f=f_loc, _x=x_p: _jvp(_f, (_x,), (v,))[1]

            try:
                res, elapsed = _sync_time(
                    lambda: check_invertibility(
                        Jv_fn, n=d_model, m=APPROX_PROBES,
                        device=DEVICE, dtype=torch.float32,
                    )
                )
                est_total = elapsed * seq_len
                print(f"  {seq_len:>4}  {sub:<6}  {elapsed:>8.3f}  {est_total:>10.1f}  {res.sigma_min_estimate:>10.4f}")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {seq_len:>4}  {sub:<6}  {'OOM':>8}")
            except Exception:
                msg = traceback.format_exc().splitlines()[-1]
                print(f"  {seq_len:>4}  {sub:<6}  {'ERROR':>8}  {msg}")

        torch.cuda.empty_cache()


# ─── full Jacobian ─────────────────────────────────────────────────────────────

def bench_full_jacobian(model, model_name):
    """
    Times _jac_batched over the full sequence, then reports sec/token = total / seq_len.
    """
    print(f"\n{'='*66}")
    print(f"  full Jacobian  |  {model_name}")
    print(f"{'='*66}")
    print(f"  {'seq':>4}  {'sub':<6}  {'sec/tok':>8}  {'total':>8}  {'shape'}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*20}")

    layer = _layers(model)[0]

    for seq_len in SEQ_LENS:
        batch = _random_batch(model, seq_len)
        store = capture_all_hidden(model, batch)

        for sub in _sublayers_for(model):
            if (0, sub) not in store:
                continue

            h_B = store[(0, sub)].to(
                device=next(layer.parameters()).device,
                dtype=next(layer.parameters()).dtype,
            )
            fn = _sublayer_fn(layer, sub, model=model)

            try:
                jac, elapsed = _sync_time(lambda: _jac_batched(fn, h_B, sublayer=sub))
                print(f"  {seq_len:>4}  {sub:<6}  {elapsed/seq_len:>8.3f}  {elapsed:>8.3f}  {list(jac.shape)}")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {seq_len:>4}  {sub:<6}  {'OOM':>8}")
            except Exception:
                msg = traceback.format_exc().splitlines()[-1]
                print(f"  {seq_len:>4}  {sub:<6}  {'ERROR':>8}  {msg}")

        torch.cuda.empty_cache()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    import warnings; warnings.filterwarnings("ignore")

    for model_name in MODELS:
        print(f"\n\nLoading {model_name!r} ...")
        try:
            model, _ = hj.load(model_name, device=DEVICE)
            model = model.float()
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        bench_approx_sigma(model, model_name)
        bench_full_jacobian(model, model_name)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
