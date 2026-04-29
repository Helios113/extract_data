"""
Benchmark Jacobian implementations for a single (layer, sublayer) block.

The Jacobian is d(h + g(LN(h)))/dh at each sequence position p — LN is inside
the graph so the derivative is w.r.t. the raw residual h, not LN(h).

For causal attention, position p's output depends on x[0:p+1], so the forward
must run the full prefix. Only the gradient at position p is kept; earlier
positions are discarded — unavoidable for causal attention.

Methods
-------
loop        baseline: chunked autograd.grad, one backward per chunk of output dims
            peak VRAM: jac_chunk × (p+1) × d per position
loop_vmap   option 4: replace is_grads_batched with vmap(grad) for the chunk dim
            same memory profile, eliminates Python overhead inside the chunk loop
loop_compile option 3: torch.compile applied to fn before running loop
            zero memory change, reduces kernel-launch overhead from repeated backwards

Usage
-----
  python benchmark_jac.py --device cuda
  python benchmark_jac.py --device cuda --jac_chunk 128
"""

import argparse
import gc
import time

import torch
from torch.func import jacrev, grad, vmap

from src.hf_jacobian.custom_model import Config, CustomModel
from src.hf_jacobian.jacobian import _layers, _sub, SUBLAYERS


# ─── sublayer function ───────────────────────────────────────────────────────

def _make_fn(model, layer_idx, sublayer):
    """
    Returns f where f(x_prefix: (t, d)) → (t, d) is the full residual:
      h + g(LN(h))
    Differentiating this w.r.t. h gives I + dg/dh via chain rule through LN.
    """
    layer = _layers(model)[layer_idx]
    _, ln_mod  = _sub(layer, *SUBLAYERS[sublayer][0])
    _, sub_mod = _sub(layer, *SUBLAYERS[sublayer][1])

    def f(x):                            # x: (t, d)
        out = sub_mod(ln_mod(x.unsqueeze(0)))
        out = out[0] if isinstance(out, tuple) else out
        return x + out.squeeze(0)        # (t, d)

    return f


# ─── method 1: chunked autograd loop ─────────────────────────────────────────

def jac_loop(fn, x, jac_chunk):
    """x: (seq, d) → jac: (seq, d, d)"""
    seq, d = x.shape
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    jac = torch.zeros(seq, d, d, device=x.device, dtype=x.dtype)

    for p in range(seq):
        x_p = x[:p+1].detach().clone().requires_grad_(True)
        out = fn(x_p)
        for i0 in range(0, d, jac_chunk):
            i1 = min(i0 + jac_chunk, d)
            g = torch.autograd.grad(
                out[p], x_p,
                grad_outputs=eye[i0:i1],
                is_grads_batched=True,
                retain_graph=(i1 < d),
            )[0]                         # (chunk, p+1, d)
            jac[p, i0:i1] = g[:, p]     # (chunk, d)

    return jac                           # (seq, d, d)


# ─── method 2: jacrev on f_p ─────────────────────────────────────────────────

def jac_jacrev(fn, x):
    """x: (seq, d) → jac: (seq, d, d)

    For each position p, defines f_p(x_prefix) = fn(x_prefix)[p] → (d,).
    jacrev(f_p)(x_prefix) → (d, p+1, d); slice [:, p, :] gives the (d, d) block.
    Peak allocation: (d, p+1, d) per position — no cross-position blocks.
    """
    seq, d = x.shape
    jac = torch.zeros(seq, d, d, device=x.device, dtype=x.dtype)

    for p in range(seq):
        x_p = x[:p+1].detach()
        f_p = lambda xp, _p=p: fn(xp)[_p]   # (p+1, d) → (d,)
        j = jacrev(f_p)(x_p)                 # (d, p+1, d)
        jac[p] = j[:, p, :]                  # (d, d)

    return jac                               # (seq, d, d)


# ─── method 4: vmap over chunk dim ───────────────────────────────────────────

def jac_loop_vmap(fn, x, jac_chunk):
    """
    Same memory profile as jac_loop (chunk × (p+1) × d per position).
    Replaces is_grads_batched with vmap over seed vectors — one vmapped
    backward call per chunk instead of a Python loop over scalar grads.
    """
    seq, d = x.shape
    eye  = torch.eye(d, device=x.device, dtype=x.dtype)
    jac  = torch.zeros(seq, d, d, device=x.device, dtype=x.dtype)

    for p in range(seq):
        x_p = x[:p+1].detach()

        # torch.func.grad composes with vmap; seed · out[p] is a scalar
        def scalar_fn(xp, seed, _p=p):
            return (fn(xp)[_p] * seed).sum()

        # vmap over seeds: (chunk, d) → (chunk, p+1, d)
        batched_grad = vmap(grad(scalar_fn), in_dims=(None, 0))

        for i0 in range(0, d, jac_chunk):
            i1 = min(i0 + jac_chunk, d)
            seeds = eye[i0:i1]                    # (chunk, d)
            g = batched_grad(x_p, seeds)          # (chunk, p+1, d)
            jac[p, i0:i1] = g[:, p, :]

    return jac


# ─── timing + memory ─────────────────────────────────────────────────────────

def measure(fn, device, n_warmup=1, n_runs=3):
    for _ in range(n_warmup):
        result = fn()
        if device == "cuda":
            torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
    return result, min(times), mem_mb


# ─── single cell ─────────────────────────────────────────────────────────────

def _flush(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def benchmark_cell(d, seq, device, jac_chunk, sublayer="ffn"):
    cfg    = Config(d_model=d, n_heads=max(1, d // 64), n_layers=2, vocab_size=256)
    model  = CustomModel(cfg).to(device).eval()
    fn     = _make_fn(model, layer_idx=0, sublayer=sublayer)
    x = torch.randn(seq, d, device=device)

    _flush(device)
    ref, t_loop, m_loop = measure(lambda: jac_loop(fn, x, jac_chunk), device)

    _flush(device)
    r2, t_rev, m_rev = measure(lambda: jac_jacrev(fn, x), device)

    _flush(device)
    r3, t_vmap, m_vmap = measure(lambda: jac_loop_vmap(fn, x, jac_chunk), device)

    return {
        "d": d, "seq": seq,
        "loop_s":  t_loop, "loop_mb":  m_loop,
        "rev_s":   t_rev,  "rev_mb":   m_rev,  "rev_err":  (ref - r2).abs().max().item(),
        "vmap_s":  t_vmap, "vmap_mb":  m_vmap, "vmap_err": (ref - r3).abs().max().item(),
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",    default="cpu")
    p.add_argument("--jac_chunk", type=int, default=64)
    args = p.parse_args()

    grid = [
        (64,  8), (64,  32),
        (256, 8), (256, 32),
        (768, 8), (768, 32),
    ]

    mem = lambda v: f"{v:8.1f}" if v == v else "      n/a"
    hdr = (f"{'d':>5} {'seq':>5} {'chunk':>6} | "
           f"{'loop(s)':>7} {'(MB)':>8} | "
           f"{'jacrev(s)':>9} {'(MB)':>8} {'err':>8} | "
           f"{'vmap(s)':>7} {'(MB)':>8} {'err':>8}")
    print(hdr)
    print("-" * len(hdr))

    for d, seq in grid:
        _flush(args.device)
        try:
            r = benchmark_cell(d, seq, args.device, args.jac_chunk)
            print(
                f"{r['d']:>5} {r['seq']:>5} {args.jac_chunk:>6} | "
                f"{r['loop_s']:>7.3f} {mem(r['loop_mb'])} | "
                f"{r['rev_s']:>9.3f} {mem(r['rev_mb'])} {r['rev_err']:>8.2e} | "
                f"{r['vmap_s']:>7.3f} {mem(r['vmap_mb'])} {r['vmap_err']:>8.2e}"
            )
        except Exception as e:
            print(f"{d:>5} {seq:>5} {args.jac_chunk:>6} | ERROR: {e}")


if __name__ == "__main__":
    main()
