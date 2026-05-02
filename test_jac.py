import sys
import time
import torch
from torch.func import jacrev
import tqdm
sys.path.insert(0, "src")
import hf_jacobian as hj
from hf_jacobian.jacobian import _layers, _sublayer_fn, capture_all_hidden, _causal_block_jac

# ── config ────────────────────────────────────────────────────────────────────
TEXT        = "hello world, my name is Preslav. I am a "
MODEL_NAME  = "gpt2"
LAYERS      = [0, 1, 2,3,4,5,7,8,9,10,11]          # which transformer layers to probe
SUBLAYERS   = ["attn", "ffn"]    # sublayers within each layer
CHEB_DEGREE = 60                 # Chebyshev polynomial degree
CHEB_NVECS  = 30                 # Hutchinson probe vectors
CHEB_EPS    = 1e-4               # lower spectral bound: a = eps * lambda_max
# ─────────────────────────────────────────────────────────────────────────────


def run():
    model, tok = hj.load(MODEL_NAME)
    inputs = torch.randint(100,(1024,))
    # tokens = [tok.decode([t]) for t in inputs[0].tolist()]
    seq    = len(inputs)
    dtype  = next(model.parameters()).dtype
    # print(inputs)
    dets = []
    # t0 = time.perf_counter()
    total_steps = len(LAYERS)*len(SUBLAYERS)*seq
    with tqdm.tqdm(total=total_steps, desc="samples") as pbar:
        for layer_idx in LAYERS:
            for sub in SUBLAYERS:      
                hidden_out, jac = _causal_block_jac(
                                    model, inputs, layer_idx, sub, 768)
                jac = jac.squeeze(0)
                # print(jac.shape)
                
                for i in range(seq):
                    sv = torch.linalg.svdvals(jac[i])              # (d,) descending
                    dets.append( sv.log().sum().item())
                    pbar.update(1)
    
    
    return dets    


def make_f_local(fn, h_seq, p):
    """Return f_local: R^d -> R^d, the per-token sublayer map at position p.

    h_seq is the full frozen context (seq, d). f_local replaces only token p,
    runs the sublayer, and returns the output at p. This gives the local d×d
    Jacobian dout[p]/din[p].
    """
    ctx = h_seq.detach().clone()

    def f_local(x, _ctx=ctx, _p=p):
        full = torch.cat([_ctx[:_p], x.unsqueeze(0), _ctx[_p + 1:]], dim=0)
        return fn(full.unsqueeze(0)).squeeze(0)[_p]

    return f_local
      
def run1():
    model, tok = hj.load(MODEL_NAME)
    inputs = torch.randint(100,(128,))
    # tokens = [tok.decode([t]) for t in inputs[0].tolist()]
    seq    = len(inputs)
    dtype  = next(model.parameters()).dtype
   
    store = capture_all_hidden(model, inputs)
    layers = _layers(model)
    dets = []
    
    total_steps = len(LAYERS)*len(SUBLAYERS)*seq
    # with tqdm.tqdm(total=total_steps, desc="samples") as pbar:
    for layer_idx in LAYERS:
        for sublayer in SUBLAYERS:
            key   = (layer_idx, sublayer)
            if key not in store:
                continue
            h    = store[key]        # (1, seq, d)
            fn   = _sublayer_fn(layers[layer_idx], sublayer, model=model)

            for p in range(seq):
                t0 = time.perf_counter()
                
                f_local = make_f_local(fn, h[0], p)
                x_p     = h[0, p].detach().clone()

                jac = jacrev(f_local)(x_p).float() 
                sv  = torch.linalg.svdvals(jac)              # (d,) descending
                dets.append( sv.log().sum().item())

                
                print("time per iter:", time.perf_counter()-t0)
            
    return dets


def hutchinson_trace(f_local, x, n_vecs=30):
    """Estimate Tr(J) via Hutchinson: E[v^T J v] with v ~ N(0, I).

    Uses JVPs so memory is O(d) rather than O(d^2).
    """
    from torch.func import jvp
    estimates = []
    for _ in range(n_vecs):
        v = torch.randn_like(x)
        _, jv = jvp(f_local, (x,), (v,))   # jv = J @ v
        estimates.append((v * jv).sum())
    return torch.stack(estimates).mean()


def compare_trace(n_positions=8, n_vecs=CHEB_NVECS):
    """Compare Hutchinson trace estimate vs exact tr(J) from full Jacobian."""
    model, tok = hj.load(MODEL_NAME)
    inputs = torch.randint(100, (128,))
    store = capture_all_hidden(model, inputs)
    layers = _layers(model)

    print(f"{'layer':>5} {'sub':>4} {'pos':>3}  {'tr_exact':>12} {'tr_hutch':>12} {'rel_err':>9}  {'t_exact':>8} {'t_hutch':>8}")
    print("-" * 75)

    for layer_idx in LAYERS[:3]:          # keep output short; change slice as needed
        for sublayer in SUBLAYERS:
            key = (layer_idx, sublayer)
            if key not in store:
                continue
            h  = store[key]
            fn = _sublayer_fn(layers[layer_idx], sublayer, model=model)

            for p in range(n_positions):
                f_local = make_f_local(fn, h[0], p)
                x_p     = h[0, p].detach().clone()

                t0 = time.perf_counter()
                jac = jacrev(f_local)(x_p).float()
                tr_exact = jac.diagonal().sum().item()
                t_exact = time.perf_counter() - t0

                t0 = time.perf_counter()
                tr_hutch = hutchinson_trace(f_local, x_p, n_vecs=n_vecs).item()
                t_hutch = time.perf_counter() - t0

                rel_err = abs(tr_hutch - tr_exact) / (abs(tr_exact) + 1e-8)
                print(f"{layer_idx:>5} {sublayer:>4} {p:>3}  {tr_exact:>12.4f} {tr_hutch:>12.4f} {rel_err:>9.4f}  {t_exact:>8.3f}s {t_hutch:>8.3f}s")


if __name__ == "__main__":
    compare_trace()