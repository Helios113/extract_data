import torch, time, sys
sys.path.insert(0, 'src')
import hf_jacobian as hj
from hf_jacobian.jacobian import _layers, _sublayer_fn, capture_all_hidden, _jac_attn, _jac_ffn
from test_jac2 import check_invertibility
from torch.func import jvp

# ── config ────────────────────────────────────────────────────────────────────
D_MODEL  = 1024
SEQ      = 16     # sequence length for the comparison test
N_PROBES = 128     # probe vectors for check_invertibility
N_POS    = 4      # token positions to compare
# ─────────────────────────────────────────────────────────────────────────────


def _jac_mem_mb(seq, d, sublayer, bytes_per_elem=4):
    """Estimate peak memory (MB) for one Jacobian computation.

    attn: vmap(jacrev) over seq independent (d->d) functions.
          Peak tensor is the output stack: seq * d * d floats.
    ffn:  jacrev on full (seq,d) input builds a (seq,d,seq,d) intermediate.
          Peak tensor: seq^2 * d^2 floats.
    """
    n_elems = seq * d * d if sublayer == "attn" else seq * seq * d * d
    return n_elems * bytes_per_elem / 1024**2


def _approx_time_single(fn, h_seq, n_probes):
    """Time check_invertibility (JVP probes) for a single token position."""
    f_loc = make_f_local(fn, h_seq, 0)
    x_p   = h_seq[0].detach().clone()
    Jv_fn = lambda v, _f=f_loc, _x=x_p: jvp(_f, (_x,), (v,))[1]
    t0 = time.perf_counter()
    check_invertibility(Jv_fn, n=h_seq.shape[-1], m=n_probes)
    return time.perf_counter() - t0


def bench():
    print(f"{'d_model':>6} {'seq':>6}  "
          f"{'attn/tok (s)':>13}  {'attn MB':>8}  "
          f"{'ffn/tok (s)':>12}  {'ffn MB':>8}  "
          f"{'approx attn (s)':>16}  {'approx ffn (s)':>15}")
    print('-' * 100)
    for d_model in [512, 1024, 2048]:
        cfg   = hj.Config(d_model=d_model, n_heads=4, n_layers=2, vocab_size=256, mlp_expand=4)
        model = hj.CustomModel(cfg).eval()
        for seq in [1, 2, 4, 8]:
            inputs  = torch.randint(0, cfg.vocab_size, (seq,))
            store   = capture_all_hidden(model, inputs)
            layers  = _layers(model)

            h_attn  = store[(0, 'attn')][0:1].detach()
            h_ffn   = store[(0, 'ffn')][0:1].detach()
            fn_attn = _sublayer_fn(layers[0], 'attn', model=model)
            fn_ffn  = _sublayer_fn(layers[0], 'ffn',  model=model)

            t0 = time.perf_counter(); _jac_attn(fn_attn, h_attn); t_attn = (time.perf_counter() - t0) / seq
            t0 = time.perf_counter(); _jac_attn(fn_ffn,  h_ffn);  t_ffn  = (time.perf_counter() - t0) / seq
            t_approx_attn = _approx_time_single(fn_attn, h_attn[0], N_PROBES)
            t_approx_ffn  = _approx_time_single(fn_ffn,  h_ffn[0],  N_PROBES)

            m = _jac_mem_mb(seq, d_model, 'attn')
            print(f'{d_model:>6} {seq:>6}  '
                  f'{t_attn:>13.4f}  {m:>8.1f}  '
                  f'{t_ffn:>12.4f}  {m:>8.1f}  '
                  f'{t_approx_attn:>16.4f}  {t_approx_ffn:>15.4f}')


def make_f_local(fn, h_seq, p):
    """f_local: R^d -> R^d — sublayer output at position p with context frozen."""
    ctx = h_seq.detach().clone()

    def f_local(x):
        full = ctx.clone()
        full[p] = x
        return fn(full)[p]
    return f_local


def compare_invertibility():
    cfg    = hj.Config(d_model=D_MODEL, n_heads=4, n_layers=2, vocab_size=256, mlp_expand=4)
    model  = hj.CustomModel(cfg).eval()
    inputs = torch.randint(0, cfg.vocab_size, (SEQ,))
    store  = capture_all_hidden(model, inputs)   # single forward pass
    layers = _layers(model)

    print(f"d_model={D_MODEL}  seq={SEQ}  n_probes={N_PROBES}\n")
    hdr = f"{'sub':>4} {'pos':>3}  {'exact_smin':>12} {'approx_smin':>12}  {'exact_smax':>12} {'approx_smax':>12}  {'exact_rank':>10} {'approx_rank':>11}  {'invertible':>10}  {'t_exact/tok':>11} {'t_approx':>8}"
    print(hdr)
    print("-" * len(hdr))

    for sublayer in ("attn", "ffn"):
        h   = store[(0, sublayer)][0]    # (seq, d)
        fn  = _sublayer_fn(layers[0], sublayer, model=model)
        h_B = h.unsqueeze(0)             # (1, seq, d)

        # ── approx first (cheap) ─────────────────────────────────────────────
        approx_results = []
        for p in range(N_POS):
            x_p   = h[p].detach().clone()
            f_loc = make_f_local(fn, h, p)
            Jv_fn = lambda v, _f=f_loc, _x=x_p: jvp(_f, (_x,), (v,))[1]
            t0     = time.perf_counter()
            result = check_invertibility(Jv_fn, n=D_MODEL, m=N_PROBES)
            t_approx = time.perf_counter() - t0
            approx_results.append((result, t_approx))
            print(f"  {sublayer} pos={p}  approx done  invertible={result.is_invertible}  smin≈{result.sigma_min_estimate:.4f}  ({t_approx:.2f}s)")

        # ── exact second (expensive) ──────────────────────────────────────────
        t0    = time.perf_counter()
        J_all = (_jac_attn if sublayer == "attn" else _jac_ffn)(fn, h_B)
        t_exact_per_tok = (time.perf_counter() - t0) / SEQ

        print()
        for p in range(N_POS):
            J_p        = J_all[0, p].float()
            sv_exact   = torch.linalg.svdvals(J_p)
            smin_exact = sv_exact[-1].item()
            smax_exact = sv_exact[0].item()
            rank_exact = int((sv_exact / sv_exact[0] > 1e-6).sum().item())
            result, t_approx = approx_results[p]
            print(
                f"{sublayer:>4} {p:>3}  "
                f"{smin_exact:>12.4f} {result.sigma_min_estimate:>12.4f}  "
                f"{smax_exact:>12.4f} {result.sigma_max_estimate:>12.4f}  "
                f"{rank_exact:>10} {result.rank_estimate:>11}  "
                f"{str(result.is_invertible):>10}  "
                f"{t_exact_per_tok:>11.3f}s {t_approx:>8.3f}s"
            )
        print()


def make_synthetic_residual(d, n_singular, seed=0):
    """Build a synthetic J_f = I + J_g with planted eigenvalues.

    J_g = Q diag(eigs) Q^T  (symmetric, so eigenvalues = eigs exactly).
    J_f = I + J_g has eigenvalues 1 + eigs[i].
    Setting eigs[i] = -1 makes those eigenvalues of J_f exactly 0 → singular.

    n_singular: how many eigenvalues of J_g are planted at -1.
    Remaining eigenvalues are uniform in (-0.5, 0.5) so J_f is well-conditioned
    everywhere else (eigenvalues of J_f in (0.5, 1.5)).

    Returns (J_f, true_rank) where true_rank = d - n_singular.
    """
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d))   # single orthonormal basis
    eigs = torch.cat([
        torch.full((n_singular,), -1.0),
        torch.rand(d - n_singular) - 0.5,        # in (-0.5, 0.5)
    ])
    J_g = Q @ torch.diag(eigs) @ Q.T
    J_f = torch.eye(d) + J_g
    return J_f, d - n_singular


def compare_singular_detection(n_singulars=(0, 1, 4, 16, 64, 128)):
    """Compare check_invertibility vs exact SVD on synthetic J_f = I + J_g.

    Tests whether the probe method correctly detects singularity and tracks
    sigma_min as we plant increasing numbers of -1 eigenvalues in J_g.
    This is the right test for residual-structured Jacobians: rank deficiency
    in J_g alone doesn't make J_f singular; you need eigenvalue = -1.
    """
    # Use a smaller d so m=d is affordable; the approximation quality
    # is independent of the absolute scale of d.
    d = 1024
    m = d   # full certificate: m=d guarantees detection with probability 1
    print(f"d={d}  n_probes={m} (=d, deterministic certificate)\n")
    hdr = (f"{'n_singular':>10}  "
           f"{'true_rank':>10} {'approx_rank':>11}  "
           f"{'exact_smin':>12} {'approx_smin':>12}  "
           f"{'exact_smax':>12} {'approx_smax':>12}  "
           f"{'invertible':>10}")
    print(hdr)
    print("-" * len(hdr))

    for n_sing in n_singulars:
        J_f, _ = make_synthetic_residual(d, n_sing)

        # exact
        sv_exact   = torch.linalg.svdvals(J_f)
        rank_exact = int((sv_exact / sv_exact[0] > 1e-6).sum())
        smin_exact = sv_exact[-1].item()
        smax_exact = sv_exact[0].item()

        # approx via check_invertibility — only needs Jv products
        Jv_fn = lambda v, _J=J_f.double(): _J @ v
        result = check_invertibility(Jv_fn, n=d, m=m)

        print(f"{n_sing:>10}  "
              f"{rank_exact:>10} {result.rank_estimate:>11}  "
              f"{smin_exact:>12.6f} {result.sigma_min_estimate:>12.6f}  "
              f"{smax_exact:>12.6f} {result.sigma_max_estimate:>12.6f}  "
              f"{str(result.is_invertible):>10}")


def find_singular_input(sublayer="attn", n_steps=200, lr=0.05, seed=0):
    """Gradient descent to find an input x that minimises sigma_min(J_f(x)).

    We optimise over the full hidden sequence h (seq, d), differentiating
    through the Jacobian computation itself. The loss is -sigma_min(J_f(x_0))
    (we minimise sigma_min to push J_f toward singularity at position 0).

    Uses jacrev to get J_f, then slogdet / svdvals to get sigma_min.
    Differentiating through jacrev is expensive but only runs n_steps times.
    """
    torch.manual_seed(seed)
    cfg   = hj.Config(d_model=D_MODEL, n_heads=4, n_layers=2, vocab_size=256, mlp_expand=4)
    model = hj.CustomModel(cfg).eval()

    # initialise from a real forward pass so the starting point is in-distribution
    with torch.no_grad():
        inputs = torch.randint(0, cfg.vocab_size, (SEQ,))
        store  = capture_all_hidden(model, inputs)
        layers = _layers(model)
        h_init = store[(0, sublayer)][0].clone()   # (seq, d)

    fn = _sublayer_fn(layers[0], sublayer, model=model)

    # optimise h as a free parameter
    h = h_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)

    print(f"Searching for near-singular input  sublayer={sublayer}  d={D_MODEL}  seq={SEQ}")
    print(f"{'step':>6}  {'sigma_min':>12}  {'sigma_max':>12}  {'condition':>12}")
    print("-" * 50)

    from torch.func import jacrev as _jacrev

    for step in range(n_steps):
        opt.zero_grad()

        # local Jacobian at position 0: d(out[0])/d(in[0])
        h_det = h.detach()
        ctx   = h_det.clone()

        def f0(x):
            full = ctx.clone()
            full[0] = x
            return fn(full)[0]

        # differentiate through jacrev — this builds J then computes sigma_min
        J = _jacrev(f0)(h[0]).float()
        sv = torch.linalg.svdvals(J)          # descending
        loss = sv[-1]                          # minimise sigma_min

        loss.backward()
        opt.step()

        if step % 20 == 0 or step == n_steps - 1:
            smin = sv[-1].item()
            smax = sv[0].item()
            cond = smax / (smin + 1e-12)
            print(f"{step:>6}  {smin:>12.6f}  {smax:>12.6f}  {cond:>12.2f}")

    print(f"\nFinal sigma_min: {sv[-1].item():.6f}  (0 = singular)")


def correlate_sigma_min(n_pos=8, n_probes=128, seed=0):
    """Pearson correlation between exact sigma_min (SVD) and approx sigma_min (JVP probes).

    Sweeps all sublayers across all layers and n_pos token positions on a
    single random sequence. Prints a per-sublayer table and an overall correlation.
    """
    torch.manual_seed(seed)
    cfg    = hj.Config(d_model=D_MODEL, n_heads=4, n_layers=8, vocab_size=256, mlp_expand=4)
    model  = hj.CustomModel(cfg).eval()
    layers = _layers(model)
    n_layers = len(layers)

    inputs = torch.randint(0, cfg.vocab_size, (max(n_pos, 1),))
    store  = capture_all_hidden(model, inputs)

    n_pos = min(n_pos, inputs.shape[0])
    exact_all  = []
    approx_all = []

    print(f"d={D_MODEL}  seq={inputs.shape[0]}  layers={n_layers}  n_probes={n_probes}\n")
    hdr = f"{'layer':>5} {'sub':>4} {'pos':>3}  {'exact_smin':>12} {'approx_smin':>12}  {'ratio':>8}"
    print(hdr)
    print("-" * len(hdr))

    for layer_idx in range(n_layers):
        for sublayer in ("attn", "ffn"):
            h   = store[(layer_idx, sublayer)][0]   # (seq, d)
            fn  = _sublayer_fn(layers[layer_idx], sublayer, model=model)
            h_B = h.unsqueeze(0)
            print("in correlation bef")
            J_all = _jac_attn(fn, h_B) if sublayer == "attn" else _jac_ffn(fn, h_B)
            print("in correlation")
            for p in range(n_pos):
                J_p        = J_all[0, p].float()
                smin_exact = torch.linalg.svdvals(J_p)[-1].item()

                f_loc = make_f_local(fn, h, p)
                x_p   = h[p].detach().clone()
                Jv_fn = lambda v, _f=f_loc, _x=x_p: jvp(_f, (_x,), (v,))[1]
                result = check_invertibility(Jv_fn, n=D_MODEL, m=n_probes)
                smin_approx = result.sigma_min_estimate

                exact_all.append(smin_exact)
                approx_all.append(smin_approx)
                ratio = smin_approx / (smin_exact + 1e-12)
                print(f"{layer_idx:>5} {sublayer:>4} {p:>3}  {smin_exact:>12.6f} {smin_approx:>12.6f}  {ratio:>8.3f}")

    e = torch.tensor(exact_all, dtype=torch.float64)
    a = torch.tensor(approx_all, dtype=torch.float64)
    corr = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    print(f"\nPearson r(exact, approx) = {corr:.6f}  (n={len(exact_all)} samples)")


if __name__ == "__main__":
    correlate_sigma_min()
