import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModel, AutoTokenizer

_LN_PRE_ATT = ("input_layernorm", "ln_1", "ln1", "layernorm_before")
_LN_PRE_FFN = ("post_attention_layernorm", "ln_2", "ln2", "layernorm_after")
_ATT        = ("self_attn", "attention", "attn")
_FFN        = ("mlp", "feed_forward", "ffn")

SUBLAYERS = {
    "attn": (_LN_PRE_ATT, _ATT),
    "ffn":  (_LN_PRE_FFN, _FFN),
}

ARCH_SUB_IS_FIRST = {
    "GPT2Model":  {"attn": True,  "ffn": False},
    "LlamaModel": {"attn": False, "ffn": False},
}


def load(model_name: str, device: str = "cpu"):
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok

def tokenize(tok, text: str, device: str = "cpu") -> torch.Tensor:
    return tok(text, return_tensors="pt").input_ids.to(device)


def _layers(model):
    inner = next(
        (getattr(model, a) for a in ("model", "transformer", "gpt_neox", "encoder")
         if hasattr(model, a)),
        model,
    )
    return next(
        getattr(inner, a) for a in ("layers", "h", "layer", "blocks", "block")
        if hasattr(inner, a)
    )


def _sub(layer, *candidates):
    for name in candidates:
        if hasattr(layer, name):
            return name, getattr(layer, name)
    raise ValueError(
        f"None of {candidates} found in {type(layer).__name__}. "
        f"Available: {list(layer._modules)}"
    )


def _get_sublayer_x(model, inputs, layer_idx, sublayer):
    """
    Single no_grad forward pass. Returns (B, seq, d) residual input for the sublayer:
      "attn" → block input x
      "ffn"  → attn-residual output  x + attn(ln_1(x))
    """
    layer    = _layers(model)[layer_idx]
    captured = [None]

    if sublayer == "attn":
        handle = layer.register_forward_pre_hook(
            lambda _mod, args: captured.__setitem__(0, args[0].detach().clone())
        )
    else:
        _, attn_mod = _sub(layer, *_ATT)
        def _hook(_mod, inp, out):
            a = out[0] if isinstance(out, tuple) else out
            captured[0] = (inp[0] + a).detach().clone()
        handle = attn_mod.register_forward_hook(_hook)

    with torch.no_grad():
        if inputs.is_floating_point():
            model(inputs_embeds=inputs)
        else:
            model(inputs)
    handle.remove()
    return captured[0]  # (B, seq, d)


class _InSituJac(TorchDispatchMode):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.inject_id  = None
        self.capture_id = None
        self.x_leaf     = None
        self.output     = None
        self.verbose    = verbose

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        if func != torch.ops.aten.add.Tensor:
            return func(*args, **(kwargs or {}))

        if self.inject_id is not None or self.capture_id is not None:
            id0 = id(args[0]) if isinstance(args[0], torch.Tensor) else None
            id1 = id(args[1]) if isinstance(args[1], torch.Tensor) else None

            if self.inject_id is not None and self.inject_id in (id0, id1):
                self.inject_id = None
                result = func(*args, **(kwargs or {}))
                self.x_leaf = result.detach().clone().requires_grad_(True)
                if self.verbose:
                    print(f"  [inject]   x_leaf at add     {tuple(result.shape)}")
                return self.x_leaf

            if self.capture_id is not None and self.capture_id in (id0, id1):
                self.capture_id = None
                self.output = func(*args, **(kwargs or {}))
                if self.verbose:
                    print(f"  [capture]  residual add      {tuple(self.output.shape)}")
                return self.output

        return func(*args, **(kwargs or {}))


def _capture(model, inputs, layer_idx, sublayer, verbose: bool = False):
    if sublayer not in SUBLAYERS:
        raise ValueError(f"sublayer must be one of {list(SUBLAYERS)}, got {sublayer!r}")

    layer = _layers(model)[layer_idx]
    ln_candidates, sub_candidates = SUBLAYERS[sublayer]
    ln_name, ln   = _sub(layer, *ln_candidates)
    sub_name, sub = _sub(layer, *sub_candidates)

    if verbose:
        print(f"sublayer : {sublayer}  (layer {layer_idx})")
        print(f"  ln  = .{ln_name}  ({type(ln).__name__})")
        print(f"  sub = .{sub_name}  ({type(sub).__name__})")

    intercept = _InSituJac(verbose=verbose)
    handles   = []

    if sublayer == "attn":
        def inject(mod, args):
            intercept.x_leaf = args[0].detach().clone().requires_grad_(True)
            return (intercept.x_leaf,)
        handles.append(layer.register_forward_pre_hook(inject))
    else:
        _, attn = _sub(layer, *_ATT)
        def set_inject(mod, inp, out):
            intercept.inject_id = id(out[0] if isinstance(out, tuple) else out)
        handles.append(attn.register_forward_hook(set_inject))

    def set_capture(mod, inp, out):
        intercept.capture_id = id(out[0] if isinstance(out, tuple) else out)
    handles.append(sub.register_forward_hook(set_capture))

    with intercept:
        if inputs.is_floating_point():
            model(inputs_embeds=inputs)
        else:
            model(inputs)

    for h in handles:
        h.remove()

    x, out = intercept.x_leaf, intercept.output
    if verbose:
        print(f"  x_leaf : {tuple(x.shape)}   output : {tuple(out.shape)}")
    return x, out


def _sublayer_fn(layer, sublayer):
    """Returns f: (seq, d) → (seq, d) — full residual x + g(LN(x)).
    Jacobian of f w.r.t. x is I + dg/dh, with chain rule through LN included."""
    _, ln_mod  = _sub(layer, *SUBLAYERS[sublayer][0])
    _, sub_mod = _sub(layer, *SUBLAYERS[sublayer][1])

    def f(x):   # x: (seq, d), variable length — works for any prefix
        out = sub_mod(ln_mod(x.unsqueeze(0)))
        out = out[0] if isinstance(out, tuple) else out
        return x + out.squeeze(0)   # (seq, d)

    return f


def _diag_jac(fn, x_b: torch.Tensor, chunk_d: int) -> torch.Tensor:
    """
    Per-position (d, d) Jacobian blocks via truncated-sequence causal forward
    and chunked is_grads_batched backward.

    For position p the forward uses only x_b[:p+1] (causal masking guarantees
    out[p] is identical to the full-sequence result).  Each backward call
    processes chunk_d output dimensions simultaneously.

    Peak extra memory: chunk_d × (sublayer activation size for seq tokens).
    Reduce chunk_d if you hit OOM; increase it for fewer Python round-trips.
    """
    seq, d = x_b.shape
    jac = torch.zeros(seq, d, d, device=x_b.device, dtype=x_b.dtype)

    for p in range(seq):
        x_t = x_b[:p + 1].detach().clone().requires_grad_(True)  # (p+1, d)
        out = fn(x_t)                                              # (p+1, d)

        for i0 in range(0, d, chunk_d):
            i1 = min(i0 + chunk_d, d)
            eye_chunk = torch.zeros(i1 - i0, d, device=x_b.device, dtype=x_b.dtype)
            eye_chunk[torch.arange(i1 - i0), torch.arange(i0, i1)] = 1
            g = torch.autograd.grad(
                out[p], x_t,
                grad_outputs=eye_chunk,         # (chunk, d) — batched seeds
                retain_graph=(i1 < d),
                is_grads_batched=True,
            )[0]                                # (chunk, p+1, d)
            jac[p, i0:i1] = g[:, p]            # (chunk, d)

    return jac   # (seq, d, d)


class _CaptureAllMode(TorchDispatchMode):
    """
    Intercepts every residual add (aten.add.Tensor) during a single forward pass
    and stores the result keyed by (layer_idx, sublayer).

    Forward hooks on each sublayer module set `pending[key] = id(sublayer_out)`
    just before the add fires; the dispatch intercept matches on that id and
    saves the add result (= x + g(LN(x)), the true residual output).
    """
    def __init__(self):
        super().__init__()
        self.pending: dict[int, tuple] = {}   # tensor_id → store key
        self.store:   dict[tuple, torch.Tensor] = {}

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.add.Tensor:
            a0 = args[0] if len(args) > 0 else None
            a1 = args[1] if len(args) > 1 else None
            id0 = id(a0) if isinstance(a0, torch.Tensor) else None
            id1 = id(a1) if isinstance(a1, torch.Tensor) else None
            for tid in (id0, id1):
                if tid is not None and tid in self.pending:
                    key = self.pending.pop(tid)
                    self.store[key] = result.detach().cpu().clone()
                    break
        return result


def capture_all_hidden(
    model, inputs
) -> dict[tuple, torch.Tensor]:
    """
    Single no_grad forward. Returns a dict keyed by (layer_idx, sublayer) →
    hidden_out (B, seq, d), plus two special keys:
      ("embed",  "out") → embed_out    (B, seq, d)
      ("final",  "out") → final_hidden (B, seq, d)

    Uses TorchDispatchMode to intercept the residual adds directly, so the
    captured tensors are the true residual stream values (not post-LN inputs).
    All tensors are on CPU.
    """
    layers  = _layers(model)
    mode    = _CaptureAllMode()
    handles = []

    # embed_out: residual entering block 0 (pre-LN, so a pre_hook is correct here)
    def _hook_embed(_mod, args):
        mode.store[("embed", "out")] = args[0].detach().cpu().clone()
    handles.append(layers[0].register_forward_pre_hook(_hook_embed))

    for i, layer in enumerate(layers):
        # Mark the sublayer output id so the dispatch mode knows which add to capture
        _, attn_mod = _sub(layer, *_ATT)
        def _set_attn(_mod, _inp, out, _i=i):
            mode.pending[id(out[0] if isinstance(out, tuple) else out)] = (_i, "attn")
        handles.append(attn_mod.register_forward_hook(_set_attn))

        _, ffn_mod = _sub(layer, *_FFN)
        def _set_ffn(_mod, _inp, out, _i=i):
            mode.pending[id(out[0] if isinstance(out, tuple) else out)] = (_i, "ffn")
        handles.append(ffn_mod.register_forward_hook(_set_ffn))

    # final_hidden: output of the last block (the block's output IS the residual)
    def _hook_final(_mod, _inp, out):
        a = out[0] if isinstance(out, tuple) else out
        mode.store[("final", "out")] = a.detach().cpu().clone()
    handles.append(layers[-1].register_forward_hook(_hook_final))

    with mode, torch.no_grad():
        if inputs.is_floating_point():
            model(inputs_embeds=inputs)
        else:
            model(inputs)

    for h in handles:
        h.remove()

    return mode.store


def capture_sublayer(
    model, inputs, layer_idx: int, sublayer: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (hidden_in, hidden_out):
      hidden_in  : (B, seq, d)  residual input  h
      hidden_out : (B, seq, d)  residual output h + g(h)
    Used by the Jacobian path — do not call for activation-only extraction.
    """
    x, output = _capture(model, inputs, layer_idx, sublayer)
    return x.detach(), output.detach()


def capture_endpoints(
    model, inputs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (embed_out, final_hidden):
      embed_out    : (B, seq, d)  output of the embedding layer (input to block 0)
      final_hidden : (B, seq, d)  final hidden state before the unembedding matrix
    """
    captured_embed = [None]
    captured_final = [None]

    layers = _layers(model)

    handle_embed = layers[0].register_forward_pre_hook(
        lambda _mod, args: captured_embed.__setitem__(0, args[0].detach().clone())
    )
    handle_final = layers[-1].register_forward_hook(
        lambda _mod, _inp, out: captured_final.__setitem__(
            0, (out[0] if isinstance(out, tuple) else out).detach().clone()
        )
    )

    with torch.no_grad():
        if inputs.is_floating_point():
            model(inputs_embeds=inputs)
        else:
            model(inputs)

    handle_embed.remove()
    handle_final.remove()
    assert captured_embed[0] is not None and captured_final[0] is not None
    return captured_embed[0], captured_final[0]


def _jac_single(fn, x_b, jac_chunk):
    """Chunked Jacobian for one batch item. x_b: (seq, d) → (seq, d, d)."""
    seq, d = x_b.shape
    eye = torch.eye(d, device=x_b.device, dtype=x_b.dtype)
    jac = torch.zeros(seq, d, d, device=x_b.device, dtype=x_b.dtype)

    for p in range(seq):
        x_p = x_b[:p+1].detach().clone().requires_grad_(True)
        out = fn(x_p)
        for i0 in range(0, d, jac_chunk):
            i1 = min(i0 + jac_chunk, d)
            g = torch.autograd.grad(
                out[p], x_p,
                grad_outputs=eye[i0:i1],
                is_grads_batched=True,
                retain_graph=(i1 < d),
            )[0]                        # (chunk, p+1, d)
            jac[p, i0:i1] = g[:, p]    # (chunk, d)

    return jac


def _causal_block_jac(
    model, inputs, layer_idx: int, sublayer: str, jac_chunk: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (hidden_out, jac):
      hidden_out : (B, seq, d)    residual output h + g(LN(h))
      jac        : (B, seq, d, d) per-position Jacobian d(h + g(LN(h)))/dh

    Uses the sublayer function directly (no full-model graph), with a causal
    prefix truncation per position and chunked backward over output dims.
    jac_chunk: output dims per backward pass — lower = less VRAM, higher = faster.
    """
    store = capture_all_hidden(model, inputs)
    hidden_out = store[(layer_idx, sublayer)]   # (B, seq, d), on CPU

    layer = _layers(model)[layer_idx]
    fn    = _sublayer_fn(layer, sublayer)

    B = hidden_out.shape[0]
    jacs = torch.stack([_jac_single(fn, hidden_out[b], jac_chunk) for b in range(B)])
    return hidden_out, jacs


def print_model(model):
    layers = _layers(model)
    print(f"{type(model).__name__}  —  {len(layers)} layers")
    print(f"\nLayer[0] children:")
    for name, mod in layers[0].named_children():
        print(f"  .{name:<30} {type(mod).__name__}")


def position_jacobians(
    model,
    inputs: torch.Tensor,
    layer_idx: int,
    sublayer: str,
    jac_chunk: int = 64,
) -> torch.Tensor:
    """
    Per-position (d, d) block Jacobian.

    inputs: int token ids (B, seq) or float latent representations (B, seq, d).
    Returns (B, seq, d, d).
    """
    _, jac = _causal_block_jac(model, inputs, layer_idx, sublayer, jac_chunk)
    return jac


def position_jacobians_seq(
    model,
    inputs: torch.Tensor,
    layer_idx: int,
    sublayer: str,
    jac_chunk: int = 64,
    compute_det: bool = True,
    compute_sigma_ratio: bool = True,
) -> dict:
    """
    Per-position Jacobian stats for all batch items.

        inputs: int token ids (B, seq) or float latent representations (B, seq, d).
        Returns dict with requested stats keys:
            det         (B, seq) — determinant of each J
            sigma_ratio (B, seq) — max(σ) / min(σ), i.e. the condition number
    """
    if sublayer not in SUBLAYERS:
        raise ValueError(f"sublayer must be one of {list(SUBLAYERS)}, got {sublayer!r}")

    layer = _layers(model)[layer_idx]
    x = _get_sublayer_x(model, inputs, layer_idx, sublayer)  # (B, seq, d)
    fn = _sublayer_fn(layer, sublayer)

    d = x.shape[-1]
    eye = torch.eye(d, device=x.device, dtype=x.dtype)

    dets = [] if compute_det else None
    sigmas = [] if compute_sigma_ratio else None
    for x_b in x:
        jac_b = _diag_jac(fn, x_b, jac_chunk) + eye
        if compute_det:
            dets.append(torch.linalg.det(jac_b))
        if compute_sigma_ratio:
            sv = torch.linalg.svdvals(jac_b)
            sigmas.append(sv[..., 0] / sv[..., -1])

    stats = {}
    if compute_det:
        stats["det"] = torch.stack(dets)
    if compute_sigma_ratio:
        stats["sigma_ratio"] = torch.stack(sigmas)
    return stats


def jacobian_stats(jac: torch.Tensor) -> dict:
    """
    jac: (B, seq, d, d) batched stack of square Jacobians.
    Returns dict with:
      det         (B, seq) — determinant of each J
      sigma_ratio (B, seq) — max(σ) / min(σ), i.e. the condition number
    """
    det = torch.linalg.det(jac)
    sv  = torch.linalg.svdvals(jac)         # (B, seq, d), descending
    return {"det": det, "sigma_ratio": sv[..., 0] / sv[..., -1]}
