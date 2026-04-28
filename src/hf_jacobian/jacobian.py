import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModel, AutoTokenizer

# Candidate submodule names across common HF architectures
_LN_PRE_ATT = ("input_layernorm", "ln_1", "ln1", "layernorm_before")
_LN_PRE_FFN = ("post_attention_layernorm", "ln_2", "ln2", "layernorm_after")
_ATT        = ("self_attn", "attention", "attn")
_FFN        = ("mlp", "feed_forward", "ffn")

SUBLAYERS = {
    "attn": (_LN_PRE_ATT, _ATT),
    "ffn":  (_LN_PRE_FFN, _FFN),
}

# Per-architecture config: is the sublayer output the first (a) or second (b) operand
# in the residual add?  True = first (e.g. GPT-2 attn: attn_out + residual).
# Used only for the inject step (replacing the attn-add result for the FFN case).
# For capture we just check both, which is safe.
ARCH_SUB_IS_FIRST = {
    "GPT2Model":  {"attn": True,  "ffn": False},
    "LlamaModel": {"attn": False, "ffn": False},
    # add more as needed; falls back to checking both if unknown
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


class _InSituJac(TorchDispatchMode):
    """
    Two dispatch jobs in one forward pass:

    inject_id  (FFN only):
        The attn-add result is the residual input for the FFN.  There is no
        module boundary there, so we intercept it in dispatch and return a fresh
        detached leaf (x1_leaf) instead.  Because the model's Python code then
        does `residual = x1_leaf`, autograd correctly records the FFN add as
        add(x1_leaf, ffn_out) — gradient tracking is intact.

    capture_id (both):
        After the target sublayer fires (set_capture hook sets capture_id), the
        next matching aten.add is the residual add we care about.  We run it
        normally (args untouched) and save the result so we can backprop later.
        Autograd already sees x_leaf / x1_leaf in the args at this point because
        it was placed there by the pre-hook (attn) or inject step (ffn).
    """

    def __init__(self):
        super().__init__()
        self.inject_id  = None  # id of the attn sublayer output (FFN case)
        self.capture_id = None  # id of the target sublayer output
        self.x_leaf     = None
        self.output     = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func != torch.ops.aten.add.Tensor:
            return func(*args, **(kwargs or {}))

        tensors = [t for t in args[:2] if isinstance(t, torch.Tensor)]
        ids     = {id(t) for t in tensors}

        # FFN inject: replace attn-add result with x1_leaf
        if self.inject_id is not None and self.inject_id in ids:
            self.inject_id = None
            result = func(*args, **(kwargs or {}))
            self.x_leaf = result.detach().clone().requires_grad_(True)
            print(f"  [inject]   x1_leaf at attn add  {tuple(result.shape)}")
            return self.x_leaf

        # Capture: run add normally (x_leaf is already in args via pre-hook / inject)
        if self.capture_id is not None and self.capture_id in ids:
            self.capture_id = None
            self.output = func(*args, **(kwargs or {}))
            print(f"  [capture]  residual add          {tuple(self.output.shape)}")
            return self.output

        return func(*args, **(kwargs or {}))



def _block_jac_from_graph(output, x_leaf):
    """
    Per-position (d, d) Jacobian of output_p w.r.t. input_p.

    Same seq*d backward passes as the diagonal, but each pass keeps the full
    position-p slice of the gradient instead of just the diagonal element.

    output : (1, seq, d)
    x_leaf : (1, seq, d)
    returns: (seq, d, d)  — jac[p, i, j] = d(output[0,p,i]) / d(x_leaf[0,p,j])
    """
    seq, d = output.shape[1], output.shape[2]
    n      = seq * d
    jac    = torch.zeros(seq, d, d, device=x_leaf.device, dtype=x_leaf.dtype)
    out_flat = output.reshape(-1)
    for flat_idx in range(n):
        p, i = divmod(flat_idx, d)
        (g,) = torch.autograd.grad(out_flat[flat_idx], x_leaf, retain_graph=(flat_idx < n - 1))
        jac[p, i] = g[0, p]   # d(output[0,p,i]) / d(x_leaf[0,p,:])
    return jac


def print_model(model):
    layers = _layers(model)
    print(f"{type(model).__name__}  —  {len(layers)} layers")
    print(f"\nLayer[0] children:")
    for name, mod in layers[0].named_children():
        print(f"  .{name:<30} {type(mod).__name__}")


def _capture(model, inputs, layer_idx, sublayer):
    """
    Single forward pass: returns (x_leaf, output) where output = x_leaf + sub(ln(x_leaf)).
    x_leaf is the detached leaf at the residual input; output retains the graph.

    inputs: int token ids (1, seq) or float latent representations (1, seq, d).
    Float inputs are passed as inputs_embeds=, bypassing the model's embedding layer.

    Injection mechanism differs by sublayer:
      "attn": x is the layer input → pre-hook injects x_leaf before any op runs.
      "ffn":  x1 is produced mid-layer → dispatch replaces the attn-add result
              with x1_leaf so the model sets `residual = x1_leaf` and autograd
              correctly tracks the FFN add.
    """
    if sublayer not in SUBLAYERS:
        raise ValueError(f"sublayer must be one of {list(SUBLAYERS)}, got {sublayer!r}")

    layer = _layers(model)[layer_idx]
    ln_candidates, sub_candidates = SUBLAYERS[sublayer]
    ln_name, ln   = _sub(layer, *ln_candidates)
    sub_name, sub = _sub(layer, *sub_candidates)

    print(f"sublayer : {sublayer}  (layer {layer_idx})")
    print(f"  ln  = .{ln_name}  ({type(ln).__name__})")
    print(f"  sub = .{sub_name}  ({type(sub).__name__})")

    intercept = _InSituJac()
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
    print(f"  x_leaf : {tuple(x.shape)}   output : {tuple(out.shape)}")
    return x, out



def position_jacobians(
    model,
    inputs: torch.Tensor,
    layer_idx: int,
    sublayer: str,
) -> torch.Tensor:
    """
    Per-position (d, d) Jacobian: how each hidden dim at position p affects
    every hidden dim at position p in the output.

    inputs: int token ids (1, seq) or float latent representations (1, seq, d).
    Returns (seq, d, d).
    """
    x, out = _capture(model, inputs, layer_idx, sublayer)
    return _block_jac_from_graph(out, x)


def jacobian_stats(jac: torch.Tensor) -> dict:
    """
    jac: (seq, d, d) stack of square Jacobians, one per position.
    Returns dict with:
      det        (seq,) — determinant of each J
      sigma_ratio (seq,) — max(σ) / min(σ), i.e. the condition number
    """
    det = torch.linalg.det(jac)
    sv = torch.linalg.svdvals(jac)          # (seq, d), descending
    sigma_ratio = sv[:, 0] / sv[:, -1]
    return {"det": det, "sigma_ratio": sigma_ratio}


