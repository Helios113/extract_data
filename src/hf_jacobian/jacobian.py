import torch
from transformers import AutoModel, AutoTokenizer

# DeepGEMM JIT fails on this machine (no compatible nvcc: 12.8 too old, 13.0
# has broken 128-bit inline asm). Force Triton fallback before any import
# touches the flag.
try:
    import transformers.integrations.finegrained_fp8 as _fp8
    _fp8._deepgemm_available = False
except Exception:
    pass


# QuantFactory GGUF repos carry no tokenizer; map to the canonical base model.
_GGUF_TOKENIZER = {
    "QuantFactory/gpt2-GGUF":        "openai-community/gpt2",
    "QuantFactory/pythia-160m-GGUF": "EleutherAI/pythia-160m",
    "QuantFactory/Qwen3-0.6B-GGUF":  "Qwen/Qwen3-0.6B",
    "QuantFactory/Qwen3-1.7B-GGUF":  "Qwen/Qwen3-1.7B",
}

def load(model_name: str, device: str = "cpu"):
    kwargs = {"device_map": device}
    if "::" in model_name:
        repo, gguf_file = model_name.split("::", 1)
        kwargs["gguf_file"] = gguf_file
        model = AutoModel.from_pretrained(repo, **kwargs)
        tok_repo = _GGUF_TOKENIZER.get(repo, repo)
        tok = AutoTokenizer.from_pretrained(tok_repo)
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)
        tok = AutoTokenizer.from_pretrained(model_name)
    return model.eval(), tok

def tokenize(tok, text: str, device: str = "cpu") -> torch.Tensor:
    return tok(text, return_tensors="pt").input_ids.to(device)


def _layers(model):
    from .custom_model import CustomModel
    if isinstance(model, CustomModel):
        return model.layers
    inner = next(
        (getattr(model, a) for a in ("model", "transformer", "gpt_neox", "encoder")
         if hasattr(model, a)),
        model,
    )
    return next(
        getattr(inner, a) for a in ("layers", "h", "layer", "blocks", "block")
        if hasattr(inner, a)
    )



def _sublayer_fn_gpt2(layer, sublayer):
    if sublayer == "attn":
        def f(x):                              # x: (B, seq, d)
            out = layer.attn(layer.ln_1(x))[0]
            return x + out
    else:
        def f(x):
            out = layer.mlp(layer.ln_2(x))
            return x + out
    return f


def _sublayer_fn_llama(layer, model, sublayer):
    if sublayer == "attn":
        def f(x):                              # x: (B, seq, d)
            seq = x.shape[1]
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
            rope = model.rotary_emb(x, position_ids=position_ids)
            out = layer.self_attn(
                layer.input_layernorm(x),
                position_embeddings=rope,
            )[0]
            return x + out
    else:
        def f(x):
            out = layer.mlp(layer.post_attention_layernorm(x))
            return x + out
    return f


def _sublayer_fn_qwen3(layer, model, sublayer):
    # Same structure as Llama. Qwen3Attention.forward requires attention_mask
    # as a positional argument (not keyword-only), so pass it explicitly.
    if sublayer == "attn":
        def f(x):                              # x: (B, seq, d)
            seq = x.shape[1]
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
            rope = model.rotary_emb(x, position_ids=position_ids)
            out = layer.self_attn(
                layer.input_layernorm(x),
                position_embeddings=rope,
                attention_mask=None,
            )[0]
            return x + out
    else:
        def f(x):
            out = layer.mlp(layer.post_attention_layernorm(x))
            return x + out
    return f


def _sublayer_fn_pythia(layer, model, sublayer):
    # GPT-NeoX uses a parallel residual: x + attn(LN1(x)) + mlp(LN2(x)).
    # Attn and mlp share the same input — there is no separate attn or ffn
    # residual step. Only sublayer="block" is meaningful.
    if sublayer != "block":
        raise ValueError(
            f"GPTNeoXModel uses a parallel residual — only sublayer='block' is "
            f"supported, got {sublayer!r}"
        )
    def f(x):                                  # x: (B, seq, d)
        seq = x.shape[1]
        position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
        rope = model.rotary_emb(x, position_ids=position_ids)
        attn_out = layer.attention(
            layer.input_layernorm(x),
            attention_mask=None,
            position_embeddings=rope,
        )[0]
        mlp_out = layer.mlp(layer.post_attention_layernorm(x))
        return x + attn_out + mlp_out
    return f


def _sublayer_fn_custom(layer, _model, sublayer):
    if sublayer == "attn":
        def f(x):   # x: (seq, d)
            return layer.attn(x.unsqueeze(0)).squeeze(0)
    else:
        def f(x):
            return layer.ffn(x.unsqueeze(0)).squeeze(0)
    return f


_SUBLAYER_FN_REGISTRY = {
    "GPT2Model":    lambda layer, _model, sublayer: _sublayer_fn_gpt2(layer, sublayer),
    "LlamaModel":   _sublayer_fn_llama,
    "Qwen3Model":   _sublayer_fn_qwen3,
    "GPTNeoXModel": _sublayer_fn_pythia,
    "CustomModel":  _sublayer_fn_custom,
}


# Per-architecture getters for the attn and ffn submodules used in capture_all_hidden.
# Each value is (attn_getter, ffn_getter) where getter: layer -> nn.Module.
# For parallel-residual architectures (Pythia), both getters return the same
# module (attention) and the key stored is "block" not "attn"/"ffn".
_CAPTURE_MODS = {
    "GPT2Model":    (lambda l: l.attn,              lambda l: l.mlp),
    "LlamaModel":   (lambda l: l.self_attn,         lambda l: l.mlp),
    "Qwen3Model":   (lambda l: l.self_attn,         lambda l: l.mlp),
    "GPTNeoXModel": (lambda l: l.attention,         lambda l: l.mlp),
    "CustomModel":  (lambda l: l.attn.self_attn,    lambda l: l.ffn.mlp),
}


def _sublayer_fn(layer, sublayer, model=None):
    """Dispatch to the architecture-specific sublayer function.
    Returns f: (seq, d) → (seq, d) — full residual x + g(LN(x))."""
    arch = type(model).__name__ if model is not None else None
    if arch not in _SUBLAYER_FN_REGISTRY:
        raise ValueError(
            f"Unsupported architecture {arch!r}. "
            f"Register it in _SUBLAYER_FN_REGISTRY. "
            f"Supported: {list(_SUBLAYER_FN_REGISTRY)}"
        )
    return _SUBLAYER_FN_REGISTRY[arch](layer, model, sublayer)




def capture_all_hidden(
    model, inputs
) -> dict[tuple, torch.Tensor]:
    """
    Single no_grad forward. Returns a dict keyed by (layer_idx, sublayer) →
    hidden_out (B, seq, d), plus two special keys:
      ("embed",  "out") → embed_out    (B, seq, d)
      ("final",  "out") → final_hidden (B, seq, d)

    Capture strategy (no TorchDispatchMode):
      - embed_out          : pre-hook on layer 0  → args[0] is the raw residual stream
      - (i, "attn") out    : pre-hook on ffn mod  → its input is x + attn(LN(x))
      - (i, "ffn")  out    : post-hook on layer i → output is x + ffn(LN(x))
      - (i, "block") out   : post-hook on layer i → parallel-residual block output
      - final_hidden       : post-hook on last layer (same tensor as last ffn out)

    The ffn pre-hook input is the post-attn residual stream (safe: it is x + attn(LN(x)),
    not a post-LN value — LN is internal to the attn submodule).
    All tensors are returned on CPU.
    """
    layers  = _layers(model)
    store:  dict[tuple, torch.Tensor] = {}
    handles = []
    arch    = type(model).__name__

    if arch not in _CAPTURE_MODS:
        raise ValueError(
            f"Unsupported architecture {arch!r}. "
            f"Register it in _CAPTURE_MODS. "
            f"Supported: {list(_CAPTURE_MODS)}"
        )
    attn_getter, ffn_getter = _CAPTURE_MODS[arch]

    # embed_out: residual entering block 0
    def _hook_embed(_mod, args):
        store[("embed", "out")] = args[0].detach().cpu().clone()
    handles.append(layers[0].register_forward_pre_hook(_hook_embed))

    for i, layer in enumerate(layers):
        if arch == "GPTNeoXModel":
            # Parallel residual — single block add; layer post-hook captures it
            def _hook_block(_mod, _inp, out, _i=i):
                a = out[0] if isinstance(out, tuple) else out
                store[(_i, "block")] = a.detach().cpu().clone()
            handles.append(layer.register_forward_hook(_hook_block))
        else:
            # attn out = x + attn(LN(x)) = input to the ffn submodule
            ffn_mod = ffn_getter(layer)
            def _hook_attn_out(_mod, args, _i=i):
                x = args[0] if isinstance(args, tuple) else args
                store[(_i, "attn")] = x.detach().cpu().clone()
            handles.append(ffn_mod.register_forward_pre_hook(_hook_attn_out))

            # ffn out = x + ffn(LN(x)) = layer output
            def _hook_ffn_out(_mod, _inp, out, _i=i):
                a = out[0] if isinstance(out, tuple) else out
                store[(_i, "ffn")] = a.detach().cpu().clone()
            handles.append(layer.register_forward_hook(_hook_ffn_out))

    # final_hidden: same as last layer's post-hook, already registered above
    def _hook_final(_mod, _inp, out):
        a = out[0] if isinstance(out, tuple) else out
        store[("final", "out")] = a.detach().cpu().clone()
    handles.append(layers[-1].register_forward_hook(_hook_final))

    with torch.no_grad():
        if inputs.is_floating_point():
            model(inputs_embeds=inputs)
        else:
            model(inputs)

    for h in handles:
        h.remove()

    return store


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


def _jac_attn(fn, x_B):
    """Per-token local Jacobian for attention sublayers via vmap(jacrev).

    Each token p gets its own R^d→R^d function with context frozen, so we
    pay O(seq * d * T_attn(p)) instead of O(seq^2 * d * T_attn(seq)) from
    the full-sequence approach. Wins when seq is large.

    x_B: (B, seq, d).  Returns (B, seq, d, d).
    """
    from torch.func import jacrev, vmap

    B, seq, d = x_B.shape
    jacs = []
    for b in range(B):
        h = x_B[b].detach()                          # (seq, d)
        eye = torch.eye(seq, device=h.device).unsqueeze(-1)   # (seq, seq, 1)
        ctxs = h.unsqueeze(0).expand(seq, -1, -1).clone()
        ctxs[torch.arange(seq), torch.arange(seq)] = 0.0

        def f_single(x_p, mask, ctx, _fn=fn):
            full = ctx + mask * x_p.unsqueeze(0)      # (seq, d)
            return (_fn(full) * mask).sum(0)           # (d,)

        jacs.append(vmap(jacrev(f_single))(h, eye, ctxs))   # (seq, d, d)
    return torch.stack(jacs)                          # (B, seq, d, d)


def _jac_ffn(fn, x_B):
    """Per-token local Jacobian for FFN sublayers via jacrev on the full sequence.

    FFN is pointwise so jacrev(fn)(x) costs the same as seq independent jacrevs
    but in one kernel. Extract diagonal blocks J[p,:,p,:].

    x_B: (B, seq, d).  Returns (B, seq, d, d).
    """
    from torch.func import jacrev

    B, seq, d = x_B.shape
    idx = torch.arange(seq, device=x_B.device)
    jacs = []
    for b in range(B):
        J = jacrev(fn)(x_B[b].detach())       # (seq, d, seq, d)
        jacs.append(J[idx, :, idx, :])         # (seq, d, d)
    return torch.stack(jacs)                   # (B, seq, d, d)


def _jac_block(fn, x_B):
    """Per-token local Jacobian for parallel-residual block sublayers.

    Same vmap(jacrev) strategy as _jac_attn: freeze context, differentiate
    d(out_p)/d(in_p) for each position p in one vectorized call.

    x_B: (B, seq, d).  Returns (B, seq, d, d).
    """
    from torch.func import jacrev, vmap

    B, seq, d = x_B.shape
    x_B = x_B.float()
    jacs = []
    for b in range(B):
        h = x_B[b].detach()                               # (seq, d), float32
        eye  = torch.eye(seq, device=h.device).unsqueeze(-1)  # (seq, seq, 1)
        ctxs = h.unsqueeze(0).expand(seq, -1, -1).clone()
        ctxs[torch.arange(seq), torch.arange(seq)] = 0.0

        def f_single(x_p, mask, ctx, _fn=fn):
            full = ctx + mask * x_p.unsqueeze(0)           # (seq, d)
            return (_fn(full) * mask).sum(0)               # (d,)

        jacs.append(vmap(jacrev(f_single))(h, eye, ctxs))  # (seq, d, d)
    return torch.stack(jacs)                               # (B, seq, d, d)


def _jac_batched(fn, x_B, sublayer=None, jac_chunk=None):
    """Dispatch to the right Jacobian strategy based on sublayer type.

    attn → per-token vmap(jacrev): avoids paying O(seq^2) attention cost
           in each of seq*d backward passes.
    ffn  → full-sequence jacrev: FFN is pointwise so one graph is cheaper
           than seq separate graph builds.
    """
    if sublayer == "attn":
        return _jac_attn(fn, x_B)
    return _jac_ffn(fn, x_B)


def _causal_block_jac(
    model, inputs, layer_idx: int, sublayer: str, jac_chunk: int = 128,
    store: dict = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (hidden_out, jac):
      hidden_out : (B, seq, d)    residual-stream input to this sublayer
      jac        : (B, seq, d, d) per-position local Jacobian

    Pass a pre-computed store from capture_all_hidden to avoid re-running
    the full model forward when computing Jacobians for multiple sublayers.
    """
    if store is None:
        store = capture_all_hidden(model, inputs)
    hidden_out = store[(layer_idx, sublayer)]   # (B, seq, d), on CPU
    layer = _layers(model)[layer_idx]
    device = next(layer.parameters()).device
    dtype  = next(layer.parameters()).dtype
    hidden_out = hidden_out.to(device=device, dtype=dtype)
    fn = _sublayer_fn(layer, sublayer, model=model)
    jacs = _jac_batched(fn, hidden_out, sublayer=sublayer)
    return hidden_out, jacs


def reinit_weights(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def print_model(model):
    layers = _layers(model)
    print(f"{type(model).__name__}  —  {len(layers)} layers")
    print(f"\nLayer[0] children:")
    for name, mod in layers[0].named_children():
        print(f"  .{name:<30} {type(mod).__name__}")


def jacobian_stats(jac: torch.Tensor) -> dict:
    """
    jac: (B, seq, d, d) batched stack of square Jacobians.
    Returns dict with:
      det              (B, seq)    — determinant of each J
      singular_values  (B, seq, d) — full singular spectrum, descending
      sigma_max        (B, seq)    — largest singular value (= ‖J‖_2)
      sigma_min        (B, seq)    — smallest singular value (= distance to singular)
    sigma_max/sigma_min are kept alongside the full spectrum for convenience;
    they're just the first/last entries of singular_values. κ = sigma_max /
    sigma_min and κ⁻¹ = sigma_min / sigma_max are derivable.
    Cast jac to fp32 before SVD/det: torch.linalg's MAGMA-batched paths don't
    support bf16/fp16 on CUDA, and fp32 SVD is more accurate regardless. The
    matrix being measured is unchanged — only the spectrum measurement is
    promoted to fp32.
    """
    jac = jac.to(torch.float32)
    sign, logabsdet = torch.linalg.slogdet(jac)
    log_det = sign.float() * logabsdet          # signed log|det|
    sv  = torch.linalg.svdvals(jac)             # (B, seq, d), descending
    return {
        "log_det": log_det,
        "singular_values": sv,
        "sigma_max": sv[..., 0],
        "sigma_min": sv[..., -1],
    }
