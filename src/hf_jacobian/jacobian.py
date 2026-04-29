import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModel, AutoTokenizer


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


def _sublayer_fn_gpt2(layer, sublayer):
    if sublayer == "attn":
        def f(x):                              # x: (seq, d)
            out = layer.attn(layer.ln_1(x.unsqueeze(0)))[0]  # tuple → [0]
            return x + out.squeeze(0)
    else:
        def f(x):
            out = layer.mlp(layer.ln_2(x.unsqueeze(0)))
            return x + out.squeeze(0)
    return f


def _sublayer_fn_llama(layer, model, sublayer):
    if sublayer == "attn":
        def f(x):                              # x: (seq, d)
            seq = x.shape[0]
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
            rope = model.rotary_emb(x.unsqueeze(0), position_ids=position_ids)
            out = layer.self_attn(
                layer.input_layernorm(x.unsqueeze(0)),
                position_embeddings=rope,
            )[0]
            return x + out.squeeze(0)
    else:
        def f(x):
            out = layer.mlp(layer.post_attention_layernorm(x.unsqueeze(0)))
            return x + out.squeeze(0)
    return f


def _sublayer_fn_qwen3(layer, model, sublayer):
    # Same structure as Llama. Qwen3Attention.forward requires attention_mask
    # as a positional argument (not keyword-only), so pass it explicitly.
    if sublayer == "attn":
        def f(x):                              # x: (seq, d)
            seq = x.shape[0]
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
            rope = model.rotary_emb(x.unsqueeze(0), position_ids=position_ids)
            out = layer.self_attn(
                layer.input_layernorm(x.unsqueeze(0)),
                position_embeddings=rope,
                attention_mask=None,
            )[0]
            return x + out.squeeze(0)
    else:
        def f(x):
            out = layer.mlp(layer.post_attention_layernorm(x.unsqueeze(0)))
            return x + out.squeeze(0)
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
    def f(x):                                  # x: (seq, d)
        seq = x.shape[0]
        position_ids = torch.arange(seq, device=x.device).unsqueeze(0)
        rope = model.rotary_emb(x.unsqueeze(0), position_ids=position_ids)
        attn_out = layer.attention(
            layer.input_layernorm(x.unsqueeze(0)),
            attention_mask=None,
            position_embeddings=rope,
        )[0]
        mlp_out = layer.mlp(layer.post_attention_layernorm(x.unsqueeze(0)))
        return x + attn_out.squeeze(0) + mlp_out.squeeze(0)
    return f


_SUBLAYER_FN_REGISTRY = {
    "GPT2Model":    lambda layer, _model, sublayer: _sublayer_fn_gpt2(layer, sublayer),
    "LlamaModel":   _sublayer_fn_llama,
    "Qwen3Model":   _sublayer_fn_qwen3,
    "GPTNeoXModel": _sublayer_fn_pythia,
}


# Per-architecture getters for the attn and ffn submodules used in capture_all_hidden.
# Each value is (attn_getter, ffn_getter) where getter: layer -> nn.Module.
# For parallel-residual architectures (Pythia), both getters return the same
# module (attention) and the key stored is "block" not "attn"/"ffn".
_CAPTURE_MODS = {
    "GPT2Model":    (lambda l: l.attn,      lambda l: l.mlp),
    "LlamaModel":   (lambda l: l.self_attn, lambda l: l.mlp),
    "Qwen3Model":   (lambda l: l.self_attn, lambda l: l.mlp),
    "GPTNeoXModel": (lambda l: l.attention, lambda l: l.mlp),
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
    arch    = type(model).__name__

    if arch not in _CAPTURE_MODS:
        raise ValueError(
            f"Unsupported architecture {arch!r}. "
            f"Register it in _CAPTURE_MODS. "
            f"Supported: {list(_CAPTURE_MODS)}"
        )
    attn_getter, ffn_getter = _CAPTURE_MODS[arch]

    # embed_out: residual entering block 0 (pre-LN, so a pre_hook is correct here)
    def _hook_embed(_mod, args):
        mode.store[("embed", "out")] = args[0].detach().cpu().clone()
    handles.append(layers[0].register_forward_pre_hook(_hook_embed))

    for i, layer in enumerate(layers):
        # Mark the sublayer output id so the dispatch mode knows which add to capture
        if arch == "GPTNeoXModel":
            block_mod = ffn_getter(layer)
            def _set_block(_mod, _inp, out, _i=i):
                a = out[0] if isinstance(out, tuple) else out
                mode.pending[id(a)] = (_i, "block")
            handles.append(block_mod.register_forward_hook(_set_block))
        else:
            attn_mod = attn_getter(layer)
            def _set_attn(_mod, _inp, out, _i=i):
                a = out[0] if isinstance(out, tuple) else out
                mode.pending[id(a)] = (_i, "attn")
            handles.append(attn_mod.register_forward_hook(_set_attn))

            ffn_mod = ffn_getter(layer)
            def _set_ffn(_mod, _inp, out, _i=i):
                a = out[0] if isinstance(out, tuple) else out
                mode.pending[id(a)] = (_i, "ffn")
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
    if isinstance(inputs, torch.Tensor):
        device = inputs.device
    else:
        device = next(layer.parameters()).device
    dtype = next(layer.parameters()).dtype
    hidden_out = hidden_out.to(device=device, dtype=dtype)
    fn    = _sublayer_fn(layer, sublayer, model=model)

    B = hidden_out.shape[0]
    jacs = torch.stack([_jac_single(fn, hidden_out[b], jac_chunk) for b in range(B)])
    return hidden_out, jacs


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
      det         (B, seq) — determinant of each J
      sigma_ratio (B, seq) — max(σ) / min(σ), i.e. the condition number
    """
    det = torch.linalg.det(jac)
    sv  = torch.linalg.svdvals(jac)         # (B, seq, d), descending
    return {"det": det, "sigma_ratio": sv[..., 0] / sv[..., -1]}
