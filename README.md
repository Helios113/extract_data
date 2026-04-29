# hf-jacobian

Extract residual-stream activations (and optionally Jacobians) from transformer models.

## Setup

```bash
uv sync
```

## Run

```bash
python run.py configs/gpt2.json
```

## Config

```json
{
  "model":   "gpt2",
  "device":  "cuda",
  "output":  "out/run.h5",
  "compute_jacobians": false,
  "sampling": { "n_samples": 32, "seq_len": 64, "batch_size": 4 },
  "source": { "type": "dataset", "name": "wikitext", "config": "wikitext-2-raw-v1",
              "split": "train", "text_column": "text" }
}
```

Three optional flags (all default `false`):

| flag | effect |
|------|--------|
| `compute_jacobians` | save raw `(seq, d, d)` Jacobian per sublayer |
| `compute_jacobian_stats` | also compute and save `det` / `sigma_ratio` inline (requires `compute_jacobians`) |

Stats can also be computed offline from saved Jacobians via `hf_jacobian.jacobian_stats(jac)`.

## HDF5 layout

```
samples/
└── {i}/                            one per sample
    ├── embed_out      (seq, d)     residual stream entering block 0
    ├── final_hidden   (seq, d)     residual stream after the last block
    │
    ├── layer_0/
    │   ├── attn/
    │   │   └── hidden_out  (seq, d)
    │   └── ffn/
    │       └── hidden_out  (seq, d)
    ├── layer_1/  ...
    └── layer_N/  ...
```

With `compute_jacobians: true`, each sublayer group also contains:

```
    │   ├── attn/
    │   │   ├── hidden_out   (seq, d)
    │   │   └── jacobian     (seq, d, d)   raw per-token Jacobian  [gzip compressed]
```

With `compute_jacobian_stats: true` (requires `compute_jacobians`), additionally:

```
    │   ├── attn/
    │   │   ├── det          (seq,)     det(J_p)
    │   │   └── sigma_ratio  (seq,)     max(σ) / min(σ)
```

## Residual stream across one block

```
embed_out
    │  = layer_0/attn/hidden_out input
    ▼
[ LN → Attn → + ]  ──►  layer_0/attn/hidden_out
                                │
                                ▼
                    [ LN → FFN → + ]  ──►  layer_0/ffn/hidden_out
                                                    │
                                    = layer_1/attn/hidden_out input
                                                   ...
                                                    │
                                              final_hidden
```

`hidden_out` of each sublayer is the input to the next, so consecutive datasets
overlap: `layer_k/ffn/hidden_out == layer_{k+1}/attn/hidden_out input`, and
`layer_N/ffn/hidden_out == final_hidden`.

## Jacobian targets

- **attn**: $J_t = I + \frac{\partial\,\mathrm{Attn}(\mathrm{LN}(x))_t}{\partial x_t}$
- **ffn**: $J_t = I + \frac{\partial\,\mathrm{FFN}(\mathrm{LN}(x'))_t}{\partial x'_t}$

LN is **inside** the graph — the Jacobian is w.r.t. the raw residual `h`, not `LN(h)`.

### Supported architectures

The sublayer function used for Jacobian computation is hardcoded per architecture
to avoid fragile name-pattern matching:

| Model class | attn | ffn |
|-------------|------|-----|
| `GPT2Model` | `layer.attn(layer.ln_1(x))[0]` | `layer.mlp(layer.ln_2(x))` |
| `LlamaModel` | `layer.self_attn(layer.input_layernorm(x), position_embeddings=rope)` | `layer.mlp(layer.post_attention_layernorm(x))` |

For Llama, RoPE embeddings are computed from `model.rotary_emb` at each prefix
length — they depend only on sequence position, not on the hidden states, so
this is exact.

A generic name-based fallback (`_sub` lookup) is used for `CustomModel` and any
unregistered architecture. Add new architectures to `_SUBLAYER_FN_REGISTRY` in
[jacobian.py](src/hf_jacobian/jacobian.py).

### Computation

For each position `p`, the forward runs on `x[0:p+1]` (the full causal prefix) so
that attention sees the correct KV context. Only the gradient at position `p` is
kept; gradients w.r.t. earlier positions are discarded. This is unavoidable for
causal attention — the forward must see the prefix even though the backward only
needs one position.

The inner loop chunks over output dimensions (`jac_chunk`) to bound peak VRAM:

```
peak VRAM ≈ jac_chunk × (p+1) × d   per position
```

Reduce `jac_chunk` if you hit OOM; increase it for fewer kernel launches (faster).

`torch.compile` is incompatible with `is_grads_batched=True` — the compiled backward
produces fake tensors that the vmap internals of `is_grads_batched` cannot access.
Do not apply `torch.compile` to the sublayer function when computing Jacobians.

`vmap` over positions is also not possible: each position `p` uses a different-length
prefix `x[:p+1]`, and `vmap` requires uniform shapes across the batch dimension.
