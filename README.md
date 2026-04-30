# hf-jacobian

Extract residual-stream activations (and optionally Jacobians) from transformer models.

## Changelog

- **Three-flag Jacobian config.** `compute_jacobians` is the master switch;
  `store_full_jacobians` and `compute_jacobian_stats` independently gate writes.
  Stats-only mode (no raw J) is now possible — ~300,000× smaller files. See
  [Config](#config) for the truth table.
- **Stats schema.** `jacobian_stats` now writes `det`, `sigma_max`, `sigma_min`
  (was `det`, `sigma_ratio`). κ⁻¹ derivable post-hoc. See [HDF5 layout](#hdf5-layout).
- **fp32 SVD cast.** `jacobian_stats` upcasts `jac` to fp32 before
  `torch.linalg.det`/`svdvals` — required for bf16/fp16 models (Llama, Qwen3),
  no-op for fp32 (gpt2, Pythia). Autograd stays in native dtype; only the SVD
  measurement is promoted. See [HDF5 layout](#hdf5-layout) note.
- **Batched Jacobian autograd.** `_jac_single` → `_jac_batched`; sublayer fns
  take `(B, seq, d)` directly, no per-item Python loop. See [Computation](#computation).
- **Wandb integration (opt-out).** `wandb.init(config=cfg)` runs at startup
  unless `"wandb": false` in config. Optional `entity/project/name/tags` block.
  Adds `wandb` dep in `pyproject.toml`.
- **SLURM submit script** `run_extract.sbatch`. Per-run dir
  `runs/<jobid>_<configname>/{config.json,run.h5}`, `WANDB_ENTITY` exported
  inside the enroot container, `set +x`-guarded HF-token stub for gated models.
  See [Run](#run).
- **`configs/precision/`** subfolder with the spectrum-only recipes for gpt2 /
  llama-3.2-1b / qwen3-1.7b / pythia-160m used in the precision experiment.
- **Legacy `configs/*_jac.json` migrated.** `gpt2_manifold_ellipsoid_jac.json`,
  `pythia_jac.json`, `qwen3_jac.json` got `store_full_jacobians: true` added
  explicitly, preserving old behavior under the renamed flags.
- **`.gitignore`** now excludes `runs/`, `wandb/`, `logs/` (large H5s, wandb
  local cache, slurm logs that can leak env via `set -x`).

## Setup

```bash
uv sync
```

## Run

Locally:

```bash
python run.py configs/gpt2.json
```

Or on SLURM (per-run dir, config copy, wandb wired up):

```bash
sbatch run_extract.sbatch configs/gpt2.json
```

See `run_extract.sbatch` for the enroot wrapper, log paths, and `WANDB_ENTITY`.

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

Three flags (all default `false`). `compute_jacobians` is the master switch
for the autograd Jacobian path; when on, the other two control what gets
written. Activations (`embed_out`, `final_hidden`, `hidden_out` per sublayer)
are always saved.

| flag | effect (when on) |
|------|--------|
| `compute_jacobians` | run the per-token causal-prefix autograd path |
| `store_full_jacobians` | persist raw `(seq, d, d)` matrix per sublayer (heavy: 768×768 fp32 ≈ 2.36 MB / token for gpt2) |
| `compute_jacobian_stats` | run SVD per Jacobian, save `det`, `sigma_max`, `sigma_min` (3 scalars / token, negligible) |

Valid combinations:

| `compute_jacobians` | `store_full_jacobians` | `compute_jacobian_stats` | output |
|---|---|---|---|
| false | (ignored) | (ignored) | activations only (single forward, fast) |
| true | false | false | activations only (Jacobians computed but discarded — useful for benchmarking) |
| true | true | false | activations + raw J |
| true | false | true | activations + stats only (~300,000× smaller than raw J) |
| true | true | true | activations + raw J + stats |

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

With `compute_jacobians: true, store_full_jacobians: true`, each sublayer
group also contains:

```
    │   ├── attn/
    │   │   ├── hidden_out   (seq, d)
    │   │   └── jacobian     (seq, d, d)   raw per-token Jacobian  [gzip compressed]
```

With `compute_jacobians: true, compute_jacobian_stats: true`, each sublayer
group additionally contains:

```
    │   ├── attn/
    │   │   ├── det          (seq,)     det(J_p)
    │   │   ├── sigma_max    (seq,)     largest singular value (= ‖J_p‖_2)
    │   │   └── sigma_min    (seq,)     smallest singular value (= distance to singular)
```

`κ = sigma_max / sigma_min` and `κ⁻¹ = sigma_min / sigma_max` are derivable
post-hoc — used as the precision-safety threshold (the layer is numerically
invertible at relative precision ε iff `ε < κ⁻¹`).

Note: when the model is loaded in bf16/fp16 (default for Llama-3.2-1B and
Qwen3-1.7B), the Jacobian comes out in that dtype too, but `jacobian_stats`
upcasts to fp32 before SVD/det. The matrix being measured is the native-dtype
Jacobian — only the spectrum measurement runs at fp32 (required by MAGMA, more
accurate for σ_min anyway). No effect on the analysis: σ_min/σ_max/det are
intrinsic to the matrix, fp32 just measures them more truthfully.

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
| `LlamaModel` | `layer.self_attn(layer.input_layernorm(x), position_embeddings=rope)[0]` | `layer.mlp(layer.post_attention_layernorm(x))` |
| `Qwen3Model` | same as Llama, `attention_mask=None` required explicitly | `layer.mlp(layer.post_attention_layernorm(x))` |
| `GPTNeoXModel` | **parallel residual only** — `sublayer="block"` gives `x + attn(...) + mlp(...)` | (same block) |

For models with RoPE (`LlamaModel`, `Qwen3Model`, `GPTNeoXModel`), embeddings are
recomputed from `model.rotary_emb` at each prefix length. They depend only on
sequence position so this is exact.

**Pythia note:** GPT-NeoX uses a parallel residual (`x + attn(LN1(x)) + mlp(LN2(x))`
in a single add). There are no separate attn/ffn residual steps, so only
`sublayer="block"` is valid — `"attn"` and `"ffn"` will raise.

Unsupported architectures raise immediately with a clear error. Add new ones to
`_SUBLAYER_FN_REGISTRY` in [jacobian.py](src/hf_jacobian/jacobian.py).

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

`vmap` over the *data* batch dim B *is* used: `_jac_batched` calls
`torch.autograd.grad(out[:, p], x_p, grad_outputs=eye[i0:i1].expand(_, B, d),
is_grads_batched=True)` once per (position, output-chunk), getting B
independent Jacobians at no extra autograd cost. Speedup is significant for
larger models; for gpt2 the bottleneck is launch overhead, not compute.
