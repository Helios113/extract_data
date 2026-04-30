# hf-jacobian

Extract residual-stream activations (and optionally Jacobians) from transformer models, then estimate intrinsic dimension.

## Setup

```bash
uv sync
```

## Data storage (`upload.py`)

Large output files are stored on a Hetzner Storage Box and tracked via content-addressed pointer files in `.ptrs/`. This replaces DVC — no server daemon, no config beyond `upload.cfg`.

### Configure

Copy and fill in `upload.cfg` (already gitignored):

```ini
[remote]
host = <username>.your-storagebox.de
username = <username>
password = <password>
# Target directory on the remote
remote_path = data
# Number of parallel SFTP connections for push/pull (default: 4)
workers = 4
```

### Commands

```bash
uv run python upload.py push out/               # upload a file or directory
uv run python upload.py pull qwen3_random_tokens.h5   # download by name
uv run python upload.py ls                      # interactive browser — navigate dirs, select files to pull
```
### Typical workflow

```bash
# run experiment
uv run python run.py configs/qwen3_latent.json

# push output, optionally delete local copy to free disk
uv run python upload.py push out/qwen3_latent.h5

# commit the pointer so the version is tracked
git add .ptrs/qwen3_latent.h5.ptr
git commit -m "push qwen3_latent results"

# later, on another machine — pull just what you need
uv run python upload.py ls
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
  "weights": "real",
  "output":  "out/run.h5",
  "compute_jacobians": false,
  "compute_jacobian_stats": false,
  "sampling": { "n_samples": 32, "seq_len": 64, "batch_size": 4 },
  "source": { ... }
}
```

`"weights"` can be `"real"` (default) or `"random"` (randomly re-initialised weights, same architecture).

### Data sources

**HuggingFace dataset** — tokenised text, activations driven by real token IDs:
```json
{ "type": "dataset", "name": "wikitext", "config": "wikitext-2-raw-v1",
  "split": "train", "text_column": "text" }
```

**Manifold** — synthetic geometric points fed as `inputs_embeds`:
```json
{ "type": "manifold", "manifold": "plane", "manifold_dim": 10,
  "ambient_dim": 11, "noise_std": 0.0, "seed": 0,
  "neighbourhood_radius": 0.1 }
```
- `manifold`: `plane` | `sphere` | `ellipsoid` | `hyperboloid`
- `ambient_dim` defaults to `manifold_dim`; projected to `d_model` automatically
- `neighbourhood_radius` (plane only): each sequence is sampled within an L2 ball of this radius around a random anchor — all tokens in a sequence stay local
- `scales` (ellipsoid only): list of `d+1` axis lengths

**Random tokens** — uniform random token IDs looked up in the model's embedding matrix:
```json
{ "type": "random_tokens", "seed": 0, "vocab_size": null }
```
- `vocab_size`: cap the vocabulary (default: full model vocab)

**Benchmark** — skdim `BenchmarkManifolds`:
```json
{ "type": "benchmark", "name": "Affine3", "seed": 42 }
```

## HDF5 layout

All float datasets are gzip-9 + byte-shuffle compressed.

```
/meta                               attrs: model, weights, d_model, n_layers,
│                                          source_type, n_samples, seq_len, batch_size,
│                                          compute_jacobians, compute_jacobian_stats,
│                                          + source-specific attrs (see below)
└── samples/
    └── {i}/                        one per sample
        ├── embed_out    (seq, d)   residual stream entering block 0
        ├── final_hidden (seq, d)   residual stream after the last block
        │
        ├── input_ids    (seq,)     int64  [dataset only]
        ├── token_ids    (seq,)     int64  [random_tokens only]
        ├── manifold_points (seq, ambient_dim)  [manifold / benchmark only]
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
        │   │   └── jacobian     (seq, d, d)   [gzip-9 compressed]
```

With `compute_jacobian_stats: true` (requires `compute_jacobians`):

```
        │   ├── attn/
        │   │   ├── det          (seq,)   det(J_p)
        │   │   └── sigma_ratio  (seq,)   max(σ) / min(σ)
```

### Source-specific meta attrs

| source_type | extra attrs |
|---|---|
| `dataset` | `dataset_name`, `dataset_config`, `split`, `text_column`, `tokenizer` |
| `manifold` | `manifold`, `manifold_dim`, `ambient_dim`, `noise_std`, `seed`, `project_dim`, `neighbourhood_radius`, `scales` |
| `random_tokens` | `seed`, `vocab_size` |
| `benchmark` | `benchmark_name`, `benchmark_true_id`, `ambient_dim` |

## Intrinsic dimension analysis

### Compute ESS + TwoNN

```bash
python analyze_id.py <h5_file> <out.csv> [--pos 10 49 -1] [--depth layer_3/attn] [--ess-k 100] [--ess-d 1]
```

- `--pos`: one or more token positions (negative indices supported, e.g. `-1` = last)
- `--depth`: single depth key; omit for all depths
- Always runs ESS-a and ESS-b in parallel; outputs columns `twonn`, `ess_a`, `ess_b`, `n`
- CSV header contains `#`-prefixed metadata lines — read with `pd.read_csv(f, comment='#')`
- Results are flushed after each cell; partial output is preserved on cancel
- If `out.csv` exists, a timestamp suffix is added — existing files are never overwritten

### Plot

```bash
python plot_id.py <csv_file> [--method twonn|ess_a|ess_b] [--pos 10 49] [--out fig.png]
```

- Colour = token position, line style = estimator
- `--method` selects a single estimator; default plots all available
- Expects columns `twonn`, `ess_a`, `ess_b` as produced by `analyze_id.py`

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

| Model class | attn | ffn |
|-------------|------|-----|
| `GPT2Model` | `layer.attn(layer.ln_1(x))[0]` | `layer.mlp(layer.ln_2(x))` |
| `LlamaModel` | `layer.self_attn(layer.input_layernorm(x), position_embeddings=rope)[0]` | `layer.mlp(layer.post_attention_layernorm(x))` |
| `Qwen3Model` | same as Llama, `attention_mask=None` required explicitly | `layer.mlp(layer.post_attention_layernorm(x))` |
| `GPTNeoXModel` | **parallel residual only** — `sublayer="block"` gives `x + attn(...) + mlp(...)` | (same block) |

For models with RoPE (`LlamaModel`, `Qwen3Model`, `GPTNeoXModel`), embeddings are
recomputed from `model.rotary_emb` at each prefix length.

**Pythia note:** GPT-NeoX uses a parallel residual (`x + attn(LN1(x)) + mlp(LN2(x))`
in a single add). Only `sublayer="block"` is valid — `"attn"` and `"ffn"` will raise.

Unsupported architectures raise immediately with a clear error. Add new ones to
`_SUBLAYER_FN_REGISTRY` in [jacobian.py](src/hf_jacobian/jacobian.py).

### Computation

For each position `p`, the forward runs on `x[0:p+1]` so attention sees the correct
KV context. Only the gradient at position `p` is kept.

The inner loop chunks over output dimensions (`jac_chunk`) to bound peak VRAM:

```
peak VRAM ≈ jac_chunk × (p+1) × d   per position
```

Reduce `jac_chunk` if you hit OOM; increase it for fewer kernel launches (faster).



Back project the space?
Train GPT1
Dim est of the manifold
