# hf-jacobian

Extract residual-stream activations (and optionally Jacobians) from transformer models, then estimate intrinsic dimension.

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
  "weights": "real",
  "output":  "out/run.h5",
  "compute_jacobians": false,
  "compute_jacobian_stats": false,
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

All float datasets are lzf + byte-shuffle compressed. Arrays are stored flat
(one dataset per tensor, first axis = sample index) rather than in per-sample
groups. This lets h5py compress larger contiguous arrays and avoids per-dataset
HDF5 metadata overhead.

**Write performance** (Qwen3-1.7B, B=32, seq=50, measured on one A100):

| strategy | write time | file size |
|---|---|---|
| per-sample gzip-9 (old) | 15.1 s | 295.8 MB |
| per-sample lzf | 5.5 s | 331.2 MB |
| **batched lzf (current)** | **4.6 s** | **319.8 MB** |

The forward pass for the same batch takes ~1.7 s, so HDF5 is no longer the
bottleneck.

```
/meta                               attrs: model, weights, d_model, n_layers,
│                                          source_type, n_samples, seq_len, batch_size,
│                                          compute_jacobians, compute_jacobian_stats,
│                                          + source-specific attrs (see below)
│
├── embed_out        (N, seq, d)   residual stream entering block 0
├── final_hidden     (N, seq, d)   residual stream after the last block
│
├── input_ids        (N, seq)      int64  [dataset only]
├── token_ids        (N, seq)      int64  [random_tokens only]
├── manifold_points  (N, seq, ambient_dim)  [manifold / benchmark only]
│
├── layer_0/
│   ├── attn/
│   │   └── hidden_out  (N, seq, d)
│   └── ffn/
│       └── hidden_out  (N, seq, d)
├── layer_1/  ...
└── layer_N/  ...
```

With `compute_jacobians: true, store_full_jacobians: true`, each sublayer
group also contains:

```
├── layer_0/
│   └── attn/
│       ├── hidden_out   (N, seq, d)
│       └── jacobian     (N, seq, d, d)   [lzf compressed]
```

With `compute_jacobians: true, compute_jacobian_stats: true`, each sublayer
group additionally contains:

```
├── layer_0/
│   └── attn/
│       ├── det          (N, seq)     det(J_p)
│       ├── sigma_max    (N, seq)     largest singular value (= ‖J_p‖_2)
│       └── sigma_min    (N, seq)     smallest singular value (= distance to singular)
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

| Model | HF id | max seq len | attn | ffn |
|-------|-------|------------:|------|-----|
| GPT-2 | `gpt2` | 1 024 | `layer.attn(layer.ln_1(x))[0]` | `layer.mlp(layer.ln_2(x))` |
| Pythia-160M | `EleutherAI/pythia-160m` | 2 048 | **parallel residual only** — `sublayer="block"` | (same block) |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | 131 072 | `layer.self_attn(layer.input_layernorm(x), position_embeddings=rope)[0]` | `layer.mlp(layer.post_attention_layernorm(x))` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 40 960 | same as Llama, `attention_mask=None` required explicitly | `layer.mlp(layer.post_attention_layernorm(x))` |

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



`vmap` over positions is also not possible: each position `p` uses a different-length
prefix `x[:p+1]`, and `vmap` requires uniform shapes across the batch dimension.

`vmap` over the *data* batch dim B *is* used: `_jac_batched` calls
`torch.autograd.grad(out[:, p], x_p, grad_outputs=eye[i0:i1].expand(_, B, d),
is_grads_batched=True)` once per (position, output-chunk), getting B
independent Jacobians at no extra autograd cost. Speedup is significant for
larger models; for gpt2 the bottleneck is launch overhead, not compute.
