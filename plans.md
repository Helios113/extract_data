# Plan: Unified Extraction Script (`run.py`)

## Context

The repo has three separate extraction scripts that share a common core loop but differ only in how they source input data:

- `extract.py` — single text string → Jacobians for all layers
- `extract_dataset.py` — HuggingFace streaming dataset → N chunked token sequences
- `extract_manifold.py` — geometric manifold sampler → N sequences of float embeddings

The user wants one script, `run.py`, where a JSON config fully specifies the model, sequence length, data source, and output. Running it produces an HDF5 with Jacobian stats and latent (hidden state) representations for every (layer, sublayer) pair across all samples.

---

## New File

**`/nfs-share/pa511/code_bases/extract_data/run.py`**

---

## Config Format

```json
{
  "model":   "gpt2",
  "device":  "cpu",
  "output":  "out/run.h5",          // optional; can be overridden via CLI positional arg
  "sampling": {
    "n_samples":  32,
    "seq_len":    64,
    "batch_size": 4
  },
  "source": {
    "type": "dataset",              // "dataset" | "manifold" | "benchmark"
    ...
  }
}
```

### Source: `"dataset"` (HuggingFace)

```json
"source": {
  "type":        "dataset",
  "name":        "wikitext",
  "config":      "wikitext-2-raw-v1",
  "split":       "train",
  "text_column": "text"
}
```

### Source: `"manifold"` (geometric — existing types)

```json
"source": {
  "type":         "manifold",
  "manifold":     "sphere",
  "manifold_dim": 5,
  "ambient_dim":  768,
  "noise_std":    0.0,
  "seed":         0,
  "scales":       null,
  "project_dim":  null
}
```
`ambient_dim` must equal `d_model`, or set `project_dim` for a fixed random projection.

### Source: `"benchmark"` (skdim BenchmarkManifolds)

```json
"source": {
  "type": "benchmark",
  "name": "M7_Roll",               // any key from BenchmarkManifolds.dict_gen
  "seed": 42
}
```
Points are always projected to `d_model` via a fixed random orthonormal projection (ambient dims vary per manifold). True intrinsic dimension is read from `bm.truth` and stored in `/meta`.

---

## HDF5 Layout (unified across all sources)

```
/meta
  @attrs: model, source_type, n_samples, seq_len, batch_size
          + source-specific: [dataset_name, dataset_config, split]
                           | [manifold, manifold_dim, ambient_dim, noise_std]
                           | [benchmark_name, benchmark_true_id, ambient_dim]
/samples/{i}/input_ids         — (seq_len,) int64          # dataset only
/samples/{i}/manifold_points   — (seq_len, ambient_dim)    # manifold/benchmark
/samples/{i}/layer_{j}/{sub}/hidden_state  — (seq_len, d)
/samples/{i}/layer_{j}/{sub}/det           — (seq_len,)
/samples/{i}/layer_{j}/{sub}/sigma_ratio   — (seq_len,)
```

---

## Code Structure

### Imports / reuse

| Reused symbol | From |
|---|---|
| `extract_target()` | `extract.py` |
| `chunk_dataset()` | `extract_dataset.py` |
| `ManifoldConfig`, `ManifoldDataset` | `hf_jacobian.manifold_dataset` |
| `_layers()`, `hj.load()` | `hf_jacobian` / `hf_jacobian.jacobian` |
| `skdim.datasets.BenchmarkManifolds` | `skdim` |

### Functions

```python
def parse_args()
    # positional: config, output (output optional if set in config)
    # optional: --device

def write_meta(f, cfg, model_name, n_actual, source_type, source_cfg)
    # writes /meta group with all attrs

def write_sample_inputs(f, si, batch_idx, input_ids_or_points)
    # writes either input_ids or manifold_points for one sample

def write_sample_stats(f, si, layer_idx, sublayer, hidden, stats, batch_idx)
    # writes hidden_state, det, sigma_ratio for one sample

def iter_dataset_batches(cfg, tok, samp_cfg, device)
    # -> Iterator[(offset, ids_tensor, None)]
    # reuses chunk_dataset(); yields (batch_start, (B, seq_len) int64, None)

def iter_manifold_batches(cfg, samp_cfg, device)
    # -> Iterator[(offset, pts_tensor, raw_pts)]
    # builds ManifoldConfig + ManifoldDataset + DataLoader
    # yields (batch_start, (B, seq_len, d_model) float, (B, seq_len, D) float)

def iter_benchmark_batches(cfg, samp_cfg, d_model, device)
    # -> Iterator[(offset, pts_tensor, raw_pts)]
    # calls BenchmarkManifolds.generate(n=n_samples*seq_len)[name]
    # reshapes (N, D) -> (n_samples, seq_len, D), projects D -> d_model
    # yields batches of (B, seq_len, d_model) float

def main()
    # 1. parse args + load config
    # 2. hj.load(model_name, device) → model, tok
    # 3. dispatch to iter_*_batches()
    # 4. open HDF5, write_meta()
    # 5. common loop:
    #      for offset, batch, raw in iter_batches:
    #        write_sample_inputs(...)
    #        for layer_idx, sublayer:
    #          hidden, stats = extract_target(model, batch, layer_idx, sublayer)
    #          for b in range(B): write_sample_stats(...)
    # 6. print saved path
```

### Benchmark projection detail

`BenchmarkManifolds.generate(n=n_samples*seq_len)[name]` returns `(N, D)` float64 array. Project to `d_model` with a fixed random orthonormal matrix `P ∈ R^{D×d_model}` (via `torch.linalg.qr` seeded from `source_cfg["seed"]`). Reshape to `(n_samples, seq_len, d_model)` and batch normally.

---

## Example Configs to Create

- `configs/run_dataset.json` — wikitext example
- `configs/run_manifold.json` — sphere example
- `configs/run_benchmark.json` — M7_Roll example

---

## CLI

```
python run.py config.json [output.h5] [--device cuda]
```
`output` defaults to `cfg["output"]` if present in the config; required otherwise.

---

## Verification

```bash
# dataset source
uv run python run.py configs/run_dataset.json out/test_dataset.h5 --device cpu

# geometric manifold
uv run python run.py configs/run_manifold.json out/test_manifold.h5

# benchmark manifold
uv run python run.py configs/run_benchmark.json out/test_benchmark.h5

# inspect output
uv run python -c "
import h5py
with h5py.File('out/test_dataset.h5') as f:
    print(dict(f['meta'].attrs))
    print(list(f['samples']['0'].keys()))
    print(f['samples/0/layer_0/attn/hidden_state'].shape)
"
```
