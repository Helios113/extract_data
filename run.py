"""
Unified extraction script: Jacobian stats + latent representations.

One config file sets the model, sequence length, and data source. The script
runs extraction for all (layer, sublayer) pairs and saves everything to HDF5.

Config JSON format:
  {
    "model":   "gpt2",
    "device":  "cpu",           (optional, default "cpu")
    "weights": "real",          (optional, "real" | "random"; default "real")
    "output":  "out/run.h5",    (optional; can be overridden via CLI positional arg)
    "sampling": {
      "n_samples":  32,
      "seq_len":    64,
      "batch_size": 4           (optional, default 1)
    },
    "source": { "type": "dataset" | "manifold" | "benchmark", ... }
  }

Source "dataset" — HuggingFace streaming dataset:
  { "type": "dataset", "name": "wikitext", "config": "wikitext-2-raw-v1",
    "split": "train", "text_column": "text" }

Source "manifold" — geometric manifold (plane/sphere/ellipsoid/hyperboloid):
  { "type": "manifold", "manifold": "sphere", "manifold_dim": 5,
    "ambient_dim": 768, "noise_std": 0.0, "seed": 0, "scales": null,
    "project_dim": null }
  ambient_dim must equal d_model, or set project_dim for a random projection.

Source "benchmark" — skdim BenchmarkManifolds:
  { "type": "benchmark", "name": "M7_Roll", "seed": 42 }
  Points are projected to d_model via a fixed random orthonormal matrix.
  True intrinsic dimension is stored in /meta.

HDF5 layout:
  /meta attrs: model, source_type, n_samples, seq_len, batch_size,
               [source-specific attrs]
  /samples/{i}/input_ids         — (seq_len,) int64          # dataset only
  /samples/{i}/manifold_points   — (seq_len, ambient_dim)    # manifold / benchmark
  /samples/{i}/layer_{j}/{sub}/hidden_state  — (seq_len, d)
  /samples/{i}/layer_{j}/{sub}/det           — (seq_len,)
  /samples/{i}/layer_{j}/{sub}/sigma_ratio   — (seq_len,)

Usage:
  python run.py config.json [output.h5] [--device DEVICE]
"""

import argparse
import json
from pathlib import Path
from typing import Iterator
import tqdm

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import hf_jacobian as hj
from extract_dataset import chunk_dataset
from hf_jacobian.jacobian import _layers, capture_all_hidden, _causal_block_jac, jacobian_stats
from hf_jacobian.manifold_dataset import ManifoldConfig, ManifoldDataset, _ortho_frame

_SUBLAYERS = ("attn", "ffn")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified Jacobian + latent extraction")
    p.add_argument("config", help="JSON config file")
    p.add_argument("output", nargs="?", default=None,
                   help="Output HDF5 path (overrides config 'output' key)")
    p.add_argument("--device", default=None, help="Override device from config")
    return p.parse_args()


# ─── source iterators ────────────────────────────────────────────────────────

def iter_dataset_batches(
    src: dict, tok, n_samples: int, seq_len: int, batch_size: int, device: str
) -> Iterator[tuple[int, torch.Tensor, None]]:
    from datasets import load_dataset
    ds = load_dataset(
        src["name"],
        src.get("config", None),
        split=src.get("split", "train"),
        streaming=True,
    )
    if "seed" in src:
        ds = ds.shuffle(seed=src["seed"], buffer_size=10_000)
    chunks = chunk_dataset(tok, ds, src.get("text_column", "text"), seq_len, n_samples)

    for start in range(0, n_samples, batch_size):
        batch_ids = chunks[start : min(start + batch_size, n_samples)]
        ids_t     = torch.tensor(batch_ids, dtype=torch.long).to(device)
        yield start, ids_t, None


def iter_random_token_batches(
    src: dict, model, n_samples: int, seq_len: int, batch_size: int, device: str
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    seed      = src.get("seed", 0)
    vocab_size = src.get("vocab_size", None)

    emb_weight = model.get_input_embeddings().weight  # (V, d_model)
    if vocab_size is None:
        vocab_size = emb_weight.shape[0]

    gen = torch.Generator()
    gen.manual_seed(seed)

    total  = n_samples * seq_len
    ids    = torch.randint(0, vocab_size, (total,), generator=gen)          # (total,)
    embeds = emb_weight[ids].detach().reshape(n_samples, seq_len, -1)       # (n, seq, d)
    raw_ids = ids.reshape(n_samples, seq_len)                                # (n, seq) int

    for start in range(0, n_samples, batch_size):
        batch  = embeds[start : min(start + batch_size, n_samples)].to(device)
        raw    = raw_ids[start : min(start + batch_size, n_samples)]
        yield start, batch, raw


def iter_manifold_batches(
    src: dict, n_samples: int, seq_len: int, batch_size: int, d_model: int, device: str
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    ambient_dim = src.get("ambient_dim", src["manifold_dim"])
    cfg = ManifoldConfig(
        manifold              = src["manifold"],
        manifold_dim          = src["manifold_dim"],
        ambient_dim           = ambient_dim,
        n_samples             = n_samples,
        seq_len               = seq_len,
        noise_std             = src.get("noise_std", 0.0),
        seed                  = src.get("seed", 0),
        scales                = src.get("scales", None),
        neighbourhood_radius  = src.get("neighbourhood_radius", None),
    )
    dataset = ManifoldDataset(cfg, project_dim=src.get("project_dim", d_model))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Generating {n_samples} × {seq_len} {cfg.manifold} "
          f"(d={cfg.manifold_dim}, D={cfg.ambient_dim}) samples ...\n")

    sample_idx = 0
    for batch in loader:
        yield sample_idx, batch.to(device), batch.cpu()
        sample_idx += batch.shape[0]


def iter_benchmark_batches(
    src: dict, n_samples: int, seq_len: int, batch_size: int, d_model: int, device: str
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    import skdim
    name = src["name"]
    seed = src.get("seed", 42)

    bm   = skdim.datasets.BenchmarkManifolds(random_state=seed)
    data = bm.generate(n=n_samples * seq_len)
    X_np = data[name].astype(np.float32)  # (N, D)
    D    = X_np.shape[1]

    gen = torch.Generator()
    gen.manual_seed(seed)
    if D >= d_model:
        P = _ortho_frame(d_model, D, gen)    # (D, d_model) — orthonormal cols in R^D
    else:
        P = _ortho_frame(D, d_model, gen).T  # (D, d_model) — orthonormal rows in R^d_model

    X_t    = torch.from_numpy(X_np) @ P                          # (N, d_model)
    seqs   = X_t.reshape(n_samples, seq_len, d_model)            # (n_samples, seq_len, d_model)
    raw    = torch.from_numpy(X_np).reshape(n_samples, seq_len, D)

    print(f"BenchmarkManifold {name!r}: D={D} → projected to d_model={d_model}\n")

    for start in range(0, n_samples, batch_size):
        batch     = seqs[start : min(start + batch_size, n_samples)].to(device)
        raw_batch = raw[start : min(start + batch_size, n_samples)]
        yield start, batch, raw_batch


# ─── HDF5 helpers ────────────────────────────────────────────────────────────

def write_meta(
    f, model_name: str, src_type: str, src: dict,
    n_samples: int, seq_len: int, batch_size: int, extra: dict,
    weights: str, d_model: int, n_layers: int,
    compute_jacobians: bool, compute_jacobian_stats: bool,
):
    meta = f.create_group("meta")
    meta.attrs["model"]                  = model_name
    meta.attrs["weights"]                = weights
    meta.attrs["d_model"]                = d_model
    meta.attrs["n_layers"]               = n_layers
    meta.attrs["source_type"]            = src_type
    meta.attrs["n_samples"]              = n_samples
    meta.attrs["seq_len"]                = seq_len
    meta.attrs["batch_size"]             = batch_size
    meta.attrs["compute_jacobians"]      = compute_jacobians
    meta.attrs["compute_jacobian_stats"] = compute_jacobian_stats

    if src_type == "dataset":
        meta.attrs["dataset_name"]   = src["name"]
        meta.attrs["dataset_config"] = src.get("config", "")
        meta.attrs["split"]          = src.get("split", "train")
        meta.attrs["text_column"]    = src.get("text_column", "text")
        meta.attrs["tokenizer"]      = model_name
    elif src_type == "manifold":
        meta.attrs["manifold"]               = src["manifold"]
        meta.attrs["manifold_dim"]           = src["manifold_dim"]
        meta.attrs["ambient_dim"]            = src.get("ambient_dim", src["manifold_dim"])
        meta.attrs["noise_std"]              = src.get("noise_std", 0.0)
        meta.attrs["seed"]                   = src.get("seed", 0)
        meta.attrs["project_dim"]            = src.get("project_dim", d_model)
        meta.attrs["neighbourhood_radius"]   = src.get("neighbourhood_radius") or 0.0
        scales = src.get("scales")
        meta.attrs["scales"] = scales if scales is not None else []
    elif src_type == "random_tokens":
        meta.attrs["seed"]       = src.get("seed", 0)
        meta.attrs["vocab_size"] = src.get("vocab_size", 0)   # 0 = full model vocab
    elif src_type == "benchmark":
        for k, v in extra.items():
            meta.attrs[k] = v



def _ds_append(f, name: str, batch_np):
    """Append a (B, ...) batch to a resizable dataset, creating it on first call."""
    if name in f:
        ds = f[name]
        old = ds.shape[0]
        ds.resize(old + batch_np.shape[0], axis=0)
        ds[old:] = batch_np
    else:
        maxshape = (None,) + batch_np.shape[1:]
        f.create_dataset(name, data=batch_np, maxshape=maxshape,
                         compression="lzf", shuffle=True, chunks=(1,) + batch_np.shape[1:])




# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    device     = args.device or cfg.get("device", "cpu")
    output     = args.output or cfg.get("output")
    if output is None:
        raise ValueError("Output path required: pass as positional arg or set 'output' in config")

    samp       = cfg["sampling"]
    n_samples  = samp["n_samples"]
    seq_len    = samp["seq_len"]
    batch_size = samp.get("batch_size", 1)
    src        = cfg["source"]
    src_type   = src["type"]
    compute_jacobians      = cfg.get("compute_jacobians", False)
    store_full_jacobians   = cfg.get("store_full_jacobians", False)
    compute_jacobian_stats = cfg.get("compute_jacobian_stats", False)
    jac_chunk              = cfg.get("jac_chunk", 64)
    weights                = cfg.get("weights", "real")
    if weights not in ("real", "random"):
        raise ValueError(f"'weights' must be 'real' or 'random', got {weights!r}")

    wb = cfg.get("wandb", {})
    if wb is not False:
        import wandb
        wandb.init(
            entity=wb.get("entity"),
            project=wb.get("project", "extract-data"),
            name=wb.get("name") or f"{model_name}-{src_type}",
            tags=wb.get("tags"),
            config=cfg,
        )


    print(f"Loading {model_name!r} on {device} ...")
    if model_name == "custom":
        custom_cfg = hj.Config(**cfg.get("custom_model_config", {}))
        model = hj.CustomModel(custom_cfg).to(device).eval()
        checkpoint = cfg.get("checkpoint")
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        tok     = None
        d_model = custom_cfg.d_model
    else:
        model, tok = hj.load(model_name, device=device)
        d_model    = model.config.hidden_size
    if weights == "random":
        hj.reinit_weights(model)
        print("  weights re-initialised\n")
    n_layers  = len(_layers(model))
    sublayers = ("block",) if type(model).__name__ == "GPTNeoXModel" else _SUBLAYERS
    print(f"  {n_layers} layers, d_model={d_model}\n")

    # pre-flight: compute source-specific extra metadata
    extra_meta: dict = {}
    if src_type == "benchmark":
        import skdim
        _bm = skdim.datasets.BenchmarkManifolds(random_state=src.get("seed", 42))
        extra_meta["benchmark_name"]    = src["name"]
        extra_meta["benchmark_true_id"] = int(_bm.truth.loc[src["name"], "Intrinsic Dimension"])
        extra_meta["ambient_dim"]       = int(_bm.generate(n=seq_len)[src["name"]].shape[1])

    # build iterator
    if src_type == "dataset":
        if tok is None:
            raise ValueError("source type 'dataset' requires a tokenizer — CustomModel has none; use 'manifold' or 'benchmark' instead")
        batches = iter_dataset_batches(src, tok, n_samples, seq_len, batch_size, device)
    elif src_type == "manifold":
        batches = iter_manifold_batches(src, n_samples, seq_len, batch_size, d_model, device)
    elif src_type == "benchmark":
        batches = iter_benchmark_batches(src, n_samples, seq_len, batch_size, d_model, device)
    elif src_type == "random_tokens":
        if model is None:
            raise ValueError("source type 'random_tokens' requires a loaded model")
        batches = iter_random_token_batches(src, model, n_samples, seq_len, batch_size, device)
    else:
        raise ValueError(f"Unknown source type {src_type!r}. Choose: dataset, manifold, benchmark, random_tokens")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        write_meta(
            f, model_name, src_type, src, n_samples, seq_len, batch_size, extra_meta,
            weights=weights, d_model=d_model, n_layers=n_layers,
            compute_jacobians=compute_jacobians, compute_jacobian_stats=compute_jacobian_stats,
        )

        def _gpu_postfix():
            if not torch.cuda.is_available():
                return {}
            dev = torch.cuda.current_device()
            mem_alloc = torch.cuda.memory_allocated(dev) / 1024**3
            mem_total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
            mem_free  = mem_total - mem_alloc
            util = torch.cuda.utilization(dev)
            return {"gpu": f"{util}%", "vram_free": f"{mem_free:.1f}GB"}

        actual_samples = 0
        total_steps = n_samples * n_layers * len(sublayers)
        with tqdm.tqdm(total=total_steps, desc="samples") as pbar:
            for offset, batch, raw in batches:
                B = batch.shape[0]
                actual_samples += B
                if batch.is_floating_point():
                    batch = batch.to(dtype=next(model.parameters()).dtype)

                if not compute_jacobians:
                    store = capture_all_hidden(model, batch)

                    if src_type == "dataset":
                        _ds_append(f, "input_ids", batch.cpu().numpy())
                    elif src_type == "random_tokens":
                        _ds_append(f, "token_ids", raw.numpy())
                    elif raw is not None:
                        _ds_append(f, "manifold_points", raw.numpy())

                    _ds_append(f, "embed_out",    store[("embed", "out")].float().cpu().numpy())
                    _ds_append(f, "final_hidden", store[("final", "out")].float().cpu().numpy())
                    for layer_idx in range(n_layers):
                        for sub in sublayers:
                            if (layer_idx, sub) not in store:
                                continue
                            _ds_append(f, f"layer_{layer_idx}/{sub}/hidden_out",
                                       store[(layer_idx, sub)].float().cpu().numpy())
                    pbar.update(n_layers * len(sublayers) * B)
                    pbar.set_postfix(_gpu_postfix())
                else:
                    for layer_idx in range(n_layers):
                        for sub in sublayers:
                            try:
                                hidden_out, jac = _causal_block_jac(
                                    model, batch, layer_idx, sub, jac_chunk
                                )
                            except ValueError:
                                continue
                            _ds_append(f, f"layer_{layer_idx}/{sub}/hidden_out",
                                       hidden_out.float().cpu().numpy())
                            if store_full_jacobians:
                                _ds_append(f, f"layer_{layer_idx}/{sub}/jacobian",
                                           jac.float().cpu().numpy())
                            if compute_jacobian_stats:
                                stats = jacobian_stats(jac)
                                for key in ("det", "sigma_max", "sigma_min", "singular_values"):
                                    t = stats.get(key)
                                    if t is not None:
                                        _ds_append(f, f"layer_{layer_idx}/{sub}/{key}",
                                                   t.float().cpu().numpy())
                            pbar.update(1)
                            pbar.set_postfix(_gpu_postfix())

                    from hf_jacobian.jacobian import capture_endpoints
                    embed_out, final_hidden = capture_endpoints(model, batch)
                    _ds_append(f, "embed_out",    embed_out.float().cpu().numpy())
                    _ds_append(f, "final_hidden", final_hidden.float().cpu().numpy())

        f["meta"].attrs["n_samples"] = actual_samples
        f["meta"].attrs["n_tokens"]  = actual_samples * seq_len
        print(f"\nActual samples: {actual_samples}  |  tokens: {actual_samples * seq_len}")


if __name__ == "__main__":
    main()
