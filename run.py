"""
Unified extraction script: Jacobian stats + latent representations.

One config file sets the model, sequence length, and data source. The script
runs extraction for all (layer, sublayer) pairs and saves everything to HDF5.

Config JSON format:
  {
    "model":   "gpt2",
    "device":  "cpu",           (optional, default "cpu")
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
    chunks = chunk_dataset(tok, ds, src.get("text_column", "text"), seq_len, n_samples)

    for start in range(0, len(chunks), batch_size):
        batch_ids = chunks[start : start + batch_size]
        ids_t     = torch.tensor(batch_ids, dtype=torch.long).to(device)
        yield start, ids_t, None


def iter_manifold_batches(
    src: dict, n_samples: int, seq_len: int, batch_size: int, device: str
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    cfg = ManifoldConfig(
        manifold     = src["manifold"],
        manifold_dim = src["manifold_dim"],
        ambient_dim  = src["ambient_dim"],
        n_samples    = n_samples,
        seq_len      = seq_len,
        noise_std    = src.get("noise_std", 0.0),
        seed         = src.get("seed", 0),
        scales       = src.get("scales", None),
    )
    dataset = ManifoldDataset(cfg, project_dim=src.get("project_dim", None))
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
        batch     = seqs[start : start + batch_size].to(device)
        raw_batch = raw[start : start + batch_size]
        yield start, batch, raw_batch


# ─── HDF5 helpers ────────────────────────────────────────────────────────────

def write_meta(
    f, model_name: str, src_type: str, src: dict,
    n_samples: int, seq_len: int, batch_size: int, extra: dict
):
    meta = f.create_group("meta")
    meta.attrs["model"]       = model_name
    meta.attrs["source_type"] = src_type
    meta.attrs["n_samples"]   = n_samples
    meta.attrs["seq_len"]     = seq_len
    meta.attrs["batch_size"]  = batch_size

    if src_type == "dataset":
        meta.attrs["dataset_name"]   = src["name"]
        meta.attrs["dataset_config"] = src.get("config", "")
        meta.attrs["split"]          = src.get("split", "train")
    elif src_type == "manifold":
        meta.attrs["manifold"]     = src["manifold"]
        meta.attrs["manifold_dim"] = src["manifold_dim"]
        meta.attrs["ambient_dim"]  = src["ambient_dim"]
        meta.attrs["noise_std"]    = src.get("noise_std", 0.0)
    elif src_type == "benchmark":
        for k, v in extra.items():
            meta.attrs[k] = v


def save_inputs(f, si: int, b: int, src_type: str, batch: torch.Tensor, raw):
    grp = f.require_group(f"samples/{si}")
    if src_type == "dataset":
        grp.create_dataset("input_ids",       data=batch[b].cpu().numpy())
    elif raw is not None:
        grp.create_dataset("manifold_points",  data=raw[b].numpy())



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
    compute_jacobian_stats = cfg.get("compute_jacobian_stats", False)
    jac_chunk              = cfg.get("jac_chunk", 64)

    print(f"Loading {model_name!r} on {device} ...")
    model, tok = hj.load(model_name, device=device)
    n_layers   = len(_layers(model))
    d_model    = model.config.hidden_size
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
        batches = iter_dataset_batches(src, tok, n_samples, seq_len, batch_size, device)
    elif src_type == "manifold":
        batches = iter_manifold_batches(src, n_samples, seq_len, batch_size, device)
    elif src_type == "benchmark":
        batches = iter_benchmark_batches(src, n_samples, seq_len, batch_size, d_model, device)
    else:
        raise ValueError(f"Unknown source type {src_type!r}. Choose: dataset, manifold, benchmark")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        write_meta(f, model_name, src_type, src, n_samples, seq_len, batch_size, extra_meta)

        total_steps = n_samples * n_layers * len(_SUBLAYERS)
        with tqdm.tqdm(total=total_steps, desc="samples") as pbar:
            for offset, batch, _ in batches:
                B = batch.shape[0]

                if not compute_jacobians:
                    # single forward — captures everything at once
                    store = capture_all_hidden(model, batch)
                    for b in range(B):
                        sg = f.require_group(f"samples/{offset + b}")
                        sg.create_dataset("embed_out",    data=store[("embed", "out")][b].numpy())
                        sg.create_dataset("final_hidden", data=store[("final", "out")][b].numpy())
                        for layer_idx in range(n_layers):
                            for sub in _SUBLAYERS:
                                if (layer_idx, sub) not in store:
                                    continue
                                grp = f.require_group(f"samples/{offset + b}/layer_{layer_idx}/{sub}")
                                grp.create_dataset("hidden_out", data=store[(layer_idx, sub)][b].numpy())
                    pbar.update(n_layers * len(_SUBLAYERS) * B)
                else:
                    # per-(layer, sublayer) forward for Jacobian graph
                    for layer_idx in range(n_layers):
                        for sub in _SUBLAYERS:
                            try:
                                _, hidden_out, jac = _causal_block_jac(
                                    model, batch, layer_idx, sub, jac_chunk
                                )
                            except ValueError:
                                continue
                            h_out_np = hidden_out.cpu().numpy()
                            jac_np   = jac.cpu().numpy()
                            stats    = jacobian_stats(jac) if compute_jacobian_stats else None
                            for b in range(B):
                                grp = f.require_group(f"samples/{offset + b}/layer_{layer_idx}/{sub}")
                                grp.create_dataset("hidden_out", data=h_out_np[b])
                                grp.create_dataset("jacobian",   data=jac_np[b], compression="gzip")
                                if stats is not None:
                                    det_np   = stats.get("det")
                                    sigma_np = stats.get("sigma_ratio")
                                    if det_np is not None:
                                        grp.create_dataset("det",         data=det_np.cpu().numpy()[b])
                                    if sigma_np is not None:
                                        grp.create_dataset("sigma_ratio", data=sigma_np.cpu().numpy()[b])
                            pbar.update(1)

                    # endpoints captured separately in the Jacobian path
                    from hf_jacobian.jacobian import capture_endpoints
                    embed_out, final_hidden = capture_endpoints(model, batch)
                    for b in range(B):
                        sg = f.require_group(f"samples/{offset + b}")
                        sg.create_dataset("embed_out",    data=embed_out[b].cpu().numpy())
                        sg.create_dataset("final_hidden", data=final_hidden[b].cpu().numpy())



if __name__ == "__main__":
    main()
