"""
Extract hidden states and Jacobian stats over manifold-sampled sequences.

Each sequence is a block of `seq_len` points sampled from a geometric manifold
(plane, sphere, ellipsoid, or hyperboloid) embedded in R^ambient_dim, fed to
the transformer as inputs_embeds instead of token embeddings.

Config JSON format:
  {
    "model":  "gpt2",
    "device": "cpu",
    "manifold": {
      "type":         "sphere",
      "manifold_dim": 5,
      "ambient_dim":  768,
      "noise_std":    0.0,
      "scales":       null
    },
    "sampling": {
      "n_samples":  32,
      "seq_len":    64,
      "batch_size": 4
    }
  }

ambient_dim must match the model's d_model unless you also set "project_dim"
inside the manifold block, which adds a fixed random linear projection.

HDF5 layout:
  /meta attrs: model, manifold_type, manifold_dim, ambient_dim, noise_std,
               n_samples, seq_len, batch_size
  /samples/{i}/manifold_points              — (seq_len, ambient_dim)
  /samples/{i}/layer_{j}/{sub}/hidden_state — (seq_len, d_model)
  /samples/{i}/layer_{j}/{sub}/det          — (seq_len,)
  /samples/{i}/layer_{j}/{sub}/sigma_ratio  — (seq_len,)

Usage:
  python extract_manifold.py config.json out.h5
  python extract_manifold.py config.json out.h5 --device cuda
"""

import argparse
import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

import hf_jacobian as hj
from extract import extract_target
from hf_jacobian.jacobian import _layers
from hf_jacobian.manifold_dataset import ManifoldConfig, ManifoldDataset

_SUBLAYERS = ("attn", "ffn")


def parse_args():
    p = argparse.ArgumentParser(description="Extract Jacobian stats over manifold samples")
    p.add_argument("config", help="JSON config file")
    p.add_argument("output", help="Output HDF5 file path")
    p.add_argument("--device", default=None, help="Override device from config")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name  = cfg["model"]
    device      = args.device or cfg.get("device", "cpu")
    m_cfg_raw   = cfg["manifold"]
    samp_cfg    = cfg["sampling"]

    n_samples   = samp_cfg["n_samples"]
    seq_len     = samp_cfg["seq_len"]
    batch_size  = samp_cfg.get("batch_size", 1)
    project_dim = m_cfg_raw.get("project_dim", None)

    manifold_cfg = ManifoldConfig(
        manifold     = m_cfg_raw["type"],
        manifold_dim = m_cfg_raw["manifold_dim"],
        ambient_dim  = m_cfg_raw["ambient_dim"],
        n_samples    = n_samples,
        seq_len      = seq_len,
        noise_std    = m_cfg_raw.get("noise_std", 0.0),
        seed         = m_cfg_raw.get("seed", 0),
        scales       = m_cfg_raw.get("scales", None),
    )

    print(f"Loading {model_name!r} on {device} ...")
    model, _ = hj.load(model_name, device=device)
    n_layers  = len(_layers(model))
    print(f"Model has {n_layers} layers")

    print(f"\nGenerating {n_samples} × {seq_len} {manifold_cfg.manifold} "
          f"(d={manifold_cfg.manifold_dim}, D={manifold_cfg.ambient_dim}) samples ...")
    dataset = ManifoldDataset(manifold_cfg, project_dim=project_dim)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"  {len(dataset)} sequences, batch_size={batch_size}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["model"]         = model_name
        meta.attrs["manifold_type"] = manifold_cfg.manifold
        meta.attrs["manifold_dim"]  = manifold_cfg.manifold_dim
        meta.attrs["ambient_dim"]   = manifold_cfg.ambient_dim
        meta.attrs["noise_std"]     = manifold_cfg.noise_std
        meta.attrs["n_samples"]     = n_samples
        meta.attrs["seq_len"]       = seq_len
        meta.attrs["batch_size"]    = batch_size

        sample_idx = 0
        for batch in loader:
            B      = batch.shape[0]
            pts_t  = batch.to(device)          # (B, seq_len, D)
            print(f"[samples {sample_idx}–{sample_idx + B - 1}]")

            for b in range(B):
                f.require_group(f"samples/{sample_idx + b}").create_dataset(
                    "manifold_points", data=batch[b].numpy()
                )

            for layer_idx in range(n_layers):
                for sublayer in _SUBLAYERS:
                    try:
                        hidden, stats = extract_target(model, pts_t, layer_idx, sublayer)
                    except ValueError as e:
                        print(f"  layer {layer_idx}/{sublayer} skipped: {e}")
                        continue

                    for b in range(B):
                        grp = f.require_group(
                            f"samples/{sample_idx + b}/layer_{layer_idx}/{sublayer}"
                        )
                        grp.create_dataset("hidden_state", data=hidden[b].numpy())
                        grp.create_dataset("det",          data=stats["det"][b].numpy())
                        grp.create_dataset("sigma_ratio",  data=stats["sigma_ratio"][b].numpy())

            sample_idx += B

    print(f"\nSaved → {args.output!r}")


if __name__ == "__main__":
    main()
