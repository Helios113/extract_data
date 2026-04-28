"""
Extract hidden states and Jacobian stats for all layers and sublayers.

Config JSON format:
  {
    "model":  "gpt2",
    "device": "cpu"          (optional, default "cpu")
  }

HDF5 layout:
  /meta attrs: model, text
  /meta/input_ids                  — (seq,) int64
  /layer_{i}/{sub}/hidden_state    — (seq, d)  residual input at the sublayer
  /layer_{i}/{sub}/det             — (seq,)    det(J_p)
  /layer_{i}/{sub}/sigma_ratio     — (seq,)    σ_max / σ_min

Usage:
  python extract.py config.json out.h5
  python extract.py config.json out.h5 --text "The quick brown fox"
"""

import argparse
import json
from pathlib import Path

import h5py

import hf_jacobian as hj
from hf_jacobian.jacobian import _capture, _block_jac_from_graph, _layers

_DEFAULT_TEXT = (
    "The transformer architecture has become the dominant paradigm in natural "
    "language processing, enabling models to capture long-range dependencies "
    "through self-attention mechanisms."
)

_SUBLAYERS = ("attn", "ffn")


def parse_args():
    p = argparse.ArgumentParser(description="Extract Jacobian stats to HDF5")
    p.add_argument("config", help="JSON config file (model name, optional device)")
    p.add_argument("output", help="Output HDF5 file path")
    p.add_argument("--text", default=_DEFAULT_TEXT)
    return p.parse_args()


def extract_target(model, input_ids, layer_idx, sublayer):
    x_leaf, output = _capture(model, input_ids, layer_idx, sublayer)
    jac    = _block_jac_from_graph(output, x_leaf)
    stats  = hj.jacobian_stats(jac)
    hidden = x_leaf.detach().cpu()[0]   # (seq, d)
    return hidden, {k: v.cpu() for k, v in stats.items()}


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    device     = cfg.get("device", "cpu")

    print(f"Loading {model_name!r} on {device} ...")
    model, tok = hj.load(model_name, device=device)

    n_layers = len(_layers(model))
    print(f"Model has {n_layers} layers")

    input_ids = hj.tokenize(tok, args.text, device=device)
    print(f"Tokenized to {input_ids.shape[1]} tokens\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["model"] = model_name
        meta.attrs["text"]  = args.text
        meta.create_dataset("input_ids", data=input_ids[0].cpu().numpy())

        for layer_idx in range(n_layers):
            for sublayer in _SUBLAYERS:
                key = f"layer_{layer_idx}/{sublayer}"
                print(f"[{key}]")
                try:
                    hidden, stats = extract_target(model, input_ids, layer_idx, sublayer)
                except ValueError as e:
                    print(f"  skipped: {e}")
                    continue

                grp = f.require_group(key)
                grp.create_dataset("hidden_state", data=hidden.numpy())
                grp.create_dataset("det",          data=stats["det"].numpy())
                grp.create_dataset("sigma_ratio",  data=stats["sigma_ratio"].numpy())

    print(f"\nSaved → {args.output!r}")


if __name__ == "__main__":
    main()
