"""
Extract hidden states and Jacobian stats over samples from a HuggingFace dataset.

Samples are processed in batches: the forward pass runs over all B samples at once,
and the seq*d Jacobian backward passes each cover the whole batch via the sum trick
(valid because batch items are independent). This gives a factor-B reduction in both
forward and backward pass count compared to per-sample processing.

Config JSON format:
  {
    "model":  "gpt2",
    "device": "cpu",              (optional, default "cpu")
    "dataset": {
      "name":        "wikitext",
      "config":      "wikitext-2-raw-v1",   (optional HF dataset config name)
      "split":       "train",               (optional, default "train")
      "text_column": "text"                 (optional, default "text")
    },
    "sampling": {
      "n_samples":  32,
      "seq_len":    128,
      "batch_size": 8             (optional, default 1)
    }
  }

HDF5 layout:
  /meta attrs: model, dataset_name, dataset_config, split, n_samples, seq_len, batch_size
  /samples/{i}/input_ids                    — (seq_len,)
  /samples/{i}/layer_{j}/{sub}/hidden_state — (seq_len, d)
  /samples/{i}/layer_{j}/{sub}/det          — (seq_len,)
  /samples/{i}/layer_{j}/{sub}/sigma_ratio  — (seq_len,)

Usage:
  python extract_dataset.py config.json out.h5
  python extract_dataset.py config.json out.h5 --device cuda
"""

import argparse
import json
from pathlib import Path

import h5py
import torch
from datasets import load_dataset

import hf_jacobian as hj
from extract import extract_target
from hf_jacobian.jacobian import _layers

_SUBLAYERS = ("attn", "ffn")


def parse_args():
    p = argparse.ArgumentParser(description="Extract Jacobian stats over a HF dataset")
    p.add_argument("config", help="JSON config file")
    p.add_argument("output", help="Output HDF5 file path")
    p.add_argument("--device", default=None, help="Override device from config")
    return p.parse_args()


def chunk_dataset(tok, dataset, text_column: str, seq_len: int, n_samples: int):
    """
    Tokenize examples from the dataset, concatenate into a flat token stream,
    and slice into non-overlapping seq_len chunks.
    Returns a list of n_samples lists of token ids.
    """
    all_ids = []
    need    = n_samples * seq_len

    for ex in dataset:
        text = ex[text_column].strip() if isinstance(ex[text_column], str) else ""
        if not text:
            continue
        all_ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        if len(all_ids) >= need:
            break

    chunks = [
        all_ids[i : i + seq_len]
        for i in range(0, len(all_ids) - seq_len + 1, seq_len)
    ]
    return chunks[:n_samples]


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    device     = args.device or cfg.get("device", "cpu")
    ds_cfg     = cfg["dataset"]
    samp_cfg   = cfg["sampling"]

    ds_name    = ds_cfg["name"]
    ds_config  = ds_cfg.get("config", None)
    ds_split   = ds_cfg.get("split", "train")
    text_col   = ds_cfg.get("text_column", "text")
    n_samples  = samp_cfg["n_samples"]
    seq_len    = samp_cfg["seq_len"]
    batch_size = samp_cfg.get("batch_size", 1)

    print(f"Loading {model_name!r} on {device} ...")
    model, tok = hj.load(model_name, device=device)
    n_layers   = len(_layers(model))
    print(f"Model has {n_layers} layers")

    print(f"\nLoading dataset {ds_name!r}" + (f" ({ds_config})" if ds_config else "") + f" [{ds_split}] ...")
    ds = load_dataset(ds_name, ds_config, split=ds_split, streaming=True)

    print(f"Chunking into {n_samples} × {seq_len}-token samples ...")
    chunks = chunk_dataset(tok, ds, text_col, seq_len, n_samples)
    if len(chunks) < n_samples:
        print(f"  Warning: only {len(chunks)} complete chunks available (wanted {n_samples})")
    n_actual = len(chunks)
    print(f"  Got {n_actual} samples, batch_size={batch_size}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["model"]          = model_name
        meta.attrs["dataset_name"]   = ds_name
        meta.attrs["dataset_config"] = ds_config or ""
        meta.attrs["split"]          = ds_split
        meta.attrs["n_samples"]      = n_actual
        meta.attrs["seq_len"]        = seq_len
        meta.attrs["batch_size"]     = batch_size

        for batch_start in range(0, n_actual, batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            B            = len(batch_chunks)
            sample_range = range(batch_start, batch_start + B)

            ids_t = torch.tensor(batch_chunks, dtype=torch.long).to(device)  # (B, seq_len)
            print(f"[samples {batch_start}–{batch_start + B - 1}]")

            # Save input_ids per sample upfront
            for b, si in enumerate(sample_range):
                f.require_group(f"samples/{si}").create_dataset(
                    "input_ids", data=ids_t[b].cpu().numpy()
                )

            # One forward+backward pass per (layer, sublayer) covers the whole batch
            for layer_idx in range(n_layers):
                for sublayer in _SUBLAYERS:
                    try:
                        hidden, stats = extract_target(model, ids_t, layer_idx, sublayer)
                        # hidden: (B, seq_len, d)   stats: (B, seq_len)
                    except ValueError as e:
                        print(f"  layer {layer_idx}/{sublayer} skipped: {e}")
                        continue

                    for b, si in enumerate(sample_range):
                        grp = f.require_group(f"samples/{si}/layer_{layer_idx}/{sublayer}")
                        grp.create_dataset("hidden_state", data=hidden[b].numpy())
                        grp.create_dataset("det",          data=stats["det"][b].numpy())
                        grp.create_dataset("sigma_ratio",  data=stats["sigma_ratio"][b].numpy())

    print(f"\nSaved → {args.output!r}")


if __name__ == "__main__":
    main()
