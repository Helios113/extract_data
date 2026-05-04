"""
Load an HDF5 file produced by run.py and compute ESS and TwoNN intrinsic-dimension
estimates for every (token_position, depth) cell.

Depth is enumerated as:
  embed, layer_0/attn, layer_0/ffn, layer_1/attn, ..., layer_{L-1}/ffn, final

Usage:
  python analyze_id.py <h5_file> [--pos 42] [--depth layer_3/attn] [--out results.csv]
  python analyze_id.py <h5_file> <out_csv> [--pos 42] [--depth layer_3/attn]
  python analyze_id.py out/run.h5 results.csv             # full grid
  python analyze_id.py out/run.h5 results.csv --pos 10    # all depths at token 10
  python analyze_id.py out/run.h5 results.csv --depth layer_2/ffn
  python analyze_id.py out/run.h5 results.csv --pos 10 --depth layer_2/ffn
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "src")
from hf_jacobian import ess, twonn


# ── helpers ──────────────────────────────────────────────────────────────────

def depth_keys(f: h5py.File) -> list[str]:
    """Return ordered depth labels matching the residual stream order."""
    keys = ["embed"]
    layer_names = sorted(
        [k for k in f.keys() if k.startswith("layer_")],
        key=lambda k: int(k.split("_")[1]),
    )
    for ln in layer_names:
        for sub in ("attn", "ffn"):
            if sub in f[ln]:
                keys.append(f"{ln}/{sub}")
    keys.append("final")
    return keys


def load_latents(f: h5py.File, depth: str, pos: int) -> torch.Tensor:
    """
    Return hidden vectors at (depth, pos) across all samples.
    Returns (N, d_model) float32 tensor.
    """
    if depth == "embed":
        X = f["embed_out"][:, pos, :]
    elif depth == "final":
        X = f["final_hidden"][:, pos, :]
    else:
        X = f[f"{depth}/hidden_out"][:, pos, :]
    return torch.from_numpy(np.array(X, dtype=np.float32))


def resolve_out_path(p: str) -> Path:
    path = Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_stem(f"{path.stem}_{ts}")


def load_entropy_lens(model_name: str):
    """Return (ln, lm_head) for entropy computation, architecture-aware."""
    causal = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    ).eval()
    arch = type(causal.base_model).__name__

    # Final layer-norm + unembedding head, keyed by base-model class name.
    _LN_ATTR = {
        "GPT2Model":    lambda m: m.transformer.ln_f,
        "LlamaModel":   lambda m: m.model.norm,
        "Qwen3Model":   lambda m: m.model.norm,
        "GPTNeoXModel": lambda m: m.gpt_neox.final_layer_norm,
    }
    if arch not in _LN_ATTR:
        raise ValueError(
            f"Entropy lens: unsupported architecture {arch!r}. "
            f"Supported: {list(_LN_ATTR)}"
        )
    ln      = _LN_ATTR[arch](causal)
    lm_head = causal.lm_head
    return ln, lm_head


def compute_entropy(X: torch.Tensor, ln, lm_head) -> torch.Tensor:
    """Mean Shannon entropy over N hidden vectors. Returns scalar tensor."""
    with torch.no_grad():
        logits = lm_head(ln(X.float()))          # (N, vocab)
        probs  = logits.softmax(dim=-1)
        h      = -(probs * (probs + 1e-15).log()).sum(dim=-1)   # (N,)
    return h.mean()


def compute_cell(X: torch.Tensor, ess_k: int = 100, ess_d: int = 1,
                 ln=None, lm_head=None, curvature: bool = False) -> dict:
    id_twonn   = twonn(X)
    res_a      = ess(X, k=ess_k, d=ess_d, ver="a", curvature=curvature)
    res_b      = ess(X, k=ess_k, d=ess_d, ver="b", curvature=curvature)
    result = {
        "twonn": id_twonn,
        "ess_a": res_a["dimension"],
        "ess_b": res_b["dimension"],
        "n":     X.shape[0],
    }
    if curvature:
        # average the curvature estimate from both ESS variants
        result["curvature_mean"] = (res_a["curvature_mean"] + res_b["curvature_mean"]) / 2
    if ln is not None and lm_head is not None:
        result["entropy_mean"] = compute_entropy(X, ln, lm_head).item()
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_file")
    ap.add_argument("out_csv", help="Path to write results CSV.")
    ap.add_argument("--pos", type=int, nargs="+", default=None,
                    help="Token position(s) (0-indexed). One or more values. If omitted, all positions.")
    ap.add_argument("--depth", type=str, default=None,
                    help="Depth key, e.g. 'embed', 'layer_3/attn', 'final'. "
                         "If omitted, all depths.")
    ap.add_argument("--ess-k", type=int, default=100,
                    help="Neighbourhood size k for ESS (default 100).")
    ap.add_argument("--ess-d", type=int, default=1,
                    help="Simplex dimension d for ESS (default 1).")
    ap.add_argument("--curvature", action="store_true",
                    help="Add local PCA curvature column (averaged over ESS-a and ESS-b neighbourhoods).")
    ap.add_argument("--entropy", action="store_true",
                    help="Add entropy lens column. Loads the model from meta.attrs['model'].")
    ap.add_argument("--entropy-model", type=str, default=None,
                    help="Override model name for entropy lens (default: read from HDF5 meta).")
    args = ap.parse_args()

    with h5py.File(args.h5_file, "r") as f:
        seq_len   = int(f["meta"].attrs["seq_len"])
        n_samples = int(f["meta"].attrs["n_samples"])
        meta_attrs = dict(f["meta"].attrs)
        depths    = depth_keys(f)

        ln, lm_head = None, None
        if args.entropy:
            model_name = args.entropy_model or str(meta_attrs["model"])
            print(f"Loading entropy lens from {model_name!r} ...")
            ln, lm_head = load_entropy_lens(model_name)

        positions  = [p % seq_len for p in args.pos] if args.pos is not None else list(range(seq_len))
        sel_depths = [args.depth] if args.depth is not None else depths

        if args.depth is not None and args.depth not in depths:
            sys.exit(f"Unknown depth '{args.depth}'. Valid: {depths}")
        bad = [p for p in positions if not (0 <= p < seq_len)]
        if bad:
            sys.exit(f"Position(s) {bad} out of range [0, {seq_len}).")

        print(f"File      : {args.h5_file}")
        print(f"N samples : {n_samples}   seq_len : {seq_len}")
        print(f"Depths    : {len(sel_depths)}   Positions : {len(positions)}")
        print()
        header_fmt = f"{'depth':<22}  {'pos':>5}  {'TwoNN':>8}  {'ESS-a':>8}  {'ESS-b':>8}  {'N':>6}"
        if args.curvature:
            header_fmt += f"  {'Curvature':>10}"
        if args.entropy:
            header_fmt += f"  {'Entropy':>10}"
        print(header_fmt)
        extra_cols = (13 if args.curvature else 0) + (13 if args.entropy else 0)
        print("-" * (68 + extra_cols))

        out_path = resolve_out_path(args.out_csv)
        with open(out_path, "w", newline="") as csv_file:
            # metadata header — skipped by pandas with comment='#'
            csv_file.write(f"# run_timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
            csv_file.write(f"# h5_file: {args.h5_file}\n")
            csv_file.write(f"# pos: {args.pos if args.pos is not None else 'all'}\n")
            csv_file.write(f"# depth: {args.depth}\n")
            csv_file.write(f"# ess_k: {args.ess_k}\n")
            csv_file.write(f"# ess_d: {args.ess_d}\n")
            for k, v in meta_attrs.items():
                csv_file.write(f"# {k}: {v}\n")

            writer = csv.writer(csv_file)
            csv_cols = ["depth", "pos", "twonn", "ess_a", "ess_b", "n"]
            if args.curvature:
                csv_cols.append("curvature_mean")
            if args.entropy:
                csv_cols.append("entropy_mean")
            writer.writerow(csv_cols)

            for depth in sel_depths:
                for pos in positions:
                    X = load_latents(f, depth, pos)
                    res = compute_cell(X, ess_k=args.ess_k, ess_d=args.ess_d,
                                       ln=ln, lm_head=lm_head, curvature=args.curvature)
                    row_str = (
                        f"{depth:<22}  {pos:>5}  "
                        f"{res['twonn']:>8.3f}  {res['ess_a']:>8.3f}  {res['ess_b']:>8.3f}  {res['n']:>6}"
                    )
                    if args.curvature:
                        row_str += f"  {res['curvature_mean']:>10.4f}"
                    if args.entropy:
                        row_str += f"  {res['entropy_mean']:>10.4f}"
                    print(row_str)
                    csv_row = [depth, pos, res["twonn"], res["ess_a"], res["ess_b"], res["n"]]
                    if args.curvature:
                        csv_row.append(res["curvature_mean"])
                    if args.entropy:
                        csv_row.append(res["entropy_mean"])
                    writer.writerow(csv_row)
                    csv_file.flush()

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
