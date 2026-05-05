"""
Load an HDF5 file produced by run.py and compute, for every (token_position, depth):
  - TwoNN intrinsic dimension (one global estimate per cell)
  - ESS-a and ESS-b per-point ID (one value per sample)
  - LocalPCA curvature per-point (one value per sample, uses ESS-a ID as tangent dim)
  - Shannon entropy per-point (one value per sample, via final LN + LM head)

Each output CSV row represents one (depth, pos, sample) triple.

k-sweep: pass --ess-k as a comma-separated list (e.g. --ess-k 50,100,200) to run
ESS and LocalPCA for each k independently.  TwoNN and entropy are k-independent and
only computed once per cell.

Usage:
  python analyze_id.py <h5_file> <out_csv>
  python analyze_id.py out/run.h5 results/out.csv --ess-k 50,100,200 --entropy
  python analyze_id.py out/run.h5 results/out.csv --pos 10 --depth layer_2/ffn
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
from hf_jacobian.id_estimators import (
    _ess_values_batch,
    _ess_to_id,
    _local_pca_curvature,
    twonn,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def depth_keys(f: h5py.File) -> list[str]:
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
    """Return (N, d_model) float32 tensor for (depth, pos) across all samples."""
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
    causal = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    ).eval()
    arch = type(causal.base_model).__name__
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
    return _LN_ATTR[arch](causal), causal.lm_head


def compute_entropy_per_sample(X: torch.Tensor, ln, lm_head) -> np.ndarray:
    """Shannon entropy for each of the N hidden vectors. Returns (N,) float64."""
    with torch.no_grad():
        logits = lm_head(ln(X.float()))          # (N, vocab)
        probs  = logits.softmax(dim=-1)
        h      = -(probs * (probs + 1e-15).log()).sum(dim=-1)   # (N,)
    return h.cpu().numpy().astype(np.float64)


def _knn_neighborhoods(X: torch.Tensor, k: int):
    """Return (N, k, D) neighborhoods and (N,) 1st-NN distances."""
    X_d = X.double()
    dists = torch.cdist(X_d, X_d)
    dists.fill_diagonal_(float("inf"))
    nn_dists, knn_idx = dists.topk(k, dim=1, largest=False)  # (N, k)
    return X[knn_idx], nn_dists[:, 0]


def compute_ess_and_pca(X: torch.Tensor, k: int, d: int = 1) -> dict:
    """
    ESS-a, ESS-b per-point IDs and LocalPCA curvature for a single k.
    Returns dict with keys: ess_a_pw, ess_b_pw, pca_curv_pw (all np.ndarray (N,)).
    """
    neighborhoods, _ = _knn_neighborhoods(X, k)

    # ESS-a
    ess_a_vals = _ess_values_batch(neighborhoods, d=d, ver="a").cpu().numpy()
    ess_a_ids  = np.array([_ess_to_id(float(v), d, "a") for v in ess_a_vals])

    # ESS-b (d=1 only)
    ess_b_vals = _ess_values_batch(neighborhoods, d=1, ver="b").cpu().numpy()
    ess_b_ids  = np.array([_ess_to_id(float(v), 1, "b") for v in ess_b_vals])

    # LocalPCA curvature — use nanmean of ESS-a as tangent dimension
    id_dim = max(1, round(float(np.nanmean(ess_a_ids))))
    pca_curv = _local_pca_curvature(neighborhoods, id_dim)

    return {"ess_a_pw": ess_a_ids, "ess_b_pw": ess_b_ids, "pca_curv_pw": pca_curv}


# ── CSV columns ───────────────────────────────────────────────────────────────

def build_header(ks: list[int], with_entropy: bool) -> list[str]:
    base = ["depth", "pos", "sample", "twonn"]
    for k in ks:
        base += [f"ess_a_k{k}", f"ess_b_k{k}", f"pca_curv_k{k}"]
    if with_entropy:
        base.append("entropy")
    return base


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("h5_file")
    ap.add_argument("out_csv", help="Path for output CSV.")
    ap.add_argument("--pos", type=int, nargs="+", default=None,
                    help="Token position(s) to process (0-indexed). Default: all.")
    ap.add_argument("--depth", type=str, default=None,
                    help="Single depth key, e.g. 'layer_3/attn'. Default: all.")
    ap.add_argument("--ess-k", type=str, default="100",
                    help="Comma-separated k values for ESS/LocalPCA (default: 100).")
    ap.add_argument("--ess-d", type=int, default=1,
                    help="Simplex dimension d for ESS-a (default 1).")
    ap.add_argument("--no-entropy", action="store_true",
                    help="Skip per-sample entropy computation via LM head.")
    ap.add_argument("--entropy-model", type=str, default=None,
                    help="Override model name for entropy lens.")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.ess_k.split(",")]

    with h5py.File(args.h5_file, "r") as f:
        seq_len   = int(f["meta"].attrs["seq_len"])
        n_samples = int(f["meta"].attrs["n_samples"])
        meta_attrs = dict(f["meta"].attrs)
        depths    = depth_keys(f)

        ln, lm_head = None, None
        if not args.no_entropy:
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

        header = build_header(ks, not args.no_entropy)
        n_cells = len(sel_depths) * len(positions)

        print(f"File       : {args.h5_file}")
        print(f"N samples  : {n_samples}   seq_len : {seq_len}")
        print(f"Depths     : {len(sel_depths)}   Positions : {len(positions)}")
        print(f"k-sweep    : {ks}")
        print(f"Cells      : {n_cells}")
        print()

        out_path = resolve_out_path(args.out_csv)
        with open(out_path, "w", newline="") as csv_file:
            csv_file.write(f"# run_timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
            csv_file.write(f"# h5_file: {args.h5_file}\n")
            csv_file.write(f"# ess_k: {ks}\n")
            csv_file.write(f"# ess_d: {args.ess_d}\n")
            for key, val in meta_attrs.items():
                csv_file.write(f"# {key}: {val}\n")

            writer = csv.writer(csv_file)
            writer.writerow(header)

            cell_idx = 0
            for depth in sel_depths:
                for pos in positions:
                    cell_idx += 1
                    print(f"  [{cell_idx}/{n_cells}] {depth}  pos={pos} ...", end="", flush=True)

                    X = load_latents(f, depth, pos)   # (N, d_model)
                    N = X.shape[0]

                    # TwoNN — one scalar for the whole cell
                    id_twonn = twonn(X)

                    # ESS + LocalPCA for each k — returns per-sample arrays
                    per_k = {k: compute_ess_and_pca(X, k, d=args.ess_d) for k in ks}

                    # Entropy per sample
                    entropy_pw = None
                    if ln is not None:
                        entropy_pw = compute_entropy_per_sample(X, ln, lm_head)

                    # Write one row per sample
                    for i in range(N):
                        row = [depth, pos, i, id_twonn]
                        for k in ks:
                            row += [
                                per_k[k]["ess_a_pw"][i],
                                per_k[k]["ess_b_pw"][i],
                                per_k[k]["pca_curv_pw"][i],
                            ]
                        if entropy_pw is not None:
                            row.append(entropy_pw[i])
                        writer.writerow(row)

                    csv_file.flush()

                    # quick summary for stdout
                    k0 = ks[0]
                    ess_a_mean = float(np.nanmean(per_k[k0]["ess_a_pw"]))
                    ess_b_mean = float(np.nanmean(per_k[k0]["ess_b_pw"]))
                    print(f"  twonn={id_twonn:.2f}  ess_a={ess_a_mean:.2f}  ess_b={ess_b_mean:.2f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
