"""
ESS intrinsic dimension on the full embedding matrix of a causal LM,
sweeping k from --k-min to --k-max.

Processes the entire vocabulary in chunks to avoid OOM on the distance matrix.
Plots one histogram per k value in a single figure.

Usage:
  python scripts/embedding_ess.py --model gpt2 --k-min 20 --k-max 120 --k-step 20
"""

import argparse
import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hf_jacobian.id_estimators import _ess_values_batch, _ess_to_id


def load_embedding(model_name: str) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    return model.get_input_embeddings().weight.detach()


def ess_full(
    E: torch.Tensor,
    k: int,
    d: int = 1,
    ver: str = "a",
    n_groups: int = 5000,
    chunk: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    ESS ID for every row of E, computing kNN via chunked cdist to stay in memory.
    Returns (N,) array of per-token ID estimates.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    N, D = E.shape
    E_d = E.double()

    all_ids = np.empty(N, dtype=np.float64)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Q = E_d[start:end]  # (B, D)

        dists = torch.cdist(Q, E_d)  # (B, N)
        # exclude self
        for i in range(end - start):
            dists[i, start + i] = float("inf")

        _, knn_idx = dists.topk(k, dim=1, largest=False)  # (B, k)
        neighborhoods = E[knn_idx]  # (B, k, D)

        ess_vals = _ess_values_batch(neighborhoods, d=d, ver=ver, n_groups=n_groups, rng=rng)
        ess_np = ess_vals.cpu().numpy()
        ids = np.array([_ess_to_id(float(ev), d, ver) for ev in ess_np])
        all_ids[start:end] = ids

        print(f"  [{end:6d}/{N}]  chunk mean ID={np.nanmean(ids):.2f}", flush=True)

    return all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--k-min", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=120)
    parser.add_argument("--k-step", type=int, default=20)
    parser.add_argument("--ver", default="a", choices=["a", "b"])
    parser.add_argument("--n-groups", type=int, default=5000)
    parser.add_argument("--chunk", type=int, default=512, help="rows processed per cdist call")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"Loading model '{args.model}' ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    E = load_embedding(args.model)
    vocab_size, hidden_dim = E.shape
    print(f"Embedding matrix: {E.shape}  (vocab x hidden)")

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    results = {}  # k -> np.ndarray of IDs

    for k in k_values:
        print(f"\n=== k={k} ===")
        ids = ess_full(
            E, k=k, ver=args.ver,
            n_groups=args.n_groups,
            chunk=args.chunk,
            seed=args.seed,
        )
        results[k] = ids
        valid = ids[~np.isnan(ids)]
        print(f"  mean={np.mean(valid):.2f}  median={np.median(valid):.2f}  "
              f"std={np.std(valid):.2f}  min={np.min(valid):.2f}  max={np.max(valid):.2f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(4 * n_k, 4), sharey=True)
    if n_k == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        ids = results[k]
        valid = ids[~np.isnan(ids)]
        ax.hist(valid, bins=40, edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(valid), color="C1", linestyle="--",
                   label=f"mean={np.mean(valid):.1f}")
        ax.axvline(np.median(valid), color="C2", linestyle=":",
                   label=f"med={np.median(valid):.1f}")
        ax.set_title(f"k={k}")
        ax.set_xlabel("ESS ID")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("count")
    fig.suptitle(f"Embedding ESS ID — {args.model}  ({vocab_size} tokens)", y=1.02)
    fig.tight_layout()

    out_path = args.out or f"embedding_ess_{args.model.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHistogram saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
