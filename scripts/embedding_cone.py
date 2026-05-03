"""
Sample unit vectors on the sphere that decode to a target token under the unembedding matrix.

Given the unembedding matrix U (vocab_size x hidden_dim), the cone
    {g : argmax(U @ g) == target}
is an open polyhedral cone in hidden space. We sample from it by running
Langevin dynamics on the sphere (soft, using log P(target|g) as energy) and
then hard-rejecting any sample where argmax != target.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hf_jacobian.id_estimators import twonn, ess

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_unembedding(model_name: str) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return model.lm_head.weight.detach()  # (vocab_size, hidden_dim)


def token_str(tokenizer, idx: int) -> str:
    return repr(tokenizer.convert_ids_to_tokens(idx))


def cone_centre(U: torch.Tensor, target: int) -> torch.Tensor:
    diff = U[target].unsqueeze(0) - U  # (vocab, hidden)
    diff = torch.cat([diff[:target], diff[target + 1:]], dim=0)
    return F.normalize(diff.sum(dim=0), dim=0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F

def optimize_cone_vectors(
    U: torch.Tensor, 
    target: int, 
    n_samples: int = 500, 
    steps: int = 100, 
    lr: float = 1.0, 
    device: str = "cpu"
) -> torch.Tensor:
    """
    Find unit vectors in the target cone using projected gradient descent.
    """
    U = U.to(device)
    hidden_dim = U.shape[1]

    # Initialize random vectors on the unit hypersphere
    G = F.normalize(torch.randn(n_samples, hidden_dim, device=device), dim=1)

    for _ in range(steps):
        G = G.detach().requires_grad_(True)
        logits = G @ U.T  # (n_samples, vocab_size)
        
        # Cross-entropy pushes the target logit up and all others down
        target_tensor = torch.full((n_samples,), target, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target_tensor)
        
        loss.backward()
        
        with torch.no_grad():
            # Gradient descent step (minimizing loss)
            G_next = G - lr * G.grad
            
            # Project back to the unit hypersphere
            G = F.normalize(G_next, dim=1)

    # Hard rejection to guarantee validity
    with torch.no_grad():
        in_cone = (G @ U.T).argmax(dim=1) == target
        # print(in_cone==True)
        accepted = G[in_cone].cpu()

    if len(accepted) < n_samples:
        print(f"Warning: Only {len(accepted)}/{n_samples} vectors converged inside the cone.")

    return accepted


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def report(U: torch.Tensor, g: torch.Tensor, target: int, tokenizer, label: str, top_k: int = 10):
    logits = U @ g
    probs = F.softmax(logits, dim=0)
    top = torch.topk(logits, top_k)
    margin = logits[target] - logits.topk(2).values[-1]

    print(f"\n{'=' * 60}")
    print(f"Method: {label}")
    print(f"  target token : {token_str(tokenizer, target)} (idx {target})")
    print(f"  logit[target]: {logits[target].item():.4f}")
    print(f"  prob[target] : {probs[target].item():.6f}")
    print(f"  margin       : {margin.item():.4f}  (logit gap to runner-up)")
    print(f"  top-{top_k} tokens:")
    for rank, (val, idx) in enumerate(zip(top.values, top.indices), 1):
        marker = " <-- TARGET" if idx.item() == target else ""
        print(f"    {rank:2d}. {token_str(tokenizer, idx.item()):30s}  logit={val.item():.4f}  p={probs[idx].item():.6f}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=5.0, help="Inverse temperature (higher = concentrate near mode)")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--collect-every", type=int, default=10, help="Steps between collection snapshots")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    print(f"Loading model '{args.model}' ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    U = load_unembedding(args.model)
    print(f"Unembedding matrix: {U.shape}  (vocab x hidden)")
    print(f"Target token: {token_str(tokenizer, args.target)} (idx {args.target})")

    
    gs = optimize_cone_vectors(
        U, args.target,
        n_samples=args.samples,
        device=args.device,
    )

    print(f"\nAccepted {len(gs)} cone vectors for ID estimation.")

    twonn_id = twonn(gs)

    print(f"\nIntrinsic dimension estimates on cone vectors:")
    print(f"  TwoNN : {twonn_id:.3f}")
    ess_result = ess(gs, 110)
    print(f"  ESS   : {ess_result['dimension']:.3f}  (per-point median: {float(__import__('numpy').nanmedian(ess_result['dimension_pw'])):.3f})")


if __name__ == "__main__":
    main()
