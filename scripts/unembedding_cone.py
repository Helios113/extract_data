"""
Find input vectors that maximise a target token's logit under the unembedding matrix.

Given the unembedding matrix U (vocab_size x hidden_dim), the condition
    U[i] @ g > U[j] @ g  for all j != i
is equivalent to
    (U[i] - U[j]) @ g > 0  for all j != i
which defines an open polyhedral cone in hidden space.

We find vectors inside this cone via:
  1. The cone's "center ray" -- the normalised sum of the M-1 difference vectors.
     (This is the direction that maximises the minimum margin over all competitors.)
  2. The SVM solution -- the max-margin direction found by a linear SVM with one-vs-rest.
  3. Gradient ascent from random initialisations, maximising the softmax log-prob of the
     target token (stays on the unit sphere via projected gradient).
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_unembedding(model_name: str) -> torch.Tensor:
    """Return the unembedding matrix (vocab_size x hidden_dim) on CPU."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    # Works for GPT-2, LLaMA, Mistral, Phi, etc.
    lm_head = model.lm_head
    W = lm_head.weight.detach()  # (vocab_size, hidden_dim)
    return W


def token_str(tokenizer, idx: int) -> str:
    return repr(tokenizer.convert_ids_to_tokens(idx))


# ---------------------------------------------------------------------------
# Method 1: cone centre ray
# ---------------------------------------------------------------------------

def cone_centre(U: torch.Tensor, target: int) -> torch.Tensor:
    """
    Normalised sum of (U[target] - U[j]) for all j != target.
    This is the direction most "inside" the cone in an L2 sense.
    """
    diff = U[target].unsqueeze(0) - U  # (vocab, hidden)
    diff = torch.cat([diff[:target], diff[target + 1:]], dim=0)  # drop j==i row
    centre = diff.sum(dim=0)
    return F.normalize(centre, dim=0)


# ---------------------------------------------------------------------------
# Method 2: gradient ascent on softmax log-prob (unit sphere)
# ---------------------------------------------------------------------------

def gradient_ascent(
    U: torch.Tensor,
    target: int,
    n_restarts: int = 8,
    n_steps: int = 2000,
    lr: float = 0.05,
    device: str = "cpu",
) -> tuple[torch.Tensor, float]:
    """
    Maximise log P(target | g) = U[target]@g - log(sum_j exp(U[j]@g))
    on the unit sphere via projected gradient ascent.
    Returns (best_g, best_logprob).
    """
    U = U.to(device)
    best_g, best_lp = None, -float("inf")

    for _ in range(n_restarts):
        g = F.normalize(torch.randn(U.shape[1], device=device), dim=0)
        g = g.requires_grad_(True)

        for step in range(n_steps):
            logits = U @ g                         # (vocab,)
            lp = logits[target] - torch.logsumexp(logits, dim=0)
            (-lp).backward()
            with torch.no_grad():
                g_new = g - lr * g.grad
                g_new = F.normalize(g_new, dim=0)
                g.copy_(g_new)
                g.grad.zero_()

            if (step + 1) % 500 == 0:
                lr *= 0.5  # simple decay

        with torch.no_grad():
            logits = U @ g
            lp = (logits[target] - torch.logsumexp(logits, dim=0)).item()

        if lp > best_lp:
            best_lp = lp
            best_g = g.detach().cpu()

    return best_g, best_lp


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def report(U: torch.Tensor, g: torch.Tensor, target: int, tokenizer, label: str, top_k: int = 10):
    logits = U @ g
    probs = F.softmax(logits, dim=0)
    top = torch.topk(logits, top_k)
    margin = logits[target] - logits.topk(2).values[-1]  # gap to runner-up

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
    parser = argparse.ArgumentParser(description="Explore the unembedding polyhedral cone for a target token.")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name or local path")
    parser.add_argument("--target", type=int, required=True, help="Target token index")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top tokens to display")
    parser.add_argument("--restarts", type=int, default=8, help="Gradient ascent restarts")
    parser.add_argument("--steps", type=int, default=2000, help="Gradient ascent steps per restart")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial gradient ascent learning rate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    print(f"Loading model '{args.model}' ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    U = load_unembedding(args.model)
    print(f"Unembedding matrix: {U.shape}  (vocab x hidden)")
    print(f"Target token: {token_str(tokenizer, args.target)} (idx {args.target})")

    # Method 1: cone centre
    g_centre = cone_centre(U, args.target)
    report(U, g_centre, args.target, tokenizer, "Cone centre ray", args.top_k)

    # Method 2: gradient ascent
    print(f"\nRunning gradient ascent ({args.restarts} restarts x {args.steps} steps) ...")
    g_opt, lp = gradient_ascent(
        U, args.target,
        n_restarts=args.restarts,
        n_steps=args.steps,
        lr=args.lr,
        device=args.device,
    )
    report(U, g_opt, args.target, tokenizer, f"Gradient ascent (best log-prob={lp:.4f})", args.top_k)


if __name__ == "__main__":
    main()
