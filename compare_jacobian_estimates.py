"""
Compare JVP/VJP residual estimates against full Jacobian stats on a tiny input.

Usage:
  python compare_jacobian_estimates.py config.json --layer 0 --sublayer attn
"""

import argparse
import json

import torch

import hf_jacobian as hj
from estimate_residual_stats import estimate_residual_layer_stats, _make_residual_func
from hf_jacobian.jacobian import _causal_block_jac, _layers


def parse_args():
    p = argparse.ArgumentParser(description="Compare residual estimates vs full Jacobian")
    p.add_argument("config", help="JSON config file")
    p.add_argument("--text", default=None, help="Text to tokenize (default: config or built-in)")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--sublayer", choices=("attn", "ffn"), default="attn")
    p.add_argument("--num_power_iters", type=int, default=15)
    p.add_argument("--num_trace_samples", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    device = cfg.get("device", "cpu")
    text = args.text or cfg.get("text") or (
        "The transformer architecture has become the dominant paradigm in natural "
        "language processing, enabling models to capture long-range dependencies "
        "through self-attention mechanisms."
    )

    print(f"Loading {model_name!r} on {device} ...")
    model, tok = hj.load(model_name, device=device)
    from estimate_residual_stats import _configure_attention
    _configure_attention(model)

    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device).to(dtype=torch.bool)

    input_ids = input_ids[:, :1]
    if attention_mask is not None:
        attention_mask = attention_mask[:, :1]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    layers = _layers(model)

    layer = layers[args.layer]
    x_in = hidden_states[args.layer]
    f = _make_residual_func(layer, args.sublayer, attention_mask)

    print("Estimating (JVP/VJP) ...")
    est = estimate_residual_layer_stats(
        f,
        x_in,
        num_power_iters=args.num_power_iters,
        num_trace_samples=args.num_trace_samples,
    )

    print("Computing full Jacobian ...")
    hidden, jac = _causal_block_jac(model, input_ids, args.layer, args.sublayer)

    J = jac[0, 0]
    sign, logabsdet = torch.linalg.slogdet(J)
    sv = torch.linalg.svdvals(J)
    sigma_max = sv[0].item()
    sigma_min = sv[-1].item()

    print("\n[estimate]")
    for k, v in est.items():
        print(f"  {k}: {v:.6f}")

    print("\n[full_jacobian]")
    print(f"  log|det(J)|: {logabsdet.item():.6f}")
    print(f"  sigma_max(J): {sigma_max:.6f}")
    print(f"  sigma_min(J): {sigma_min:.6f}")
    print(f"  sign(det): {int(sign.item())}")


if __name__ == "__main__":
    main()
