"""
Estimate Jacobian statistics for residual sublayers using JVP/VJP.

Usage:
  python estimate_residual_stats.py config.json --text "The quick brown fox"
  python estimate_residual_stats.py config.json --layer 3 --sublayer attn
  python estimate_residual_stats.py config.json --output stats.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.func import jvp, vjp

import hf_jacobian as hj
from hf_jacobian.jacobian import _layers, _sub, SUBLAYERS


def estimate_residual_layer_stats(
    f,
    x: torch.Tensor,
    num_power_iters: int = 15,
    num_trace_samples: int = 10,
) -> dict[str, float]:
    trace_est = 0.0
    for _ in range(num_trace_samples):
        z = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1
        _, jvp_out = jvp(f, (x,), (z,))
        trace_est += torch.sum(z * jvp_out)

    trace_est = trace_est / num_trace_samples

    v = torch.randn_like(x)
    v = v / torch.norm(v)

    for _ in range(num_power_iters):
        _, u = jvp(f, (x,), (v,))
        _, vjp_fn = vjp(f, x)
        v_tilde = vjp_fn(u)[0]
        v = v_tilde / torch.norm(v_tilde)

    _, u = jvp(f, (x,), (v,))
    sigma_max_f = torch.norm(u) / torch.norm(v)
    sigma_min_j = torch.clamp(1.0 - sigma_max_f, min=0.0)

    return {
        "approx_log_det_J": float(trace_est.item()),
        "sigma_max_Jf": float(sigma_max_f.item()),
        "bound_sigma_max_J": float(1.0 + sigma_max_f.item()),
        "bound_sigma_min_J": float(1.0 - sigma_max_f.item()),
        "sigma_min_J": float(sigma_min_j.item()),
    }


def _first_tensor_out(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    raise ValueError("Sublayer output did not contain a tensor")


def _configure_attention(model) -> None:
    if hasattr(model, "config"):
        try:
            model.config.attn_implementation = "eager"
        except Exception:
            pass
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdpa"):
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def _make_residual_func(layer, sublayer: str, attention_mask: torch.Tensor | None):
    ln_candidates, sub_candidates = SUBLAYERS[sublayer]
    _, ln = _sub(layer, *ln_candidates)
    _, sub = _sub(layer, *sub_candidates)

    def f(x: torch.Tensor) -> torch.Tensor:
        h = ln(x)
        if sublayer == "attn":
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(
                    enable_math=True,
                    enable_flash=False,
                    enable_mem_efficient=False,
                ):
                    out = sub(
                        h,
                        attention_mask=attention_mask,
                        head_mask=None,
                        attn_implementation="eager",
                        use_cache=False,
                        output_attentions=False,
                    )
            else:
                out = sub(
                    h,
                    attention_mask=attention_mask,
                    head_mask=None,
                    attn_implementation="eager",
                    use_cache=False,
                    output_attentions=False,
                )
            return _first_tensor_out(out)
        return sub(h)

    return f


def parse_args():
    p = argparse.ArgumentParser(description="Estimate residual Jacobian stats via JVP/VJP")
    p.add_argument("config", help="JSON config file")
    p.add_argument("--text", default=None, help="Text to tokenize (default: config or built-in)")
    p.add_argument("--output", default=None, help="Optional JSON output path")
    p.add_argument("--layer", type=int, default=None, help="Layer index to run (default: all)")
    p.add_argument("--sublayer", choices=("attn", "ffn"), default=None)
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
    _configure_attention(model)

    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device).to(dtype=torch.bool)

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

    layer_indices = [args.layer] if args.layer is not None else list(range(len(layers)))
    sublayers = [args.sublayer] if args.sublayer is not None else ["attn", "ffn"]

    results: dict[str, dict[str, float]] = {}
    for layer_idx in layer_indices:
        x_in = hidden_states[layer_idx]
        layer = layers[layer_idx]
        for sublayer in sublayers:
            f = _make_residual_func(layer, sublayer, attention_mask)
            stats = estimate_residual_layer_stats(
                f,
                x_in,
                num_power_iters=args.num_power_iters,
                num_trace_samples=args.num_trace_samples,
            )
            key = f"layer_{layer_idx}/{sublayer}"
            results[key] = stats
            print(f"[{key}]")
            for k, v in stats.items():
                print(f"  {k}: {v:.6f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {str(out_path)!r}")


if __name__ == "__main__":
    main()
