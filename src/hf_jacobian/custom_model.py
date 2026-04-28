import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .jacobian import _block_jac_from_graph, jacobian_stats


@dataclass
class Config:
    d_model:    int = 64
    n_heads:    int = 4
    n_layers:   int = 2
    vocab_size: int = 256
    mlp_expand: int = 4


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt() * self.w


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        w = (q @ k.transpose(-2, -1) / self.d_head ** 0.5).masked_fill(mask, -1e9).softmax(-1)
        return self.o((w @ v).transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.d_model * cfg.mlp_expand
        self.gate = nn.Linear(cfg.d_model, h, bias=False)
        self.up   = nn.Linear(cfg.d_model, h, bias=False)
        self.down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AttnResidual(nn.Module):
    """input_layernorm → self_attn → residual add, as a single differentiable unit."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model)
        self.self_attn       = Attention(cfg)

    def forward(self, x):
        return x + self.self_attn(self.input_layernorm(x))


class FFNResidual(nn.Module):
    """post_attention_layernorm → mlp → residual add, as a single differentiable unit."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.post_attention_layernorm = RMSNorm(cfg.d_model)
        self.mlp                      = MLP(cfg)

    def forward(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))


class Block(nn.Module):
    """Llama-style pre-norm block.

    Parameters live in AttnResidual / FFNResidual.  Flat properties expose the
    HF-compatible names (input_layernorm, self_attn, …) so jacobian.py's _sub()
    lookup works without any changes to the hook machinery.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = AttnResidual(cfg)
        self.ffn  = FFNResidual(cfg)

    @property
    def input_layernorm(self):          return self.attn.input_layernorm
    @property
    def self_attn(self):                return self.attn.self_attn
    @property
    def post_attention_layernorm(self): return self.ffn.post_attention_layernorm
    @property
    def mlp(self):                      return self.ffn.mlp

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)

    def forward(self, input_ids=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def extract_direct(
    model: CustomModel,
    inputs: torch.Tensor,
    layer_idx: int,
    sublayer: str,
):
    """
    Extract hidden state and Jacobian stats by calling AttnResidual / FFNResidual
    directly on a leaf — no hooks, no dispatch.

    inputs: int token ids (1, seq) or float latent representations (1, seq, d).
    Returns (hidden (seq, d), stats dict).
    """
    layer = model.layers[layer_idx]

    with torch.no_grad():
        x = inputs if inputs.is_floating_point() else model.embed(inputs)
        for i in range(layer_idx):
            x = model.layers[i](x)
        if sublayer == "ffn":
            x = layer.attn(x)

    x_leaf = x.detach().requires_grad_(True)
    out    = layer.attn(x_leaf) if sublayer == "attn" else layer.ffn(x_leaf)
    jac    = _block_jac_from_graph(out, x_leaf)
    stats  = jacobian_stats(jac)
    return x_leaf.detach().cpu()[0], {k: v.cpu() for k, v in stats.items()}
