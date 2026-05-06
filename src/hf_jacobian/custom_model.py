import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .jacobian import jacobian_stats


@dataclass
class Config:
    d_model:    int = 1024
    n_heads:    int = 16
    n_layers:   int = 3
    vocab_size: int = 64
    mlp_expand: int = 4
    rope_base:  int = 10000


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt() * self.w


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # position_ids: (B, seq); inv_freq: (d_head/2,)
        # returns (cos, sin) each (B, 1, seq, d_head) — ready to broadcast over heads
        t = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)  # (B, seq, d_head/2)
        emb = torch.cat([t, t], dim=-1)                                      # (B, seq, d_head)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(q, k, cos, sin):
    # q, k: (B, n_heads, seq, d_head); cos, sin: (B, 1, seq, d_head)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, position_embeddings):
        B, T, D = x.shape
        cos, sin = position_embeddings
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))
        q, k = _apply_rope(q, k, cos, sin)
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

    def forward(self, x, position_embeddings):
        return x + self.self_attn(self.input_layernorm(x), position_embeddings)


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

    def forward(self, x, position_embeddings):
        x = self.attn(x, position_embeddings)
        x = self.ffn(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_head = cfg.d_model // cfg.n_heads
        self.embed       = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rotary_emb  = RotaryEmbedding(d_head, base=cfg.rope_base)
        self.layers      = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm        = RMSNorm(cfg.d_model)

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            x = self.embed(input_ids)
        B, T, _ = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        rope = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, rope)
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

    def _rope(x):
        B, T, _ = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return model.rotary_emb(x, position_ids)

    with torch.no_grad():
        x = inputs if inputs.is_floating_point() else model.embed(inputs)
        for i in range(layer_idx):
            x = model.layers[i](x, _rope(x))
        if sublayer == "ffn":
            x = layer.attn(x, _rope(x))

    x_leaf = x.detach().requires_grad_(True)
    out    = layer.attn(x_leaf, _rope(x_leaf)) if sublayer == "attn" else layer.ffn(x_leaf)
    # jac    = _block_jac_from_graph(out, x_leaf)
    # stats  = jacobian_stats(jac)
    # return x_leaf.detach().cpu(), {k: v.cpu() for k, v in stats.items()}
