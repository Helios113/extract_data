# hf-jacobian

Compute per-token Jacobians for transformer residual branches using Hugging Face models.

Supported Jacobian targets per layer:
- `attention`: $J_t = \frac{\partial (x_t + \mathrm{Attn}(\mathrm{LN}(x))_t)}{\partial x_t}$
- `mlp`: $J_t = \frac{\partial (x'_t + \mathrm{MLP}(\mathrm{LN}(x'))_t)}{\partial x_t}$ where $x' = x + \mathrm{Attn}(\mathrm{LN}(x))$

The script computes this **token by token** for each selected layer and saves a tensor of shape:

- `[layers, seq_len, d_model, d_model]`

## Setup

This project is managed by `uv`.

```bash
uv sync
```

## Run

```bash
uv run hf-jacobian \
	--model gpt2 \
	--text "The quick brown fox jumps over the lazy dog." \
	--component attention \
	--max-tokens 8 \
	--max-layers 2 \
	--save jacobians.pt \
	--metadata jacobians.json
```

Switch to feedforward branch:

```bash
uv run hf-jacobian --component mlp
```

## Outputs

- `jacobians.pt`: serialized dict containing Jacobians and token information.
- `jacobians.json`: run metadata and output shape.

## Notes

- Jacobians are expensive: runtime and memory scale as $O(L \cdot T \cdot d^2)$.
- Start small (`--max-layers`, `--max-tokens`) and scale up.
- The script supports common GPT-style (`transformer.h`) and Llama-style (`model.layers`) block layouts.




Left to do:

Add TwoNN and ESS in torch mode


add samples from the hypersurfaces that we have


Accelerate everything
