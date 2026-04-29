# Code Style

Prefer functional programming over OOP, but use OOP when it genuinely fits (e.g. `nn.Module` subclasses for stateful weight containers, dispatch modes that must carry mutable state).

- Use plain functions by default. Collapse related logic into a single function rather than splitting across methods.
- Prefer data-in/data-out: functions take inputs, return outputs, no side effects unless necessary.
- Use `dict` / `dataclass` / `TypedDict` for structured data instead of classes with behavior.
- Keep each function minimal — if you can express it in one line, do so.
- Avoid inheritance, mixins, and deep object hierarchies.
- Use classes only when statefulness is the natural fit (PyTorch modules, dispatch modes, context managers). Keep them thin: hold state, delegate logic to functions.


Ignore all typing -- we do not care about the checker -- this is a research codebase.

# No Fallbacks

Never write fallback/generic paths for unsupported architectures. If a model class
is not explicitly registered, raise a clear error. This applies to `_sublayer_fn`,
hook setup, and any other architecture-specific dispatch. Silent fallbacks produce
wrong results without any indication of failure.

# Residual Stream Capture

Never read `inp[0]` inside a forward hook on an attn or ffn module to get the residual stream. Those modules receive a post-LayerNorm input (`LN(x)`), not `x`. Always use `TorchDispatchMode` to intercept `aten.add.Tensor` at the residual add site: register a forward hook on the sublayer to mark `id(sublayer_out)` as pending, then let the dispatch mode capture the result of the add (which is `x + g(LN(x))`). This is the pattern used by `_InSituJac` and `_CaptureAllMode`.