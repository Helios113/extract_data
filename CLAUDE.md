# Code Style

Prefer functional programming over OOP, but use OOP when it genuinely fits (e.g. `nn.Module` subclasses for stateful weight containers, dispatch modes that must carry mutable state).

- Use plain functions by default. Collapse related logic into a single function rather than splitting across methods.
- Prefer data-in/data-out: functions take inputs, return outputs, no side effects unless necessary.
- Use `dict` / `dataclass` / `TypedDict` for structured data instead of classes with behavior.
- Keep each function minimal — if you can express it in one line, do so.
- Avoid inheritance, mixins, and deep object hierarchies.
- Use classes only when statefulness is the natural fit (PyTorch modules, dispatch modes, context managers). Keep them thin: hold state, delegate logic to functions.
