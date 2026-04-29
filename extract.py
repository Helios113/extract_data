from hf_jacobian.jacobian import capture_sublayer, _causal_block_jac, jacobian_stats


def extract_target(
    model,
    inputs,
    layer_idx,
    sublayer,
    compute_jacobians: bool = False,
    compute_jacobian_stats: bool = False,
    jac_chunk: int = 64,
):
    """
    inputs: (B, seq) or (B, seq, d).
    Returns (hidden_out, jac, stats):
      hidden_out : (B, seq, d)       residual output h + g(h), always
      jac        : (B, seq, d, d)    raw Jacobian if compute_jacobians, else None
      stats      : dict | None       det/sigma_ratio if compute_jacobian_stats, else None

    compute_jacobian_stats requires compute_jacobians=True.
    """
    cpu = lambda t: t.cpu()
    if not compute_jacobians:
        _, hidden_out = capture_sublayer(model, inputs, layer_idx, sublayer)
        return cpu(hidden_out), None, None
    _, hidden_out, jac = _causal_block_jac(model, inputs, layer_idx, sublayer, jac_chunk)
    stats = jacobian_stats(jac) if compute_jacobian_stats else None
    if stats is not None:
        stats = {k: cpu(v) for k, v in stats.items()}
    return cpu(hidden_out), cpu(jac), stats

