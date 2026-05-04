"""
Unified extraction script: Jacobian stats + latent representations.

One config file sets the model, sequence length, and data source. The script
runs extraction for all (layer, sublayer) pairs and saves everything to HDF5.

Config JSON format:
  {
    "model":   "gpt2",
    "device":  "cpu",           (optional, default "cpu")
    "weights": "real",          (optional, "real" | "random"; default "real")
    "output":  "out/run.h5",    (optional; can be overridden via CLI positional arg)
    "sampling": {
      "n_samples":  32,
      "seq_len":    64,
      "batch_size": 4           (optional, default 1)
    },
    "source": { "type": "dataset" | "manifold" | "benchmark", ... }
  }

Source "dataset" — HuggingFace streaming dataset:
  { "type": "dataset", "name": "wikitext", "config": "wikitext-2-raw-v1",
    "split": "train", "text_column": "text" }

Source "manifold" — Monge patch hypersurface f(x) = Σ λᵢ xᵢ²:
  { "type": "manifold", "manifold_dim": 5, "ambient_dim": 768,
    "patch_radius": 0.4, "noise_std": 0.0, "seed": 0,
    "lambdas": [1.0, 0.5, 0.2, 0.1, 0.0],          // explicit, or:
    "lambda_params": {"entropy": 0.8, "lambda_min": 0.0, "lambda_max": 1.0,
                      "isotropic": false, "same_sign": true},
    "project_dim": null }
  ambient_dim must equal d_model, or set project_dim for a random projection.

Source "benchmark" — skdim BenchmarkManifolds:
  { "type": "benchmark", "name": "M7_Roll", "seed": 42 }
  Points are projected to d_model via a fixed random orthonormal matrix.
  True intrinsic dimension is stored in /meta.

HDF5 layout:
  /meta attrs: model, source_type, n_samples, seq_len, batch_size,
               [source-specific attrs]
  /samples/{i}/input_ids         — (seq_len,) int64          # dataset only
  /samples/{i}/manifold_points   — (seq_len, ambient_dim)    # manifold / benchmark
  /samples/{i}/layer_{j}/{sub}/hidden_state  — (seq_len, d)
  /samples/{i}/layer_{j}/{sub}/log_det       — (seq_len,)
  /samples/{i}/layer_{j}/{sub}/sigma_ratio   — (seq_len,)

Usage:
  python run.py config.json [output.h5] [--device DEVICE]
"""

import argparse
import json
from pathlib import Path
from typing import Iterator
import tqdm

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import hf_jacobian as hj
from extract_dataset import chunk_dataset
from hf_jacobian.jacobian import _layers, _sublayer_fn, capture_all_hidden, _causal_block_jac, jacobian_stats
from hf_jacobian.manifold_dataset import ManifoldConfig, ManifoldDataset, _ortho_frame
from hf_jacobian.invertibility import check_invertibility

_SUBLAYERS = ("attn", "ffn")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified Jacobian + latent extraction")
    p.add_argument("config", help="JSON config file")
    p.add_argument("output", nargs="?", default=None,
                   help="Output HDF5 path (overrides config 'output' key)")
    p.add_argument("--device", default=None, help="Override device from config")
    p.add_argument("--no-upload", action="store_true", help="Skip upload after run")
    return p.parse_args()


# ─── source iterators ────────────────────────────────────────────────────────

def iter_dataset_batches(
    src: dict, tok, n_samples: int, seq_len: int, batch_size: int, device: str,
    skip: int = 0,
) -> Iterator[tuple[int, torch.Tensor, None]]:
    from datasets import load_dataset
    ds = load_dataset(
        src["name"],
        src.get("config", None),
        split=src.get("split", "train"),
        streaming=True,
    )
    if "seed" in src:
        ds = ds.shuffle(seed=src["seed"], buffer_size=10_000)
    chunks = chunk_dataset(tok, ds, src.get("text_column", "text"), seq_len, n_samples)

    for start in range(skip, n_samples, batch_size):
        batch_ids = chunks[start : min(start + batch_size, n_samples)]
        ids_t     = torch.tensor(batch_ids, dtype=torch.long).to(device)
        yield start, ids_t, None


def iter_random_token_batches(
    src: dict, model, n_samples: int, seq_len: int, batch_size: int, device: str,
    skip: int = 0,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    seed      = src.get("seed", 0)
    vocab_size = src.get("vocab_size", None)

    emb_weight = model.get_input_embeddings().weight  # (V, d_model)
    if vocab_size is None:
        vocab_size = emb_weight.shape[0]

    gen = torch.Generator()
    gen.manual_seed(seed)

    total  = n_samples * seq_len
    ids    = torch.randint(0, vocab_size, (total,), generator=gen)          # (total,)
    embeds = emb_weight[ids].detach().reshape(n_samples, seq_len, -1)       # (n, seq, d)
    raw_ids = ids.reshape(n_samples, seq_len)                                # (n, seq) int

    for start in range(skip, n_samples, batch_size):
        batch  = embeds[start : min(start + batch_size, n_samples)].to(device)
        raw    = raw_ids[start : min(start + batch_size, n_samples)]
        yield start, batch, raw


def iter_manifold_batches(
    src: dict, n_samples: int, seq_len: int, batch_size: int, d_model: int, device: str,
    skip: int = 0,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    ambient_dim = src.get("ambient_dim", src["manifold_dim"])
    cfg = ManifoldConfig(
        manifold_dim          = src["manifold_dim"],
        ambient_dim           = ambient_dim,
        n_samples             = n_samples,
        seq_len               = seq_len,
        noise_std             = src.get("noise_std", 0.0),
        seed                  = src.get("seed", 0),
        lambdas               = src.get("lambdas", None),
        lambda_params         = src.get("lambda_params", None),
        patch_radius          = src.get("patch_radius", 1.0),
    )
    dataset = ManifoldDataset(cfg, project_dim=src.get("project_dim", d_model))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Generating {n_samples} × {seq_len} monge_patch "
          f"(d={cfg.manifold_dim}, D={cfg.ambient_dim}) samples ...\n")

    sample_idx = 0
    for batch in loader:
        if sample_idx < skip:
            sample_idx += batch.shape[0]
            continue
        yield sample_idx, batch.to(device), batch.cpu()
        sample_idx += batch.shape[0]


def iter_benchmark_batches(
    src: dict, n_samples: int, seq_len: int, batch_size: int, d_model: int, device: str,
    skip: int = 0,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    import skdim
    name = src["name"]
    seed = src.get("seed", 42)

    bm   = skdim.datasets.BenchmarkManifolds(random_state=seed)
    data = bm.generate(n=n_samples * seq_len)
    X_np = data[name].astype(np.float32)  # (N, D)
    D    = X_np.shape[1]

    gen = torch.Generator()
    gen.manual_seed(seed)
    if D >= d_model:
        P = _ortho_frame(d_model, D, gen)    # (D, d_model) — orthonormal cols in R^D
    else:
        P = _ortho_frame(D, d_model, gen).T  # (D, d_model) — orthonormal rows in R^d_model

    X_t    = torch.from_numpy(X_np) @ P                          # (N, d_model)
    seqs   = X_t.reshape(n_samples, seq_len, d_model)            # (n_samples, seq_len, d_model)
    raw    = torch.from_numpy(X_np).reshape(n_samples, seq_len, D)

    print(f"BenchmarkManifold {name!r}: D={D} → projected to d_model={d_model}\n")

    for start in range(skip, n_samples, batch_size):
        batch     = seqs[start : min(start + batch_size, n_samples)].to(device)
        raw_batch = raw[start : min(start + batch_size, n_samples)]
        yield start, batch, raw_batch


# ─── HDF5 helpers ────────────────────────────────────────────────────────────

def write_meta(
    f, model_name: str, src_type: str, src: dict,
    n_samples: int, seq_len: int, batch_size: int, extra: dict,
    weights: str, d_model: int, n_layers: int,
    compute_jacobians: bool, compute_jacobian_stats: bool,
    compute_approx_sigma: bool = False, approx_sigma_probes: int = 64,
):
    meta = f.create_group("meta")
    meta.attrs["model"]                  = model_name
    meta.attrs["weights"]                = weights
    meta.attrs["d_model"]                = d_model
    meta.attrs["n_layers"]               = n_layers
    meta.attrs["source_type"]            = src_type
    meta.attrs["n_samples"]              = n_samples
    meta.attrs["seq_len"]                = seq_len
    meta.attrs["batch_size"]             = batch_size
    meta.attrs["compute_jacobians"]      = compute_jacobians
    meta.attrs["compute_jacobian_stats"] = compute_jacobian_stats
    meta.attrs["compute_approx_sigma"]   = compute_approx_sigma
    meta.attrs["approx_sigma_probes"]    = approx_sigma_probes

    if src_type == "dataset":
        meta.attrs["dataset_name"]   = src["name"]
        meta.attrs["dataset_config"] = src.get("config", "")
        meta.attrs["split"]          = src.get("split", "train")
        meta.attrs["text_column"]    = src.get("text_column", "text")
        meta.attrs["tokenizer"]      = model_name
    elif src_type == "manifold":
        meta.attrs["manifold_dim"]           = src["manifold_dim"]
        meta.attrs["ambient_dim"]            = src.get("ambient_dim", src["manifold_dim"])
        meta.attrs["noise_std"]              = src.get("noise_std", 0.0)
        meta.attrs["seed"]                   = src.get("seed", 0)
        meta.attrs["project_dim"]            = src.get("project_dim", d_model)
        meta.attrs["patch_radius"]           = src.get("patch_radius", 1.0)
        lambdas = src.get("lambdas")
        meta.attrs["lambdas"] = lambdas if lambdas is not None else []
        lp = src.get("lambda_params")
        if lp is not None:
            import json as _json
            meta.attrs["lambda_params"] = _json.dumps(lp)
    elif src_type == "random_tokens":
        meta.attrs["seed"]       = src.get("seed", 0)
        meta.attrs["vocab_size"] = src.get("vocab_size", 0)   # 0 = full model vocab
    elif src_type == "benchmark":
        for k, v in extra.items():
            meta.attrs[k] = v



def _ds_append(f, name: str, batch_np):
    """Append a (B, ...) batch to a resizable dataset, creating it on first call."""
    if name in f:
        ds = f[name]
        old = ds.shape[0]
        ds.resize(old + batch_np.shape[0], axis=0)
        ds[old:] = batch_np
    else:
        maxshape = (None,) + batch_np.shape[1:]
        f.create_dataset(name, data=batch_np, maxshape=maxshape,
                         compression="lzf", shuffle=True, chunks=(1,) + batch_np.shape[1:])




# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_name = cfg["model"]
    device     = args.device or cfg.get("device", "cpu")
    output     = args.output or cfg.get("output")
    if output is None:
        raise ValueError("Output path required: pass as positional arg or set 'output' in config")

    samp       = cfg["sampling"]
    n_samples  = samp["n_samples"]
    seq_len    = samp["seq_len"]
    batch_size = samp.get("batch_size", 1)
    src        = cfg["source"]
    src_type   = src["type"]
    compute_jacobians      = cfg.get("compute_jacobians", False)
    store_full_jacobians   = cfg.get("store_full_jacobians", False)
    compute_jacobian_stats = cfg.get("compute_jacobian_stats", False)
    compute_approx_sigma   = cfg.get("compute_approx_sigma", False)
    approx_sigma_probes    = cfg.get("approx_sigma_probes", 64)
    jac_chunk              = cfg.get("jac_chunk", 64)
    weights                = cfg.get("weights", "real")
    if weights not in ("real", "random"):
        raise ValueError(f"'weights' must be 'real' or 'random', got {weights!r}")

    allow_override = cfg.get("allow_override", False)

    def _mark_done(config_path):
        with open(config_path) as _f:
            _d = json.load(_f)
        _d["done"] = True
        with open(config_path, "w") as _f:
            json.dump(_d, _f, indent=2)

    def _check_meta_match(meta, source_label):
        mismatches = []
        for key, expected in [
            ("model",       model_name),
            ("seq_len",     seq_len),
            ("n_samples",   n_samples),
            ("source_type", src_type),
            ("weights",     weights),
        ]:
            got = meta.get(key)
            if got != expected:
                mismatches.append(f"  {key}: {source_label} has {got!r}, config wants {expected!r}")
        return mismatches

    def _upload(path):
        if args.no_upload:
            return
        import subprocess
        result = subprocess.run(
            ["python", str(Path(__file__).parent / "upload.py"), "push", "-y", path],
            check=False,
        )
        if result.returncode != 0:
            print(f"Warning: upload failed for {path}")

    # ── remote pointer check (before loading model) ───────────────────────────
    _ptr_path = Path(__file__).parent / ".ptrs" / (output + ".ptr")
    if _ptr_path.exists():
        with open(_ptr_path) as _f:
            _ptr = json.load(_f)
        _meta = _ptr.get("h5_meta")
        if _meta:
            mismatches = _check_meta_match(_meta, "remote")
            if mismatches:
                msg = (
                    f"Remote file {output!r} exists with different metadata:\n" +
                    "\n".join(mismatches) +
                    "\nSet \"allow_override\": true in the config to overwrite it."
                )
                if not allow_override:
                    raise ValueError(msg)
                print(f"WARNING: overriding remote file — {msg}")
            else:
                # ptr exists + meta matches → always complete (we never upload incomplete)
                print(f"Run already complete on remote (skipping). Output: {output}")
                _mark_done(args.config)
                return
        else:
            print(f"Note: remote pointer exists for {output!r} but has no h5_meta. Proceeding.")

    # ── local complete check (before loading model) ───────────────────────────
    if Path(output).exists():
        try:
            with h5py.File(output, "r") as existing:
                m = existing["meta"].attrs
                mismatches = _check_meta_match(m, "local file")
                if mismatches:
                    raise ValueError(
                        f"Existing file {output!r} metadata does not match config:\n" +
                        "\n".join(mismatches)
                    )
                if m.get("status") == "complete":
                    print(f"Local run complete. Uploading {output} ...")
                    _upload(output)
                    _mark_done(args.config)
                    return
        except OSError:
            print(f"WARNING: corrupt local file {output!r} — deleting and restarting.")
            Path(output).unlink()

    wb = cfg.get("wandb", {})
    if wb is not False:
        import wandb
        wandb.init(
            entity=wb.get("entity"),
            project=wb.get("project", "extract-data"),
            name=wb.get("name") or f"{model_name}-{src_type}",
            tags=wb.get("tags"),
            config=cfg,
        )

    print(f"Loading {model_name!r} on {device} ...")
    if model_name == "custom":
        custom_cfg = hj.Config(**cfg.get("custom_model_config", {}))
        model = hj.CustomModel(custom_cfg).to(device).eval()
        checkpoint = cfg.get("checkpoint")
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        tok     = None
        d_model = custom_cfg.d_model
    else:
        model, tok = hj.load(model_name, device=device)
        d_model    = model.config.hidden_size
    if weights == "random":
        hj.reinit_weights(model)
        print("  weights re-initialised\n")
    n_layers  = len(_layers(model))
    sublayers = ("block",) if type(model).__name__ == "GPTNeoXModel" else _SUBLAYERS
    print(f"  {n_layers} layers, d_model={d_model}\n")

    # pre-flight: compute source-specific extra metadata
    extra_meta: dict = {}
    if src_type == "benchmark":
        import skdim
        _bm = skdim.datasets.BenchmarkManifolds(random_state=src.get("seed", 42))
        extra_meta["benchmark_name"]    = src["name"]
        extra_meta["benchmark_true_id"] = int(_bm.truth.loc[src["name"], "Intrinsic Dimension"])
        extra_meta["ambient_dim"]       = int(_bm.generate(n=seq_len)[src["name"]].shape[1])

    # ── resume incomplete local file ──────────────────────────────────────────
    skip = 0
    h5_mode = "w"
    if Path(output).exists():
        with h5py.File(output, "r") as existing:
            skip = int(existing["embed_out"].shape[0]) if "embed_out" in existing else 0
        print(f"Resuming from sample {skip} / {n_samples}  ({output})")
        h5_mode = "a"

    # ── build iterator ────────────────────────────────────────────────────────
    if src_type == "dataset":
        if tok is None:
            raise ValueError("source type 'dataset' requires a tokenizer — CustomModel has none; use 'manifold' or 'benchmark' instead")
        batches = iter_dataset_batches(src, tok, n_samples, seq_len, batch_size, device, skip=skip)
    elif src_type == "manifold":
        batches = iter_manifold_batches(src, n_samples, seq_len, batch_size, d_model, device, skip=skip)
    elif src_type == "benchmark":
        batches = iter_benchmark_batches(src, n_samples, seq_len, batch_size, d_model, device, skip=skip)
    elif src_type == "random_tokens":
        if model is None:
            raise ValueError("source type 'random_tokens' requires a loaded model")
        batches = iter_random_token_batches(src, model, n_samples, seq_len, batch_size, device, skip=skip)
    else:
        raise ValueError(f"Unknown source type {src_type!r}. Choose: dataset, manifold, benchmark, random_tokens")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, h5_mode) as f:
        if h5_mode == "w":
            write_meta(
                f, model_name, src_type, src, n_samples, seq_len, batch_size, extra_meta,
                weights=weights, d_model=d_model, n_layers=n_layers,
                compute_jacobians=compute_jacobians, compute_jacobian_stats=compute_jacobian_stats,
                compute_approx_sigma=compute_approx_sigma, approx_sigma_probes=approx_sigma_probes,
            )

        def _gpu_postfix():
            if not torch.cuda.is_available():
                return {}
            dev = torch.cuda.current_device()
            mem_alloc = torch.cuda.memory_allocated(dev) / 1024**3
            mem_total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
            mem_free  = mem_total - mem_alloc
            util = torch.cuda.utilization(dev)
            return {"gpu": f"{util}%", "vram_free": f"{mem_free:.1f}GB"}

        actual_samples = skip
        with tqdm.tqdm(total=n_samples, initial=skip, desc="samples") as pbar:
            for offset, batch, raw in batches:
                B = batch.shape[0]
                actual_samples += B
                if batch.is_floating_point():
                    batch = batch.to(dtype=next(model.parameters()).dtype)

                if compute_approx_sigma:
                    from torch.func import jvp as _jvp
                    store = capture_all_hidden(model, batch)

                    if src_type == "dataset":
                        _ds_append(f, "input_ids", batch.cpu().numpy())
                    elif src_type == "random_tokens":
                        _ds_append(f, "token_ids", raw.numpy())
                    elif raw is not None:
                        _ds_append(f, "manifold_points", raw.numpy())

                    _ds_append(f, "embed_out",    store[("embed", "out")].float().cpu().numpy())
                    _ds_append(f, "final_hidden", store[("final", "out")].float().cpu().numpy())

                    for layer_idx in range(n_layers):
                        layer = _layers(model)[layer_idx]
                        for sub in sublayers:
                            if (layer_idx, sub) not in store:
                                continue
                            h_B = store[(layer_idx, sub)].to(device)  # (B, seq, d)
                            _ds_append(f, f"layer_{layer_idx}/{sub}/hidden_out",
                                       h_B.float().cpu().numpy())

                            fn = _sublayer_fn(layer, sub, model=model)
                            # HF models take (B, seq, d); wrap to (seq, d) for probe closure.
                            # JVP only supports float32+; temporarily upcast layer for probes.
                            from hf_jacobian.custom_model import CustomModel as _CM
                            _model_dtype = next(model.parameters()).dtype
                            _probe_dtype = torch.float32
                            _needs_upcast = _model_dtype in (torch.float16, torch.bfloat16)
                            if _needs_upcast:
                                layer.float()
                            if not isinstance(model, _CM):
                                _fn = lambda x, _f=fn: _f(x.unsqueeze(0)).squeeze(0)
                            else:
                                _fn = fn

                            smin_batch = []
                            inv_batch  = []
                            for b in range(B):
                                h = h_B[b].to(_probe_dtype)        # (seq, d)
                                smin_seq = []
                                inv_seq  = []
                                pbar.set_postfix({**_gpu_postfix(), "layer": f"{layer_idx}/{n_layers}", "sample": f"{offset+b}/{n_samples}"})
                                for p in range(h.shape[0]):
                                    ctx = h.detach().clone()
                                    def f_loc(x, _ctx=ctx, _p=p, _f=_fn):
                                        full = _ctx.clone()
                                        full[_p] = x
                                        return _f(full)[_p]
                                    x_p   = h[p].detach().clone()
                                    Jv_fn = lambda v, _fl=f_loc, _x=x_p: _jvp(_fl, (_x,), (v,))[1]
                                    res   = check_invertibility(Jv_fn, n=d_model, m=approx_sigma_probes,
                                                                device=device, dtype=_probe_dtype)
                                    smin_seq.append(res.sigma_min_estimate)
                                    inv_seq.append(float(res.is_invertible))
                                smin_batch.append(smin_seq)
                                inv_batch.append(inv_seq)

                            if _needs_upcast:
                                layer.to(_model_dtype)
                            _ds_append(f, f"layer_{layer_idx}/{sub}/approx_sigma_min",
                                       np.array(smin_batch, dtype=np.float32))
                            _ds_append(f, f"layer_{layer_idx}/{sub}/is_invertible",
                                       np.array(inv_batch,  dtype=np.float32))
                        pbar.set_postfix(_gpu_postfix())
                    pbar.update(B)

                elif not compute_jacobians:
                    store = capture_all_hidden(model, batch)

                    if src_type == "dataset":
                        _ds_append(f, "input_ids", batch.cpu().numpy())
                    elif src_type == "random_tokens":
                        _ds_append(f, "token_ids", raw.numpy())
                    elif raw is not None:
                        _ds_append(f, "manifold_points", raw.numpy())

                    _ds_append(f, "embed_out",    store[("embed", "out")].float().cpu().numpy())
                    _ds_append(f, "final_hidden", store[("final", "out")].float().cpu().numpy())
                    for layer_idx in range(n_layers):
                        for sub in sublayers:
                            if (layer_idx, sub) not in store:
                                continue
                            _ds_append(f, f"layer_{layer_idx}/{sub}/hidden_out",
                                       store[(layer_idx, sub)].float().cpu().numpy())
                    pbar.update(B)
                    pbar.set_postfix(_gpu_postfix())
                else:
                    # full jacrev Jacobians
                    # single forward pass captures all hidden states for this batch
                    store = capture_all_hidden(model, batch)
                    _ds_append(f, "embed_out",    store[("embed", "out")].float().cpu().numpy())
                    _ds_append(f, "final_hidden", store[("final", "out")].float().cpu().numpy())

                    if src_type == "dataset":
                        _ds_append(f, "input_ids", batch.cpu().numpy())
                    elif src_type == "random_tokens":
                        _ds_append(f, "token_ids", raw.numpy())
                    elif raw is not None:
                        _ds_append(f, "manifold_points", raw.numpy())

                    for layer_idx in range(n_layers):
                        pbar.set_postfix({**_gpu_postfix(), "layer": f"{layer_idx}/{n_layers}"})
                        for sub in sublayers:
                            try:
                                hidden_out, jac = _causal_block_jac(
                                    model, batch, layer_idx, sub, store=store
                                )
                            except ValueError:
                                continue
                            _ds_append(f, f"layer_{layer_idx}/{sub}/hidden_out",
                                       hidden_out.float().cpu().numpy())
                            if store_full_jacobians:
                                _ds_append(f, f"layer_{layer_idx}/{sub}/jacobian",
                                           jac.float().cpu().numpy())
                            if compute_jacobian_stats:
                                stats = jacobian_stats(jac)
                                for key in ("log_det", "sigma_max", "sigma_min", "singular_values"):
                                    t = stats.get(key)
                                    if t is not None:
                                        _ds_append(f, f"layer_{layer_idx}/{sub}/{key}",
                                                   t.float().cpu().numpy())
                    pbar.update(B)
                    pbar.set_postfix(_gpu_postfix())

        f["meta"].attrs["n_samples"] = actual_samples
        f["meta"].attrs["n_tokens"]  = actual_samples * seq_len
        f["meta"].attrs["status"]    = "complete"
        print(f"\nActual samples: {actual_samples}  |  tokens: {actual_samples * seq_len}")

    _mark_done(args.config)
    print(f"Uploading {output} ...")
    _upload(output)


if __name__ == "__main__":
    main()
