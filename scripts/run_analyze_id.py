"""
Download h5 files from remote, run ID analysis, save parquet per run, delete h5.

Usage:
  uv run python scripts/run_analyze_id.py configs/results/analyze_id.json
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

COMPRESSION = "zstd"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import upload
from hf_jacobian.id_estimators import twonn
from scripts.analyze_id import (
    depth_keys,
    load_latents,
    compute_ess_and_pca,
    load_entropy_lens,
    compute_entropy_per_sample,
)

# ── per-process entropy lens cache ───────────────────────────────────────────
_entropy_cache: dict = {}

def _get_entropy_lens(model_name: str):
    if model_name not in _entropy_cache:
        _entropy_cache[model_name] = load_entropy_lens(model_name)
    return _entropy_cache[model_name]


# ── cell worker (runs in a subprocess) ───────────────────────────────────────

def compute_cell(args):
    depth, pos, X_numpy, ks, model_name = args
    X = torch.from_numpy(X_numpy)
    N = X.shape[0]

    id_twonn = twonn(X)
    per_k    = {k: compute_ess_and_pca(X, k) for k in ks}

    ln, lm_head = _get_entropy_lens(model_name)
    entropy = compute_entropy_per_sample(X, ln, lm_head)

    rows = []
    for i in range(N):
        row = {"depth": depth, "pos": pos, "sample": i,
               "twonn": id_twonn, "entropy": float(entropy[i])}
        for k in ks:
            row[f"ess_a_k{k}"]    = per_k[k]["ess_a_pw"][i]
            row[f"pca_curv_k{k}"] = per_k[k]["pca_curv_pw"][i]
            row[f"nbh_size"] = float(per_k[k]["neighbourhood_size"][i])
        rows.append(row)

    k0 = ks[0]
    summary = (id_twonn,
               float(np.nanmean(per_k[k0]["ess_a_pw"])),
               float(np.nanmean(per_k[k0]["pca_curv_pw"])),
               float(np.nanmean(per_k[k0]["neighbourhood_size"])),
               float(np.mean(entropy)))
    return depth, pos, rows, summary


def _load_hidden_pos(f: h5py.File, depth: str, pos: int) -> np.ndarray:
    if depth == "embed":
        data = f["embed_out"][:, pos, :]
    elif depth == "final":
        data = f["final_hidden"][:, pos, :]
    else:
        data = f[f"{depth}/hidden_out"][:, pos, :]
    return np.array(data, dtype=np.float64)


def compute_knn_cell(args):
    h5_path, depth, pos = args
    with h5py.File(h5_path, "r") as f:
        pts = _load_hidden_pos(f, depth, pos)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return dists[:, 1].astype(np.float32)


# ── remote helpers ────────────────────────────────────────────────────────────

def resolve_ptr(config_path: Path) -> dict:
    print(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    h5_rel = cfg["output"]
    ptr_path = PROJECT_ROOT / ".ptrs" / upload.ptr_name(h5_rel)
    if not ptr_path.exists():
        sys.exit(f"Pointer not found: {ptr_path}")
    with open(ptr_path) as f:
        return json.load(f)


def pull_all(ptrs: list[dict], remote_cfg: dict, workers: int = 12) -> list[Path]:
    for ptr in ptrs:
        dest = PROJECT_ROOT / ptr["original_path"]
        if dest.exists() and upload.hash_file(str(dest)) != ptr["hash"]:
            print(f"WARNING: local file differs, overwriting: {dest}")

    upload._board = upload._Board(workers) if upload._IS_TTY else upload._LogBoard()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload._pull_file, ptr, remote_cfg): ptr for ptr in ptrs}
        for fut in as_completed(futures):
            ptr = futures[fut]
            dest, _ = fut.result()
            results[ptr["original_path"]] = Path(dest)
    return [results[ptr["original_path"]] for ptr in ptrs]


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze(h5_path: Path, out_dir: Path, stem: str, ks: list[int], workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        seq_len    = int(f["meta"].attrs["seq_len"])
        n_samples  = int(f["meta"].attrs["n_samples"])
        model_name = str(f["meta"].attrs["model"])
        depths     = depth_keys(f)
        n_cells    = len(depths) * seq_len
        
        print(f"  depths={len(depths)}  seq_len={seq_len}  n_samples={n_samples}  k={ks}  cells={n_cells}")

        all_rows  = [None] * n_cells
        done      = 0
        # keep at most 2× workers tasks in flight so we don't pre-load the whole h5
        max_queue = workers * 2
        pending   = {}  # future → cell_idx
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cell_iter = (
                (idx, depth, pos)
                for idx, (depth, pos) in enumerate(
                    (d, p) for d in depths for p in range(seq_len)
                )
            )

            def submit_next():
                try:
                    idx, depth, pos = next(cell_iter)
                    X_numpy = load_latents(f, depth, pos).numpy()
                    fut = pool.submit(compute_cell, (depth, pos, X_numpy, ks, model_name))
                    pending[fut] = idx
                except StopIteration:
                    pass

            # fill initial queue
            for i in range(max_queue):
                submit_next()
                print(i, "queued")

            while pending:
                fut = next(as_completed(pending))
                idx = pending.pop(fut)
                depth, pos, rows, (tw, ea, pc, ent, neig) = fut.result()
                all_rows[idx] = rows
                done += 1
                print(f"  [{done}/{n_cells}] {depth}  pos={pos}"
                      f"  twonn={tw:.2f}  ess_a={ea:.2f}"
                      f"  pca_curv={pc:.3f}  entropy={ent:.2f}, nbh_size={neig:.2f}",
                      flush=True)
                submit_next()

    out_path = out_dir / f"{stem}_id.parquet"
    pd.DataFrame(itertools.chain.from_iterable(all_rows)).to_parquet(
        out_path, compression=COMPRESSION, index=False
    )
    print(f"  saved → {out_path}")
    return out_path


def analyze_knn(h5_path: Path, out_dir: Path, stem: str, workers: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        seq_len = int(f["meta"].attrs["seq_len"])
        depths = depth_keys(f)

    n_cells = len(depths) * seq_len
    all_rows = [None] * n_cells
    done = 0
    max_queue = workers * 2
    pending = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        cell_iter = (
            (idx, depth, pos)
            for idx, (depth, pos) in enumerate(
                (d, p) for d in depths for p in range(seq_len)
            )
        )

        def submit_next():
            try:
                idx, depth, pos = next(cell_iter)
                fut = pool.submit(compute_knn_cell, (str(h5_path), depth, pos))
                pending[fut] = (idx, depth, pos)
            except StopIteration:
                pass

        for _ in range(max_queue):
            submit_next()

        while pending:
            fut = next(as_completed(pending))
            idx, depth, pos = pending.pop(fut)
            dists = fut.result()
            all_rows[idx] = [
                {"depth": depth, "pos": pos, "sample": i, "nn_dist": float(v)}
                for i, v in enumerate(dists)
            ]
            done += 1
            print(f"  [knn {done}/{n_cells}] {depth}  pos={pos}", flush=True)
            submit_next()

    rows = [r for cell in all_rows for r in cell]

    out_path = out_dir / f"{stem}_nn.parquet"
    pd.DataFrame(rows).to_parquet(out_path, compression=COMPRESSION, index=False)
    print(f"  saved → {out_path}")
    return out_path

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: run_analyze_id.py <big_config.json>")

    big_cfg_path = Path(sys.argv[1])
    with open(big_cfg_path) as f:
        big_cfg = json.load(f)

    out_dir    = PROJECT_ROOT / big_cfg["out_dir"]
    ks         = [int(k) for k in str(big_cfg.get("ess_k", "25")).split(",")]
    remote_cfg = upload.load_config()
    workers    = remote_cfg.get("workers", 64)

    runs = [(PROJECT_ROOT / entry, (PROJECT_ROOT / entry).stem) for entry in big_cfg["runs"] if entry]
    ptrs = [resolve_ptr(config_path) for config_path, _ in runs]

    total_mb = sum(p["size"] for p in ptrs) / 1_048_576
    print(f"Pulling {len(ptrs)} file(s) ({total_mb:.1f} MB total)...")
    h5_paths = pull_all(ptrs, remote_cfg, workers=workers)

    for (config_path, stem), h5_path in zip(runs, h5_paths):
        config_str = str(config_path)
        if "real" in config_str:
            stem = f"real_{stem}"
        elif "synthetic" in config_str:
            stem = f"synthetic_{stem}"

        print(f"\n=== {stem} ===")
        analyze(h5_path, out_dir, stem, ks, workers=60)
        analyze_knn(h5_path, out_dir, stem, 60)
        
        h5_path.unlink()
        print(f"  deleted {h5_path.name}")


if __name__ == "__main__":
    main()



# export OMP_NUM_THREADS=8
# export OPENBLAS_NUM_THREADS=8
# export MKL_NUM_THREADS=8
# export VECLIB_MAXIMUM_THREADS=8
# export NUMEXPR_NUM_THREADS=8

# uv run python scripts/run_analyze_id.py