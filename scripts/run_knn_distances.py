"""
Download h5 files from remote, run NN-distance analysis, save parquet, delete h5.

Usage:
  uv run python scripts/run_knn_distances.py configs/results/knn_distances.json
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import upload
from scripts.plot_knn_distances import load_from_h5

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

COMPRESSION = "zstd"


def resolve_ptr(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    h5_rel = cfg["output"]          # e.g. "out/gpt2/0.124b/fp32/dataset/c4_n1024_s64.h5"
    ptr_path = PROJECT_ROOT / ".ptrs" / upload.ptr_name(h5_rel)
    if not ptr_path.exists():
        sys.exit(f"Pointer not found: {ptr_path}")
    with open(ptr_path) as f:
        return json.load(f)


def pull_all(ptrs: list[dict], remote_cfg: dict, workers: int = 4) -> list[Path]:
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


def analyze(h5_path: Path, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name, depths, seq_len, mean_grid, std_grid, all_dists, all_dists_list = load_from_h5(h5_path)

    rows = []
    for di, label in enumerate(depths):
        dists_dp = all_dists_list[di]   # (n_samples, seq_len)
        for pos in range(seq_len):
            for sample_i, v in enumerate(dists_dp[:, pos]):
                rows.append({"depth": label, "pos": pos, "sample": sample_i, "nn_dist": float(v)})

    out_path = out_dir / f"{stem}_nn.parquet"
    pd.DataFrame(rows).to_parquet(out_path, compression=COMPRESSION, index=False)
    print(f"  saved → {out_path}")
    return out_path


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: run_knn_distances.py <big_config.json>")

    big_cfg_path = Path(sys.argv[1])
    with open(big_cfg_path) as f:
        big_cfg = json.load(f)

    out_dir = PROJECT_ROOT / big_cfg["out_dir"]
    remote_cfg = upload.load_config()

    runs = [(PROJECT_ROOT / entry, (PROJECT_ROOT / entry).stem) for entry in big_cfg["runs"] if entry]
    ptrs = [resolve_ptr(config_path) for config_path, _ in runs]

    total_mb = sum(p["size"] for p in ptrs) / 1_048_576
    print(f"Pulling {len(ptrs)} file(s) ({total_mb:.1f} MB total)...")
    h5_paths = pull_all(ptrs, remote_cfg, workers=remote_cfg.get("workers", 4))

    for (config_path, stem), h5_path in zip(runs, h5_paths):
        print(f"\n=== {stem} ===")
        analyze(h5_path, out_dir, stem)
        h5_path.unlink()
        print(f"  deleted {h5_path.name}")


if __name__ == "__main__":
    main()
