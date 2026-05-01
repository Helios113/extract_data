"""
One-off migration: add h5_meta to pointer files that lack it, then sync to remote.

Meta is derived from matching config JSON files in configs/, not from the H5 blobs.
For each .ptr missing h5_meta, we find the config whose output path matches, read
the relevant fields, write them into the ptr, and push the updated ptr to remote.

Usage:
    python migrate_ptr_meta.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PTRS_DIR     = PROJECT_ROOT / ".ptrs"
CONFIGS_DIR  = PROJECT_ROOT / "configs"


def load_upload_config():
    import configparser
    cfg = configparser.ConfigParser()
    if not cfg.read(PROJECT_ROOT / "upload.cfg"):
        sys.exit("upload.cfg not found")
    r = cfg["remote"]
    return {
        "host":        r["host"],
        "username":    r["username"],
        "password":    r["password"],
        "remote_path": r["remote_path"].rstrip("/"),
    }


def open_sftp(config):
    import paramiko
    t = paramiko.Transport((config["host"], 23))
    t.use_compression(True)
    t.connect(username=config["username"], password=config["password"])
    return paramiko.SFTPClient.from_transport(t)


def _meta_from_config(cfg: dict) -> dict:
    """Extract the fields that run.py writes into /meta attrs."""
    samp = cfg["sampling"]
    src  = cfg["source"]
    return {
        "model":       cfg["model"],
        "weights":     cfg.get("weights", "real"),
        "source_type": src["type"],
        "n_samples":   samp["n_samples"],
        "seq_len":     samp["seq_len"],
        "batch_size":  samp.get("batch_size", 1),
        "status":      "complete",
    }


def build_output_to_config_map() -> dict[str, dict]:
    mapping = {}
    for cfg_path in CONFIGS_DIR.rglob("*.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        output = cfg.get("output")
        if output:
            mapping[output] = cfg
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ptr_files = sorted(PTRS_DIR.rglob("*.ptr"))
    missing   = [(p, json.loads(p.read_text()))
                 for p in ptr_files
                 if not json.loads(p.read_text()).get("h5_meta")]

    print(f"Found {len(ptr_files)} ptr files, {len(missing)} missing h5_meta.\n")
    if not missing:
        print("Nothing to migrate.")
        return

    output_map = build_output_to_config_map()

    can_migrate, no_config = [], []
    for ptr_path, ptr in missing:
        cfg = output_map.get(ptr["original_path"])
        if cfg is None:
            no_config.append(ptr["original_path"])
        else:
            can_migrate.append((ptr_path, ptr, cfg))

    if no_config:
        print(f"No matching config found for {len(no_config)} ptr(s) — skipping:")
        for p in no_config:
            print(f"  {p}")
        print()

    if not can_migrate:
        print("Nothing to migrate.")
        return

    print(f"{'[dry-run] ' if args.dry_run else ''}Will update {len(can_migrate)} ptr(s):")
    for _, ptr, _ in can_migrate:
        print(f"  {ptr['original_path']}")

    if args.dry_run:
        return

    upload_cfg = load_upload_config()
    sftp = open_sftp(upload_cfg)
    updated = 0
    try:
        for ptr_path, ptr, cfg in can_migrate:
            ptr["h5_meta"] = _meta_from_config(cfg)
            ptr_path.write_text(json.dumps(ptr, indent=2) + "\n")
            remote_idx = f"{upload_cfg['remote_path']}/index/{ptr['original_path']}.ptr"
            _sftp_mkdir_p(sftp, remote_idx.rsplit("/", 1)[0])
            with sftp.open(remote_idx, "w") as f:
                f.write(json.dumps(ptr, indent=2))
            print(f"  updated: {ptr['original_path']}")
            updated += 1
    finally:
        sftp.close()

    print(f"\nUpdated {updated} / {len(can_migrate)}.")


def _sftp_mkdir_p(sftp, remote_dir):
    parts = [p for p in remote_dir.split("/") if p]
    prefix = "/" if remote_dir.startswith("/") else ""
    for i in range(len(parts)):
        path = prefix + "/".join(parts[:i+1])
        try:
            sftp.stat(path)
        except FileNotFoundError:
            try:
                sftp.mkdir(path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
