#!/usr/bin/env python3
"""
Data remote storage tool -- DVC-style content-addressed storage over SFTP.

Remote layout:
  <remote_path>/
    cache/<hash[:2]>/<hash[2:]>  -- content-addressed data blobs
    index/<filename>.ptr          -- pointer files (source of truth for ls)

Local layout:
  .ptrs/<filename>.ptr            -- mirrors of pointer files, tracked by git

Commands:
  push <file|dir>  Hash, upload, write pointers, optionally delete local
  pull <name>      Download file described by .ptrs/<name>.ptr
  ls               Browse remote index interactively
"""

import configparser
from datetime import datetime, timezone
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import get_terminal_size

import paramiko
import xxhash


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "upload.cfg")
PTRS_DIR = os.path.join(PROJECT_ROOT, ".ptrs")
HETZNER_PORT = 23
DEFAULT_WORKERS = 4

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    cfg = configparser.ConfigParser()
    if not cfg.read(CONFIG_FILE):
        sys.exit(f"Config file not found: {CONFIG_FILE}")
    try:
        return {
            "host": cfg["remote"]["host"],
            "username": cfg["remote"]["username"],
            "password": cfg["remote"]["password"],
            "remote_path": cfg["remote"]["remote_path"].rstrip("/"),
            "workers": cfg["remote"].getint("workers", DEFAULT_WORKERS),
        }
    except KeyError as e:
        sys.exit(f"Missing config key: {e}")


# ---------------------------------------------------------------------------
# SFTP helpers
# ---------------------------------------------------------------------------

def connect(config):
    transport = paramiko.Transport((config["host"], HETZNER_PORT))
    transport.use_compression(True)
    transport.connect(username=config["username"], password=config["password"])
    return transport


def open_sftp(config):
    return paramiko.SFTPClient.from_transport(connect(config))


def sftp_mkdir_p(sftp, remote_dir):
    parts = [p for p in remote_dir.split("/") if p]
    if remote_dir.startswith("/"):
        paths = ["/" + "/".join(parts[:i+1]) for i in range(len(parts))]
    else:
        paths = ["/".join(parts[:i+1]) for i in range(len(parts))]
    for path in paths:
        try:
            sftp.stat(path)
        except FileNotFoundError:
            try:
                sftp.mkdir(path)
            except OSError:
                sftp.stat(path)  # only ok if another thread already created it


# ---------------------------------------------------------------------------
# Progress display
#
# n_slots lines are printed upfront. Each worker thread owns one slot (row).
# ANSI cursor movement rewrites that row in place — strictly one line per file.
# ---------------------------------------------------------------------------

class _Board:
    def __init__(self, n_slots):
        self.n_slots = n_slots
        self._tid_slot = {}   # thread id -> slot index
        self._next = 0
        # print the blank rows we'll overwrite
        sys.stdout.write("\n" * n_slots)
        sys.stdout.flush()

    def _slot(self):
        tid = threading.get_ident()
        with _print_lock:
            if tid not in self._tid_slot:
                self._tid_slot[tid] = self._next
                self._next += 1
        return self._tid_slot[tid]

    def _write(self, slot, text):
        rows_up = self.n_slots - slot
        width = get_terminal_size((80, 20)).columns
        # pad/truncate to exactly one terminal width so nothing spills
        line = f"\r{text}"[:width].ljust(width)
        with _print_lock:
            sys.stdout.write(f"\x1b[{rows_up}A{line}\x1b[{rows_up}B")
            sys.stdout.flush()

    def make_callback(self, label, total):
        label = label[:24]
        slot = self._slot()
        t0 = time.monotonic()

        def callback(transferred, _total):
            pct = transferred / total * 100 if total else 0
            elapsed = time.monotonic() - t0
            rate = transferred / elapsed if elapsed > 0.1 else 0
            remaining = (total - transferred) / rate if rate > 0 else 0
            eta = f"{int(remaining // 60)}m{int(remaining % 60):02d}s" if remaining > 0 else "--:--"
            width = get_terminal_size((80, 20)).columns
            bar_w = max(width - 46, 8)
            filled = int(bar_w * pct / 100)
            bar = "#" * filled + "-" * (bar_w - filled)
            self._write(slot, f"{label:<24} [{bar}] {pct:5.1f}% ETA {eta:>7}")

        return callback

    def finish(self, label, size):
        slot = self._tid_slot.get(threading.get_ident())
        if slot is None:
            return
        self._write(slot, f"{label:<24}  done  {size / 1_048_576:.1f} MB")


_board: _Board | None = None


def sftp_put(sftp, local_path, remote_path):
    size = os.path.getsize(local_path)
    name = os.path.basename(local_path)
    sftp.put(local_path, remote_path, callback=_board.make_callback(name, size))
    _board.finish(name, size)


def sftp_get(sftp, remote_path, local_path):
    size = sftp.stat(remote_path).st_size
    name = os.path.basename(local_path)
    sftp.get(remote_path, local_path, callback=_board.make_callback(name, size))
    _board.finish(name, size)


# ---------------------------------------------------------------------------
# Hashing  (xxh64: ~3-5x faster than MD5, sufficient for dedup/integrity)
# ---------------------------------------------------------------------------

def hash_file(path, chunk=1 << 20):
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pointer files
# ---------------------------------------------------------------------------

def ptr_name(filename):
    return filename if filename.endswith(".ptr") else filename + ".ptr"


def write_ptr(local_path, digest, size):
    os.makedirs(PTRS_DIR, exist_ok=True)
    name = os.path.basename(local_path)
    ptr_path = os.path.join(PTRS_DIR, ptr_name(name))
    data = {
        "filename": name,
        "hash": digest,
        "size": size,
        "mtime": os.path.getmtime(local_path),
        "original_path": os.path.relpath(local_path, PROJECT_ROOT),
    }
    with open(ptr_path, "w") as f:
        json.dump(data, f, indent=2)
    return ptr_path, data


def load_ptr(local_path):
    """Return cached ptr data if mtime matches, else None."""
    ptr_path = os.path.join(PTRS_DIR, ptr_name(os.path.basename(local_path)))
    if not os.path.exists(ptr_path):
        return None
    with open(ptr_path) as f:
        data = json.load(f)
    if data.get("mtime") == os.path.getmtime(local_path):
        return data
    return None


def read_ptr(name):
    ptr_path = os.path.join(PTRS_DIR, ptr_name(name))
    if not os.path.exists(ptr_path):
        sys.exit(f"Pointer not found: {ptr_path}")
    with open(ptr_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Remote path helpers
# ---------------------------------------------------------------------------

def remote_cache_path(config, digest):
    return f"{config['remote_path']}/cache/{digest[:2]}/{digest[2:]}"


def remote_index_path(config, filename):
    return f"{config['remote_path']}/index/{ptr_name(filename)}"


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def _push_file(local_path, config):
    """Run in a worker thread. Returns (local_path, ok: bool)."""
    name = os.path.basename(local_path)
    size = os.path.getsize(local_path)

    cached = load_ptr(local_path)
    if cached:
        digest = cached["hash"]
        tprint(f"[{name}] mtime unchanged, reusing hash {digest[:8]}…")
    else:
        tprint(f"[{name}] hashing ({size / 1_048_576:.1f} MB)...")
        digest = hash_file(local_path)
        tprint(f"[{name}] hash: {digest}")

    sftp = open_sftp(config)
    try:
        remote_idx = remote_index_path(config, name)
        sftp_mkdir_p(sftp, f"{config['remote_path']}/index")

        # compare with remote index
        try:
            with sftp.open(remote_idx) as f:
                existing = json.load(f)
            if existing["hash"] == digest:
                tprint(f"[{name}] hash matches remote, skipping upload.")
                # still update local ptr in case it was missing
                write_ptr(local_path, digest, size)
                return local_path, True
            else:
                # content changed — warn with timestamps and ask twice
                remote_mtime = existing.get("mtime")
                local_mtime  = os.path.getmtime(local_path)
                remote_ts = datetime.fromtimestamp(remote_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if remote_mtime else "unknown"
                local_ts  = datetime.fromtimestamp(local_mtime,  timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                with _print_lock:
                    print(f"\n[{name}] CONFLICT: remote and local have different content.")
                    print(f"  Remote version: {existing['hash'][:12]}…  modified {remote_ts}")
                    print(f"  Local  version: {digest[:12]}…  modified {local_ts}")
                    ans1 = input(f"  Overwrite remote with local version? [y/N] ").strip().lower()
                    if ans1 != "y":
                        print(f"  Skipped.")
                        return local_path, False
                    ans2 = input(f"  Are you sure? This will replace the remote version from {remote_ts}. [y/N] ").strip().lower()
                    if ans2 != "y":
                        print(f"  Skipped.")
                        return local_path, False
        except FileNotFoundError:
            pass  # new file, no conflict

        # upload blob (content-addressed — different hash = new blob, old one preserved)
        cache_remote = remote_cache_path(config, digest)
        sftp_mkdir_p(sftp, os.path.dirname(cache_remote))
        try:
            sftp.stat(cache_remote)
            tprint(f"[{name}] blob exists in cache, skipping upload.")
        except FileNotFoundError:
            sftp_put(sftp, local_path, cache_remote)

        # write pointer locally
        ptr_path, ptr_data = write_ptr(local_path, digest, size)
        tprint(f"[{name}] pointer: {ptr_path}")

        # mirror pointer to remote index
        with sftp.open(remote_idx, "w") as f:
            f.write(json.dumps(ptr_data, indent=2))

    finally:
        sftp.close()

    return local_path, True


def cmd_push(local_path, config):
    if not os.path.exists(local_path):
        sys.exit(f"Path does not exist: {local_path}")

    is_dir = os.path.isdir(local_path)
    files = (
        [os.path.join(dp, f) for dp, _, fnames in os.walk(local_path)
         for f in fnames if not f.startswith("._") and not f.startswith(".")]
        if is_dir else [local_path]
    )
    if not files:
        sys.exit("Directory is empty (or only hidden files).")

    workers = min(config["workers"], len(files))
    print(f"Starting {workers} parallel worker(s)...")

    global _board
    _board = _Board(workers)

    uploaded, skipped = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_push_file, f, config): f for f in files}
        for fut in as_completed(futures):
            path, ok = fut.result()
            (uploaded if ok else skipped).append(path)

    print(f"\nPush complete: {len(uploaded)} uploaded, {len(skipped)} skipped.")

    if uploaded:
        ans = input(f"Delete {len(uploaded)} local file(s)? [y/N] ").strip().lower()
        if ans == "y":
            for f in uploaded:
                os.remove(f)
            if is_dir:
                for dp, _, _ in os.walk(local_path, topdown=False):
                    try:
                        os.rmdir(dp)
                    except OSError:
                        pass
            print("Deleted.")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def _pull_file(ptr, config):
    """Run in a worker thread."""
    digest = ptr["hash"]
    dest = os.path.join(PROJECT_ROOT, ptr["original_path"])
    name = ptr["filename"]

    if os.path.exists(dest):
        if hash_file(dest) == digest:
            tprint(f"[{name}] already present, skipping.")
            return dest, False
        tprint(f"[{name}] WARNING: local file differs, overwriting.")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    sftp = open_sftp(config)
    try:
        sftp_get(sftp, remote_cache_path(config, digest), dest)
    finally:
        sftp.close()

    # update local pointer so mtime cache is valid on next push
    write_ptr(dest, digest, ptr["size"])

    return dest, True


def cmd_pull(name, config):
    global _board
    ptr = read_ptr(name)
    print(f"Downloading {ptr['filename']} ({ptr['size'] / 1_048_576:.1f} MB)...")
    _board = _Board(1)
    dest, pulled = _pull_file(ptr, config)
    if pulled:
        print(f"\nPulled to: {dest}")


# ---------------------------------------------------------------------------
# ls -- interactive directory browser
# ---------------------------------------------------------------------------

def _fetch_ptrs(config):
    sftp = open_sftp(config)
    try:
        index_dir = f"{config['remote_path']}/index"
        try:
            entries = sftp.listdir(index_dir)
        except FileNotFoundError:
            return []
        ptrs = []
        for fname in sorted(entries):
            if not fname.endswith(".ptr"):
                continue
            with sftp.open(f"{index_dir}/{fname}") as f:
                raw = f.read()
            if b"Mac OS X" in raw:
                try:
                    sftp.remove(f"{index_dir}/{fname}")
                    print(f"  [info] removed macOS metadata ptr: {fname!r}")
                except OSError as e:
                    print(f"  [warn] failed to remove macOS metadata ptr {fname!r}: {e}")
                continue
            if not raw.strip():
                continue
            try:
                ptrs.append(json.loads(raw))
            except json.JSONDecodeError:
                print(f"  [warn] skipping corrupt ptr: {fname!r} contents: {raw!r}")
                continue
        return ptrs
    finally:
        sftp.close()


def _build_tree(ptrs):
    """Map each directory path -> {subdirs: set, files: [ptr]}."""
    tree = {}
    for p in ptrs:
        parts = p["original_path"].replace("\\", "/").split("/")
        for depth in range(len(parts)):
            dir_path = "/".join(parts[:depth]) or "."
            if dir_path not in tree:
                tree[dir_path] = {"subdirs": set(), "files": []}
            if depth < len(parts) - 1:
                child = "/".join(parts[:depth + 1])
                tree[dir_path]["subdirs"].add(child)
            else:
                tree[dir_path]["files"].append(p)
    if "." not in tree:
        tree["."] = {"subdirs": set(), "files": []}
    return tree


def _browse(tree, config):
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Label, ListItem, ListView

    class Entry(ListItem):
        def __init__(self, label: str, kind: str, payload, **kwargs):
            super().__init__(Label(label), **kwargs)
            self.kind = kind
            self.payload = payload

    chosen = []

    class BrowserApp(App):
        CSS = """
        ListView { height: 1fr; border: solid $primary; }
        ListItem { padding: 0 1; }
        ListItem:hover { background: $boost; }
        ListItem.selected Label { color: $success; }
        ListItem.dir Label { color: $warning; }
        #status { height: 1; background: $surface; padding: 0 1; }
        """
        BINDINGS = [
            Binding("space", "select", "Toggle file"),
            Binding("p", "pull", "Pull selected"),
            Binding("q", "quit_app", "Quit"),
        ]

        def __init__(self, tree):
            super().__init__()
            self._tree = tree
            self._cwd = "."
            self._selected = set()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield ListView()
            yield Label("", id="status")
            yield Footer()

        def on_mount(self):
            self.title = "Remote file browser"
            self._refresh_list()

        def _refresh_list(self):
            node = self._tree.get(self._cwd, {"subdirs": set(), "files": []})
            subdirs = sorted(node["subdirs"])
            files   = sorted(node["files"], key=lambda p: p["filename"])
            self.sub_title = f"{self._cwd}/"

            def _safe_id(prefix, raw):
                slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
                return f"{prefix}_{slug}"

            lv = self.query_one(ListView)
            lv.clear()

            if self._cwd != ".":
                parent = "/".join(self._cwd.split("/")[:-1]) or "."
                item = Entry("▲  ../", "dir", parent, id=_safe_id("parent", self._cwd))
                item.add_class("dir")
                lv.append(item)

            for d in subdirs:
                name = d.split("/")[-1]
                item = Entry(f"▶  {name}/", "dir", d, id=_safe_id("d", d))
                item.add_class("dir")
                lv.append(item)

            for p in files:
                local = os.path.join(PROJECT_ROOT, p["original_path"])
                present = "✓" if os.path.exists(local) else "·"
                check = "☑" if p["filename"] in self._selected else "☐"
                size_mb = p["size"] / 1_048_576
                label = f"{check} {present}  {p['filename']:<40} {size_mb:7.1f} MB"
                item = Entry(label, "file", p, id=_safe_id("f", p["original_path"]))
                if p["filename"] in self._selected:
                    item.add_class("selected")
                lv.append(item)

            self._update_status()

        def _update_status(self):
            n = len(self._selected)
            msg = (f"  {n} file(s) selected — P to pull · Q to quit"
                   if n else
                   "  Enter: open dir · Space: toggle file · P: pull · Q: quit")
            self.query_one("#status", Label).update(msg)

        def on_list_view_selected(self, event: ListView.Selected):
            # Enter key — navigate into dirs, toggle files
            item = event.item
            if not isinstance(item, Entry):
                return
            if item.kind == "dir":
                self._cwd = item.payload
                self._selected.clear()
                self._refresh_list()
            else:
                self._toggle_file(item)

        def action_select(self):
            # Space key — toggle files only
            lv = self.query_one(ListView)
            item = lv.highlighted_child
            if isinstance(item, Entry) and item.kind == "file":
                self._toggle_file(item)

        def _toggle_file(self, item: "Entry"):
            p = item.payload
            name = p["filename"]
            if name in self._selected:
                self._selected.discard(name)
                item.remove_class("selected")
            else:
                self._selected.add(name)
                item.add_class("selected")
            local = os.path.join(PROJECT_ROOT, p["original_path"])
            present = "✓" if os.path.exists(local) else "·"
            check = "☑" if name in self._selected else "☐"
            size_mb = p["size"] / 1_048_576
            item.query_one(Label).update(f"{check} {present}  {name:<40} {size_mb:7.1f} MB")
            self._update_status()

        def action_pull(self):
            if not self._selected:
                self._update_status()
                return
            # collect ptr dicts from every dir in the tree
            all_files = [p for node in self._tree.values() for p in node["files"]]
            nonlocal chosen
            chosen = [p for p in all_files if p["filename"] in self._selected]
            self.exit()

        def action_quit_app(self):
            self.exit()

    BrowserApp(tree).run()
    return chosen


def cmd_ls(config):
    print("Fetching remote index...")
    ptrs = _fetch_ptrs(config)
    if not ptrs:
        print("Remote index is empty (no files pushed yet).")
        return

    tree = _build_tree(ptrs)
    chosen = _browse(tree, config)

    if not chosen:
        return

    # write local pointers for chosen files
    os.makedirs(PTRS_DIR, exist_ok=True)
    for p in chosen:
        ptr_path = os.path.join(PTRS_DIR, ptr_name(p["filename"]))
        with open(ptr_path, "w") as f:
            json.dump(p, f, indent=2)

    workers = min(config["workers"], len(chosen))
    print(f"Pulling {len(chosen)} file(s) with {workers} worker(s)...")

    global _board
    _board = _Board(workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_pull_file, p, config) for p in chosen]
        for fut in as_completed(futures):
            dest, pulled = fut.result()
            if pulled:
                tprint(f"Pulled: {dest}")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

USAGE = """Usage:
  upload.py push <file|dir>   Upload, write pointers, optionally delete local
  upload.py pull <name>       Download file described by .ptrs/<name>[.ptr]
  upload.py ls                Browse remote index and pull selected files
  upload.py clean             Remove local files that are safely backed up remotely
"""


def cmd_clean(config):
    if not os.path.exists(PTRS_DIR):
        sys.exit("No .ptrs/ directory found.")

    ptr_files = [f for f in os.listdir(PTRS_DIR) if f.endswith(".ptr")]
    if not ptr_files:
        print("No pointers found.")
        return

    # fetch remote index once
    print("Fetching remote index...")
    remote_ptrs = {p["filename"]: p for p in _fetch_ptrs(config)}

    to_delete = []
    for fname in sorted(ptr_files):
        with open(os.path.join(PTRS_DIR, fname)) as f:
            ptr = json.load(f)

        name     = ptr["filename"]
        local    = os.path.join(PROJECT_ROOT, ptr["original_path"])
        local_hash  = ptr["hash"]
        local_mtime = ptr.get("mtime")

        if not os.path.exists(local):
            continue  # already gone

        # verify pointer is fresh (mtime matches local file)
        actual_mtime = os.path.getmtime(local)
        if local_mtime is None or abs(actual_mtime - local_mtime) > 1:
            local_ts = datetime.fromtimestamp(actual_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            ptr_ts   = datetime.fromtimestamp(local_mtime,  timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if local_mtime else "unknown"
            print(f"  SKIP {name}: local file mtime ({local_ts}) doesn't match pointer ({ptr_ts}) — run push first.")
            continue

        # verify remote has the same hash
        remote = remote_ptrs.get(name)
        if not remote:
            print(f"  SKIP {name}: not found in remote index.")
            continue

        if remote["hash"] != local_hash:
            remote_ts = datetime.fromtimestamp(remote.get("mtime", 0), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            local_ts  = datetime.fromtimestamp(local_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"  SKIP {name}: hash mismatch — local ({local_hash[:12]}… {local_ts}) vs remote ({remote['hash'][:12]}… {remote_ts}).")
            continue

        size_mb = ptr["size"] / 1_048_576
        mtime_ts = datetime.fromtimestamp(local_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        to_delete.append((local, name, size_mb, mtime_ts))

    if not to_delete:
        print("Nothing to clean — no locally present files are verified on remote.")
        return

    print(f"\nFiles safe to delete ({len(to_delete)}):")
    total_mb = 0.0
    for local, name, size_mb, mtime_ts in to_delete:
        print(f"  {name:<44} {size_mb:8.1f} MB   backed up {mtime_ts}")
        total_mb += size_mb
    print(f"\n  Total: {total_mb:.1f} MB")

    ans = input("\nDelete all? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    for local, _, _, _ in to_delete:
        os.remove(local)
        print(f"  Deleted {local}")

    print("Done.")


def main():
    if len(sys.argv) < 2:
        sys.exit(USAGE)

    cmd = sys.argv[1]
    config = load_config()

    if cmd == "push":
        if len(sys.argv) < 3:
            sys.exit("Usage: upload.py push <file|dir>")
        cmd_push(sys.argv[2], config)

    elif cmd == "pull":
        if len(sys.argv) != 3:
            sys.exit("Usage: upload.py pull <name>")
        cmd_pull(sys.argv[2], config)

    elif cmd == "ls":
        cmd_ls(config)

    elif cmd == "clean":
        cmd_clean(config)

    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
