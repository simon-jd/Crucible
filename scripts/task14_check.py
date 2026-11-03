"""Run a Task14 check with local Go caches and sampled resource evidence.

This is a measuring wrapper, not a process supervisor. It does not signal,
reparent, or clean up processes. Sampling is not a continuous strict peak.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.resource_preflight import GB, PROJECT_SHARE_BYTES, project_bytes


def tree_rss(root_pid):
    records = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            pages = int((proc / "statm").read_text().split()[1])
            records[int(proc.name)] = (int(fields[1]), pages * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    owned = {root_pid}
    while True:
        expanded = owned | {pid for pid, (parent, _) in records.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    return sum(records.get(pid, (0, 0))[1] for pid in owned)


def sampled_bytes(root):
    total, seen = 0, set()
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if path.is_symlink():
                    continue
                stat = path.stat()
            except FileNotFoundError:
                continue  # Compiler temp files may disappear during a sample.
            inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                total += stat.st_size
                seen.add(inode)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required")
    cache = ROOT / ".cache/task14"
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        GOCACHE=str(cache / "build"),
        GOPATH=str(cache / "go"),
        GOMODCACHE=str(cache / "mod"),
        GOTMPDIR=str(cache / "tmp"),
        TMPDIR=str(cache / "tmp"),
        GOENV="off",
        GOTOOLCHAIN="local",
        GOPROXY="off",
        GOMAXPROCS="2",
        PATH=str(ROOT / ".tools/go/bin") + os.pathsep + env.get("PATH", ""),
        TASK14_ROUTER_BINARY=str(cache / "bin/router"),
    )
    before = project_bytes(ROOT)
    if before + 700_000_000 > PROJECT_SHARE_BYTES:
        parser.error(f"700 MB reservation would exceed {PROJECT_SHARE_BYTES // GB} GB")
    peaks = {
        "sampled_tree_rss_bytes": 0,
        "sampled_task14_cache_bytes": 0,
        "sampled_tmp_bytes": 0,
        "samples": 0,
    }
    done = threading.Event()

    def sample():
        while True:
            peaks["sampled_tree_rss_bytes"] = max(
                peaks["sampled_tree_rss_bytes"], tree_rss(os.getpid())
            )
            peaks["sampled_task14_cache_bytes"] = max(
                peaks["sampled_task14_cache_bytes"], sampled_bytes(cache)
            )
            peaks["sampled_tmp_bytes"] = max(
                peaks["sampled_tmp_bytes"], sampled_bytes(cache / "tmp")
            )
            peaks["samples"] += 1
            if done.wait(0.25):
                return

    sampler = threading.Thread(target=sample)
    started = time.monotonic()
    sampler.start()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    finally:
        done.set()
        sampler.join()
    evidence = {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "project_bytes_before": before,
        "project_bytes_after": project_bytes(ROOT),
        "sampling": "0.25s requested interval plus scan time; transient peaks may be missed",
        "scope": "wrapper and descendants RSS; task14 caches/temp only; CPU-only",
        **peaks,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
