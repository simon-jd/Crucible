#!/usr/bin/env python3
"""Deadline-bound owned process groups and token-scoped stop/verification.

Absolute deadlines assume synchronized site clocks. No daemon/service changes.
"""

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def stat(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()


def identity(pid):
    return stat(pid)[19]


def alive(pid, starttime):
    try:
        fields = stat(pid)
        return fields[19] == starttime and fields[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def members(pgid):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                fields = stat(int(path.name))
                if int(fields[2]) == pgid and fields[0] != "Z":
                    result.append({"pid": int(path.name), "starttime": fields[19]})
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return result


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stop(path):
    """Tombstone prevents a delayed transport launch after cleanup has returned."""
    done = path.with_suffix(".done.json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path.with_suffix(".stop").touch()
        state = json.loads(path.read_text()) if path.exists() else None
        receipt = json.loads(done.read_text()) if done.exists() else None
        # Never signal a reused PID or another task's process group.
        if state and alive(state["pid"], state["starttime"]):
            if os.getpgid(state["pid"]) != state["pid"]:
                raise ValueError("owned child process group changed")
            os.killpg(state["pid"], signal.SIGKILL)
        # Killing the owned child wakes its worker's bounded wait. Let that
        # worker finish its finally block and receipt without interrupting it.
    end = time.monotonic() + 2
    while time.monotonic() < end:
        if done.exists():
            receipt = json.loads(done.read_text())
        if receipt:
            targets = receipt.get("members", []) + [
                {"pid": receipt["worker_pid"], "starttime": receipt["worker_starttime"]}
            ]
            verified = receipt.get("verified") is True and not any(
                alive(item["pid"], item["starttime"]) for item in targets
            )
            if verified:
                return {"verified": True, "state": "exited", "receipt": receipt}
        elif state is None:
            return {"verified": True, "state": "not-started-tombstoned"}
        time.sleep(0.01)
    return {"verified": False, "state": "unverified", "receipt": receipt}


def listening(pid, port):
    inodes = set()
    try:
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith("socket:["):
                    inodes.add(target[8:-1])
            except FileNotFoundError:
                continue
        for table in ["tcp", "tcp6"]:
            for line in Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]:
                columns = line.split()
                if (
                    int(columns[1].split(":")[1], 16) == port
                    and columns[3] == "0A"
                    and columns[9] in inodes
                ):
                    return True
    except FileNotFoundError:
        pass
    return False


def main():
    cleaning = False

    def interrupted(signum, frame):
        if not cleaning:
            raise SystemExit(128 + signum)

    for sig in [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "stop"])
    parser.add_argument("--token", required=True)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--ready-port", type=int)
    parser.add_argument("--state-dir", default=f"/tmp/crucible-burnin-{os.getuid()}")
    arguments = sys.argv[1:]
    command = []
    if "--" in arguments:
        split = arguments.index("--")
        command, arguments = arguments[split + 1 :], arguments[:split]
    args = parser.parse_args(arguments)
    if not re.fullmatch(r"[0-9a-f]{32}", args.token):
        parser.error("invalid cleanup token")
    state_dir = Path(args.state_dir)
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state_dir.is_symlink() or state_dir.stat().st_uid != os.getuid():
        parser.error("unsafe state directory")
    if state_dir.stat().st_mode & 0o077:
        parser.error("state directory must be private (0700)")
    path = state_dir / (args.token + ".json")
    if args.action == "stop":
        try:
            result = stop(path)
        except (OSError, ValueError, KeyError) as exc:
            result = {"verified": False, "error": str(exc)}
        print(json.dumps(result | {"token": args.token}))
        return 0 if result["verified"] else 1
    if not command or args.deadline_epoch is None or not math.isfinite(args.deadline_epoch):
        parser.error("command and finite absolute --deadline-epoch required")
    if time.time() >= args.deadline_epoch:
        return 124
    child, state = None, None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if path.with_suffix(".stop").exists() or path.exists():
                    raise ValueError("token stopped or already owned")
                if time.time() >= args.deadline_epoch:
                    return 124
                try:
                    child = subprocess.Popen(
                        command, stdout=stdout, stderr=stderr, start_new_session=True
                    )
                    state = {
                        "pid": child.pid,
                        "starttime": identity(child.pid),
                        "worker_pid": os.getpid(),
                        "worker_starttime": identity(os.getpid()),
                        "deadline_epoch": args.deadline_epoch,
                    }
                    write_json(path, state)
                except BaseException:
                    # Keep launch/stop exclusion until an unregistered child is
                    # dead; otherwise stop could falsely attest "not started".
                    cleaning = True
                    if child is not None:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                    raise
            if args.ready_port:
                while child.poll() is None and time.time() < args.deadline_epoch:
                    if listening(child.pid, args.ready_port):
                        print("CRUCIBLE_READY", flush=True)
                        break
                    time.sleep(min(0.02, max(0, args.deadline_epoch - time.time())))
            code = 124
            while time.time() < args.deadline_epoch:
                try:
                    code = child.wait(timeout=min(0.05, max(0, args.deadline_epoch - time.time())))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            cleaning = True
            if child is not None:
                owned = members(child.pid)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                if state is not None:
                    grace = time.monotonic() + 1
                    while members(child.pid) and time.monotonic() < grace:
                        time.sleep(0.01)
                    receipt = state | {"members": owned, "verified": not members(child.pid)}
                    write_json(path.with_suffix(".done.json"), receipt)
                    path.unlink(missing_ok=True)
            stdout.seek(0)
            stderr.seek(0)
            sys.stdout.flush()
            sys.stdout.buffer.write(stdout.read())
            sys.stderr.buffer.write(stderr.read())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
