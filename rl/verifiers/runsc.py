"""Direct OCI gVisor adapter. Never uses `runsc do`, Docker, or host Python.

An operator must supply a functioning execution entrypoint with delegated cgroups
and namespace privileges. The default host currently lacks these prerequisites.
No flags claiming readiness substitute for runtime completion/cgroup checks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rl.verifiers.code import encode
from rl.verifiers.types import RunResult


@dataclass(frozen=True)
class RunscConfig:
    runsc: Path
    rootfs: Path
    manifest: Path
    manifest_sha256: str
    work_dir: Path
    cgroup_parent: Path
    # Trusted, explicitly provisioned launcher only. Never derived from candidate.
    entrypoint: tuple[str, ...] = ()


def verify_rootfs(rootfs: Path, manifest: Path, expected_sha256: str):
    if rootfs.is_symlink() or not rootfs.is_dir() or rootfs.resolve() == Path("/"):
        raise ValueError("dedicated_rootfs_required")
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("manifest_digest_mismatch")
    specification = json.loads(raw)
    files = specification["files"]
    actual = {}
    for path in rootfs.rglob("*"):
        relative = str(path.relative_to(rootfs))
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve().is_relative_to(rootfs.resolve()):
                raise ValueError("escaping_symlink")
            actual[relative] = {"link": target}
        elif path.is_file():
            actual[relative] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mode": path.stat().st_mode & 0o777,
            }
        elif not path.is_dir():
            raise ValueError("special_rootfs_file")
    if actual != files:
        raise ValueError("rootfs_contents_mismatch")
    required = ("usr/bin/python3", "verifier/_candidate.py", "tests/public.json")
    if any(name not in files for name in required):
        raise ValueError("missing_runner_file")
    bootstrap_hash = hashlib.sha256(
        Path(__file__).with_name("_candidate.py").read_bytes()
    ).hexdigest()
    if files["verifier/_candidate.py"].get("sha256") != bootstrap_hash:
        raise ValueError("bootstrap_digest_mismatch")
    return specification


def build_spec(
    config: RunscConfig, request: str, public_file: Path, container_id: str, timeout_s: float
):
    return {
        "ociVersion": "1.0.2",
        "root": {"path": str(config.rootfs.resolve()), "readonly": True},
        "process": {
            "terminal": False,
            "user": {"uid": 65534, "gid": 65534},
            "args": ["/usr/bin/python3", "-I", "-S", "/verifier/_candidate.py", request],
            "cwd": "/tmp",
            "env": ["PATH=/usr/bin", "PYTHONDONTWRITEBYTECODE=1"],
            "noNewPrivileges": True,
            "capabilities": {
                key: [] for key in ("bounding", "effective", "inheritable", "permitted", "ambient")
            },
            "rlimits": [
                {
                    "type": "RLIMIT_CPU",
                    "soft": max(1, math.ceil(timeout_s)),
                    "hard": max(1, math.ceil(timeout_s)),
                },
                {"type": "RLIMIT_AS", "soft": 512 * 1024**2, "hard": 512 * 1024**2},
                {"type": "RLIMIT_FSIZE", "soft": 16 * 1024**2, "hard": 16 * 1024**2},
                {"type": "RLIMIT_NOFILE", "soft": 64, "hard": 64},
                {"type": "RLIMIT_NPROC", "soft": 128, "hard": 128},
                {"type": "RLIMIT_CORE", "soft": 0, "hard": 0},
            ],
        },
        "mounts": [
            {
                "destination": "/proc",
                "type": "proc",
                "source": "proc",
                "options": ["nosuid", "nodev", "noexec", "ro"],
            },
            {
                "destination": "/tmp",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "noexec", "size=16777216", "mode=1777"],
            },
            {
                "destination": "/tests/public.json",
                "type": "bind",
                "source": str(public_file.resolve()),
                "options": ["bind", "ro", "nosuid", "nodev", "noexec"],
            },
        ],
        "linux": {
            "namespaces": [{"type": name} for name in ("pid", "mount", "network", "ipc", "uts")],
            "cgroupsPath": "/"
            + str((config.cgroup_parent.resolve() / container_id).relative_to("/sys/fs/cgroup")),
            "resources": {
                "memory": {"limit": 512 * 1024**2},
                "cpu": {"quota": 100000, "period": 100000},
                "pids": {"limit": 128},
            },
        },
    }


class LocalRunsc:
    def __init__(self, config: RunscConfig):
        self.config = config
        self.last_bundle = None

    def _preflight(self):
        config = self.config
        if not config.runsc.is_file() or not os.access(config.runsc, os.X_OK):
            raise ValueError("runsc_missing")
        if not config.cgroup_parent.resolve().is_relative_to(Path("/sys/fs/cgroup")):
            raise ValueError("delegated_cgroup_required")
        controllers = (config.cgroup_parent / "cgroup.subtree_control").read_text().split()
        if not {"cpu", "memory", "pids"}.issubset(controllers) or not os.access(
            config.cgroup_parent, os.W_OK
        ):
            raise ValueError("delegated_cgroup_required")
        verify_rootfs(config.rootfs, config.manifest, config.manifest_sha256)

    def run(self, solution, case_input, public_tests, timeout_s):
        try:
            if not 0 < timeout_s <= 300:
                raise ValueError("invalid_timeout")
            deadline = time.monotonic() + timeout_s
            self._preflight()
            request = base64.b64encode(
                encode({"solution": solution, "input": case_input}).encode()
            ).decode()
            if len(public_tests.encode()) > 65536:
                raise ValueError("public_tests_limit")
        except (OSError, ValueError, KeyError, TypeError):
            return RunResult(error="local_runtime_prerequisites_missing", cleanup_confirmed=True)
        config = self.config
        config.work_dir.mkdir(parents=True, exist_ok=True)
        bundle = Path(tempfile.mkdtemp(prefix="crucible-verifier-", dir=config.work_dir))
        self.last_bundle = bundle
        public = bundle / "public.json"
        public.write_text(public_tests)
        public.chmod(0o444)
        container_id = "crucible-" + uuid.uuid4().hex
        state_root = bundle / "state"
        state_root.mkdir(mode=0o700)
        spec = build_spec(config, request, public, container_id, timeout_s)
        (bundle / "config.json").write_text(json.dumps(spec))
        command = [
            *config.entrypoint,
            str(config.runsc.resolve()),
            "--root=" + str(state_root),
            "--network=none",
            "--platform=systrap",
        ]
        execution_deadline = deadline - min(1.0, timeout_s / 5)
        proc, cleaned, timed_out, truncated = None, False, False, False
        output = {"stdout": bytearray(), "stderr": bytearray()}
        error, exit_code = "", None
        try:
            if time.monotonic() >= execution_deadline:
                raise subprocess.TimeoutExpired(command, timeout_s)
            proc = subprocess.Popen(
                command + ["run", "--bundle=" + str(bundle), container_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
            with selectors.DefaultSelector() as selector:
                for stream, label in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, label)
                while selector.get_map():
                    remaining = execution_deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    for key, _ in selector.select(min(remaining, 0.05)):
                        chunk = os.read(key.fileobj.fileno(), 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            output[key.data].extend(chunk)
                            if len(output[key.data]) > 65536:
                                del output[key.data][65536:]
                                truncated = True
                    if truncated:
                        break
                if not timed_out and not truncated:
                    exit_code = proc.wait(timeout=max(0.001, execution_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError:
            error = "runtime_launch_failed"
        finally:
            # Kill all guest processes, then destroy runtime state. Never infer
            # cleanup from Python client timeout, parent exit, or a success string.
            for action in (
                ["kill", "--all", container_id, "KILL"],
                ["delete", "--force", container_id],
            ):
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    result = subprocess.run(
                        command + action,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=remaining,
                        check=False,
                        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                    )
                    if action[0] == "delete":
                        cgroup = config.cgroup_parent / container_id
                        empty = (
                            not cgroup.exists()
                            or "populated 0" in (cgroup / "cgroup.events").read_text().splitlines()
                        )
                        cleaned = result.returncode == 0 and empty
                except (OSError, subprocess.TimeoutExpired):
                    cleaned = False
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    cleaned = False
                proc.stdout.close()
                proc.stderr.close()
            if cleaned:
                shutil.rmtree(bundle)
        try:
            stdout = output["stdout"].decode("utf-8", errors="strict")
            stderr = output["stderr"].decode("utf-8", errors="strict")
        except UnicodeError:
            stdout, stderr, error = "", "", "invalid_utf8_output"
        return RunResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            completed=exit_code is not None,
            cleanup_confirmed=cleaned,
            timed_out=timed_out,
            output_truncated=truncated,
            error=error,
            run_id=container_id,
        )
