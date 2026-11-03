"""Bounded subprocess execution, retaining raw argv, output and exit status."""

import os
import selectors
import signal
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass
class Result:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def as_dict(self):
        return asdict(self)


class Started:
    def __init__(self, argv):
        self.argv = argv
        self.process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        self.prefix = b""

    def terminate(self):
        # Kill the whole group even if the leader has already exited.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if not sel.select(max(0, deadline - time.monotonic())):
                    break
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    break
                self.prefix += chunk
                if b"CRUCIBLE_READY\n" in self.prefix:
                    return True
        return False

    def finish(self, timeout):
        timed_out = False
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.terminate()
            stdout, stderr = self.process.communicate()
        finally:
            self.terminate()
        return Result(
            self.argv,
            124 if timed_out else self.process.returncode,
            (self.prefix + stdout).decode(errors="replace"),
            stderr.decode(errors="replace"),
            timed_out,
        )


class CommandRunner:
    def start(self, argv):
        return Started(argv)

    def run(self, argv, timeout):
        try:
            return self.start(argv).finish(timeout)
        except OSError as exc:
            return Result(argv, 127, "", str(exc))
