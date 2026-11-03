"""Explicit token cleanup; losing a local transport is never proof of remote exit."""

import json
import math
import time
import uuid

from cluster.health.nccl_matrix import ssh
from cluster.health.runner import Result


def remaining(deadline):
    value = deadline - time.time()
    if not math.isfinite(value) or value <= 0:
        raise ValueError("absolute operator/task deadline reached")
    return value


def stop_remote(inv, node, token, runner):
    argv = ssh(
        node,
        [
            "python3",
            inv["shared_root"] + "/cluster/health/remote_worker.py",
            "stop",
            "--token",
            token,
        ],
    )
    raw = None
    try:
        raw = runner.run(argv, 5)
        parsed = json.loads(raw.stdout)
        verified = (
            raw.returncode == 0 and parsed.get("token") == token and parsed.get("verified") is True
        )
        return {"token": token, "verified": verified, "command": raw.as_dict(), "report": parsed}
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record transport failure and clean up
        return {
            "token": token,
            "verified": False,
            "command": raw.as_dict()
            if raw is not None
            else Result(argv, 127, "", str(exc)).as_dict(),
            "error": str(exc),
        }


def run_remote(inv, node, command, runner, deadline_epoch):
    token = uuid.uuid4().hex
    argv = ssh(
        node,
        [
            "python3",
            inv["shared_root"] + "/cluster/health/remote_worker.py",
            "run",
            "--token",
            token,
            "--deadline-epoch",
            str(deadline_epoch),
            "--",
        ]
        + command,
    )
    result = {"passed": False, "commands": [], "cleanup": []}
    try:
        raw = runner.run(argv, remaining(deadline_epoch))
        result["commands"].append(raw.as_dict())
        result["stdout"] = raw.stdout
        result["passed"] = raw.returncode == 0
        if raw.returncode:
            result["error"] = "remote command failed or timed out"
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record transport failure and clean up
        result["cancelled"] = isinstance(exc, KeyboardInterrupt)
        result["error"] = str(exc) or type(exc).__name__
        result["commands"].append(
            Result(argv, 130 if result["cancelled"] else 127, "", result["error"]).as_dict()
        )
    finally:
        cleanup = stop_remote(inv, node, token, runner)
        result["cleanup"].append(cleanup)
        result["commands"].append(cleanup["command"])
        if not cleanup["verified"]:
            result.update(
                passed=False, error=(result.get("error", "") + "; token cleanup unverified")
            )
    return result
