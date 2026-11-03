import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KILL = ROOT / "serving/faults/kill_decode.sh"
NIC = ROOT / "serving/faults/nic_down.sh"
TRAIN = ROOT / "spec/train_eagle3.sh"


def fake(argv_tail):
    # Its own process group, like a replica spawned with start_new_session=True.
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", *argv_tail], start_new_session=True
    )


def run(*args, env=None):
    return subprocess.run(["bash", *map(str, args)], capture_output=True, text=True, env=env, timeout=30, check=False)


def test_kill_decode_kills_a_vllm_server_group():
    proc = fake(["-m", "vllm.entrypoints.openai.api_server", "--port", "1"])
    try:
        result = run(KILL, proc.pid)
        assert result.returncode == 0, result.stderr
        assert float(result.stdout) > 0
        assert proc.wait(10) == -9
    finally:
        proc.kill()


def test_kill_decode_refuses_other_processes_and_bad_input():
    proc = fake(["not-a-model-server"])
    try:
        result = run(KILL, proc.pid)
        assert result.returncode == 3 and "refusing" in result.stderr
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait(10)
    assert run(KILL, "1").returncode == 2
    assert run(KILL, "abc").returncode == 2


def test_nic_down_is_paused_without_multinode_approval():
    env = {k: v for k, v in os.environ.items() if k != "CRUCIBLE_MULTINODE_APPROVED"}
    result = run(NIC, "node-a", "eth0", "10", env=env)
    assert result.returncode == 3 and "paused" in result.stderr


def test_train_eagle3_is_blocked_without_the_pinned_environment():
    if (ROOT / ".venv-specforge").exists():
        return  # the blocked path only applies before the environment exists
    result = run(TRAIN, "config.yaml")
    assert result.returncode == 3 and "blocked" in result.stderr
