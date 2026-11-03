import hashlib
import http.server
import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from serving.baseline.model_snapshot import (
    blob_lock,
    cache_paths,
    ensure_file,
    load_plan,
    verify_file,
)


def item(data, algorithm="sha256"):
    hashed = data if algorithm == "sha256" else f"blob {len(data)}\0".encode() + data
    digest = (hashlib.sha256 if algorithm == "sha256" else hashlib.sha1)(hashed).hexdigest()
    return {
        "name": "config.json",
        "bytes": len(data),
        "algorithm": algorithm,
        "upstream_hash": digest,
        "blob": digest,
        "url": "https://huggingface.co/Qwen/Qwen3-8B/resolve/pinned/config.json",
    }


@pytest.fixture
def http_bytes(monkeypatch):
    # Download tests must not depend on how full the real project directory is;
    # the 14 GB high-water rule itself is covered by test_resource_guard_* below.
    import serving.baseline.model_snapshot as model

    measure = model.canonical_bytes
    monkeypatch.setattr(
        model, "canonical_bytes", lambda root, *a: 0 if root == model.ROOT else measure(root, *a)
    )
    state = {"body": b"model-fixture", "declared": len(b"model-fixture"), "requests": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"] += 1
            if self.path.startswith("/redirect"):
                self.send_response(302)
                self.send_header("Location", state.get("location", "/fixture"))
                # No body is sent: following must close, never drain this response.
                self.send_header("Content-Length", "1000000000")
                self.end_headers()
                self.close_connection = False
                return
            self.send_response(200)
            self.send_header("Content-Length", str(state["declared"]))
            self.end_headers()
            self.wfile.write(state["body"])

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_port}/fixture"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_pinned_plan_rejects_any_identity_or_file_change(tmp_path):
    plan = load_plan(Path(__file__).with_name("model-files.json"))
    assert plan["total_bytes"] == 16_397_459_696
    changed = tmp_path / "changed.json"
    plan["files"][0]["bytes"] += 1
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="pinned"):
        load_plan(changed)


def test_git_blob_hash_and_actual_sha256_are_both_checked(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"data")
    proof = verify_file(path, item(b"data", "git-sha1"))
    assert proof["sha256"] == hashlib.sha256(b"data").hexdigest()
    path.write_bytes(b"evil")
    with pytest.raises(ValueError, match="hash"):
        verify_file(path, item(b"data", "git-sha1"))


def test_real_http_publishes_standard_layout_and_reuses_without_network(tmp_path, http_bytes):
    state, url = http_bytes
    expected = item(state["body"])
    first = ensure_file(expected, tmp_path, download=True, test_url=url)
    blob, pointer, lock = cache_paths(tmp_path, expected)
    assert pointer.is_symlink() and pointer.resolve() == blob
    assert "/hub/.locks/models--Qwen--Qwen3-8B/" in str(lock)
    assert first["downloaded_bytes"] == len(state["body"])
    second = ensure_file(expected, tmp_path, download=True, test_url=url)
    assert second["downloaded_bytes"] == 0 and state["requests"] == 1
    assert not list(blob.parent.glob("*.incomplete"))


def test_wrong_hash_preserves_owned_partial_without_publishing(tmp_path, http_bytes):
    state, url = http_bytes
    expected = item(b"x" * len(state["body"]))
    with pytest.raises(ValueError, match="hash"):
        ensure_file(expected, tmp_path, download=True, test_url=url)
    blob, pointer, _ = cache_paths(tmp_path, expected)
    assert not blob.exists() and not os.path.lexists(pointer)
    partials = list(blob.parent.glob("*.incomplete"))
    assert len(partials) == 1 and partials[0].read_bytes() == state["body"]
    with pytest.raises(ValueError, match="partial"):
        ensure_file(expected, tmp_path, download=True, test_url=url)
    assert state["requests"] == 1


def test_size_bound_and_existing_blob_or_snapshot_are_never_overwritten(tmp_path, http_bytes):
    state, url = http_bytes
    expected = item(state["body"])
    state["declared"] += 1
    with pytest.raises(ValueError, match="size"):
        ensure_file(expected, tmp_path, download=True, test_url=url)
    blob, pointer, _ = cache_paths(tmp_path, expected)
    blob.write_bytes(b"existing")
    with pytest.raises(ValueError, match="size"):
        ensure_file(expected, tmp_path, download=True, test_url=url)
    assert blob.read_bytes() == b"existing"
    blob.unlink()
    blob.write_bytes(state["body"])
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.symlink_to("wrong-existing-target")
    with pytest.raises(ValueError, match="snapshot"):
        ensure_file(expected, tmp_path, download=True, test_url=url)
    assert os.readlink(pointer) == "wrong-existing-target"


def test_stdlib_lock_blocks_installed_filelock_in_another_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
    lock = tmp_path / "same.lock"
    code = (
        "import sys; sys.path.insert(0,sys.argv[1]); from filelock import FileLock,Timeout; "
        "lock=FileLock(sys.argv[2],timeout=.15); "
        "lock.acquire(); print('acquired',flush=True); lock.release()"
    )
    command = [
        str(root / ".venv-ml/bin/python"),
        "-B",
        "-S",
        "-c",
        code,
        str(root / ".venv-ml/lib/python3.12/site-packages"),
        str(lock),
    ]
    with blob_lock(lock, timeout=1):
        blocked = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)
        assert blocked.returncode != 0 and "filelock._error.Timeout" in blocked.stderr
    acquired = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)
    assert acquired.returncode == 0 and acquired.stdout.strip() == "acquired"


def test_installed_filelock_blocks_stdlib_lock_in_reverse_direction(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "same.lock"
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);from filelock import FileLock;"
        "lock=FileLock(sys.argv[2],timeout=1);lock.acquire();print('held',flush=True);"
        "sys.stdin.readline();lock.release()"
    )
    child = subprocess.Popen(
        [
            str(root / ".venv-ml/bin/python"),
            "-B",
            "-S",
            "-c",
            code,
            str(root / ".venv-ml/lib/python3.12/site-packages"),
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError), blob_lock(path, timeout=0.1):
            pytest.fail("cross-process lock unexpectedly acquired")
        child.stdin.write("release\n")
        child.stdin.flush()
        assert child.wait(timeout=3) == 0
        with blob_lock(path, timeout=1):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        child.stdin.close()
        child.stdout.close()


def test_truncated_response_and_expired_deadline_never_publish(tmp_path, http_bytes):
    import time

    state, url = http_bytes
    expected = item(state["body"])
    with pytest.raises(TimeoutError, match="deadline"):
        ensure_file(expected, tmp_path, download=True, test_url=url, deadline=time.monotonic() - 1)
    assert state["requests"] == 0
    state["body"] = state["body"][:-1]
    with pytest.raises(ValueError, match="truncated"):
        ensure_file(expected, tmp_path, download=True, test_url=url)
    blob, pointer, _ = cache_paths(tmp_path, expected)
    assert not blob.exists() and not os.path.lexists(pointer)
    assert sum(p.stat().st_size for p in blob.parent.glob("*.incomplete")) == len(state["body"])


def test_resource_guard_rejects_concurrent_shared_growth_and_full_download_reserve(
    tmp_path, monkeypatch
):
    from serving.baseline import model_snapshot as model

    counts = {tmp_path: 79_000_000_000, model.ROOT: 13_000_000_000}
    monkeypatch.setattr(model, "project_bytes", lambda path: counts[path])
    monkeypatch.setattr(
        model.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 100_000_000_000})()
    )
    with pytest.raises(RuntimeError, match="high-water"):
        model.resource_sample(tmp_path, 1_000_000_001)
    counts[tmp_path] = 10_000_000_000
    monkeypatch.setattr(
        model.shutil,
        "disk_usage",
        lambda path: type("Disk", (), {"free": 50_000_000_000 + model.TOTAL_BYTES - 1})(),
    )
    with pytest.raises(RuntimeError, match="high-water"):
        model.resource_sample(tmp_path, model.TOTAL_BYTES)


def test_redirect_closes_unbounded_body_and_limits_hops(tmp_path, http_bytes):
    import time

    state, url = http_bytes
    result = ensure_file(
        item(state["body"]),
        tmp_path / "ok",
        download=True,
        test_url=url.replace("/fixture", "/redirect"),
        observe=lambda remaining: None,
        deadline=time.monotonic() + 2,
    )
    assert result["response"]["path"] == "/fixture" and state["requests"] == 2
    state["location"] = "/redirect-loop"
    with pytest.raises(ValueError, match="redirect bound"):
        ensure_file(
            item(state["body"]),
            tmp_path / "loop",
            download=True,
            test_url=url.replace("/fixture", "/redirect"),
            observe=lambda remaining: None,
        )
    assert state["requests"] == 8


def test_redirect_rejects_other_origin_without_sending_request(tmp_path, http_bytes):
    state, url = http_bytes
    state["location"] = "http://127.0.0.1:1/unreviewed"
    with pytest.raises(ValueError, match="unreviewed acquisition redirect"):
        ensure_file(
            item(state["body"]),
            tmp_path,
            download=True,
            test_url=url.replace("/fixture", "/redirect"),
            observe=lambda remaining: None,
        )
    assert state["requests"] == 1


def test_hash_finishing_after_deadline_cannot_publish(tmp_path, http_bytes, monkeypatch):
    from serving.baseline import model_snapshot as model

    state, url = http_bytes
    expected = item(state["body"])
    clock = [model.time.monotonic()]
    deadline = clock[0] + 10
    original_verify = model.verify_file

    def late_hash(*args):
        result = original_verify(*args)
        clock[0] = deadline + 1
        return result

    monkeypatch.setattr(model.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(model, "verify_file", late_hash)
    with pytest.raises(TimeoutError, match="deadline"):
        model.ensure_file(
            expected,
            tmp_path,
            download=True,
            test_url=url,
            observe=lambda remaining: None,
            deadline=deadline,
        )
    blob, pointer, _ = cache_paths(tmp_path, expected)
    assert not blob.exists() and not os.path.lexists(pointer)
    assert len(list(blob.parent.glob("*.incomplete"))) == 1


@pytest.mark.parametrize("failures,expected_calls", [(1, 2), (2, 3), (3, 3)])
def test_canonical_scan_restarts_whole_walk_only_three_times(
    tmp_path, monkeypatch, failures, expected_calls
):
    from serving.baseline import model_snapshot as model

    calls, events = [], []

    def unstable(root):
        calls.append(root)
        if len(calls) <= failures:
            raise FileNotFoundError("transient directory disappeared")
        return 123

    monkeypatch.setattr(model, "project_bytes", unstable)
    if failures == 3:
        with pytest.raises(FileNotFoundError):
            model.canonical_bytes(tmp_path, events)
    else:
        assert model.canonical_bytes(tmp_path, events) == 123
    assert calls == [tmp_path] * expected_calls
    assert len(events) == failures and events[-1]["attempt"] == failures


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("other")])
def test_canonical_scan_does_not_swallow_other_errors(tmp_path, monkeypatch, error):
    from serving.baseline import model_snapshot as model

    calls = []

    def unreadable(root):
        calls.append(root)
        raise error

    monkeypatch.setattr(model, "project_bytes", unreadable)
    with pytest.raises(type(error), match=str(error)):
        model.canonical_bytes(tmp_path)
    assert calls == [tmp_path]


def test_canonical_rescan_preserves_actual_error_filenames(tmp_path, monkeypatch):
    from serving.baseline import model_snapshot as model

    error = FileNotFoundError(2, "missing", "first", None, "second")
    outcomes = iter([error, 10])

    def scan(root):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(model, "project_bytes", scan)
    events = []
    assert model.canonical_bytes(tmp_path, events) == 10
    assert events[0]["errno"] == 2
    assert events[0]["filename"] == "first" and events[0]["filename2"] == "second"
    assert events[0]["error"] == repr(error) and events[0]["message"] == str(error)
