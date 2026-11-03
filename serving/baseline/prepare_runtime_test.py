import hashlib
import os

import pytest

from serving.baseline.prepare_runtime import CAP, audit_hardlinks, check_peak, verify_record


def test_record_verification_detects_same_size_modification(tmp_path):
    import base64

    data = tmp_path / "payload.py"
    data.write_bytes(b"good")
    digest = base64.urlsafe_b64encode(hashlib.sha256(b"good").digest()).rstrip(b"=").decode()
    record = tmp_path / "RECORD"
    record.write_text(f"payload.py,sha256={digest},4\nRECORD,,\n")
    assert verify_record(tmp_path, record, tmp_path)["hashed_files"] == 1
    data.write_bytes(b"evil")
    with pytest.raises(ValueError, match="hash"):
        verify_record(tmp_path, record, tmp_path)


def test_record_cannot_escape_owned_base(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    record = site / "RECORD"
    record.write_text("../../outside,sha256=bad,1\n")
    with pytest.raises(ValueError, match="escapes"):
        verify_record(site, record, tmp_path)


def test_large_payload_requires_same_inode_and_rejects_copy(tmp_path):
    cache, target = tmp_path / "cache", tmp_path / "target"
    cache.mkdir()
    target.mkdir()
    src = cache / "blob"
    src.write_bytes(b"x" * (1 << 20))
    dst = target / "blob"
    os.link(src, dst)
    proof = audit_hardlinks(cache, target, ["blob"])
    assert proof["linked_payload_bytes"] == 1 << 20
    dst.unlink()
    dst.write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="inode"):
        audit_hardlinks(cache, target, ["blob"])


def test_peak_rejects_before_mutation_with_cleanup_headroom():
    # Project share; the cleanup reserve must fit too.
    assert check_peak(CAP - 3_600_000_000, 3_550_000_000, 50_000_000) is None
    with pytest.raises(ValueError, match="cap"):
        check_peak(CAP - 3_600_000_000, 3_550_000_001, 50_000_000)


def test_bounded_real_http_download_rejects_wrong_hash_without_publishing(tmp_path):
    import http.server
    import threading

    from serving.baseline.prepare_runtime import download_wheel

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"data")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        expected = {
            "url": f"http://127.0.0.1:{server.server_port}/test.whl",
            "compressed_bytes": 4,
            "sha256": "0" * 64,
        }
        dest = tmp_path / "test.whl"
        with pytest.raises(ValueError, match="hash"):
            download_wheel(expected, dest, test_loopback=True)
        assert not dest.exists()
        assert dest.with_suffix(".whl.partial").read_bytes() == b"data"
        dest.with_suffix(".whl.partial").unlink()
        expected["sha256"] = hashlib.sha256(b"data").hexdigest()
        download_wheel(expected, dest, test_loopback=True)
        assert dest.read_bytes() == b"data"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_lock_rejects_unreviewed_host_and_duplicate_wheel():
    from serving.baseline.prepare_runtime import validate_lock

    lock = {
        "schema": 1,
        "packages": [],
        "wheels": [
            {
                "name": "x",
                "version": "1",
                "filename": "x.whl",
                "url": "https://evil.invalid/x.whl",
                "sha256": "0" * 64,
                "compressed_bytes": 4,
                "expanded_bytes": 4,
            }
        ],
    }
    with pytest.raises(ValueError, match="host"):
        validate_lock(lock)
    lock["wheels"][0]["url"] = "https://files.pythonhosted.org/x.whl"
    lock["wheels"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_lock(lock)


def test_rewritten_large_record_is_metadata_but_large_license_is_payload(tmp_path):
    import zipfile

    from serving.baseline.prepare_runtime import sha256, verify_wheel

    wheel = tmp_path / "x.whl"
    members = {
        "x-1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: x\nVersion: 1\n",
        "x-1.dist-info/RECORD": b"x" * (1 << 20),
        "x-1.dist-info/licenses/LICENSE": b"l" * (1 << 20),
        "other-1.dist-info/RECORD": b"r" * (1 << 20),
        "x/payload.bin": b"p" * (1 << 20),
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    expected = {
        "compressed_bytes": wheel.stat().st_size,
        "expanded_bytes": sum(map(len, members.values())),
        "sha256": sha256(wheel),
    }
    assert verify_wheel(wheel, expected) == [
        "x-1.dist-info/licenses/LICENSE",
        "other-1.dist-info/RECORD",
        "x/payload.bin",
    ]


def test_explicit_continuation_requires_exact_prefix_and_installed_metadata(tmp_path):
    import base64

    from serving.baseline.prepare_runtime import sha256, verify_completed_prefix

    metadata = tmp_path / "x-1.dist-info"
    metadata.mkdir()
    content = b"Metadata-Version: 2.1\nName: x\nVersion: 1\n"
    (metadata / "METADATA").write_bytes(content)
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    record = metadata / "RECORD"
    record.write_text(
        f"x-1.dist-info/METADATA,sha256={encoded},{len(content)}\nx-1.dist-info/RECORD,,\n"
    )
    wheel = {
        "name": "x",
        "sha256": "1" * 64,
        "compressed_bytes": 10,
        "metadata_sha256": sha256(metadata / "METADATA"),
    }
    receipt = {
        "completed": [
            {
                "name": "x",
                "wheel_sha256": "1" * 64,
                "record": "x-1.dist-info/RECORD",
                "record_sha256": sha256(record),
            }
        ]
    }
    assert len(verify_completed_prefix([wheel], receipt, tmp_path)) == 1
    receipt["completed"][0]["name"] = "another"
    with pytest.raises(ValueError, match="prefix"):
        verify_completed_prefix([wheel], receipt, tmp_path)
    receipt["completed"][0]["name"] = "x"
    (metadata / "METADATA").write_bytes(content.replace(b"x", b"z"))
    with pytest.raises(ValueError, match="metadata"):
        verify_completed_prefix([wheel], receipt, tmp_path)


def test_only_pinned_script_relocation_has_content_verified_copy_exception(tmp_path, monkeypatch):
    import zipfile

    from serving.baseline import prepare_runtime as preparation
    from serving.baseline.prepare_runtime import sha256, verify_wheel

    wheel = tmp_path / "script.whl"
    members = {
        "py_spy-0.4.2.dist-info/METADATA": b"Metadata-Version: 2.1\nName: py-spy\nVersion: 0.4.2\n",
        preparation.COPIED_SCRIPT_MEMBER: b"\x7fELF" + b"x" * (preparation.COPIED_SCRIPT_BYTES - 4),
    }

    def prepare():
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in members.items():
                archive.writestr(name, value)
        return {
            "compressed_bytes": wheel.stat().st_size,
            "expanded_bytes": sum(map(len, members.values())),
            "sha256": sha256(wheel),
        }

    expected = prepare()
    with pytest.raises(ValueError, match="relocat"):
        verify_wheel(wheel, expected)
    # Fixture identity substitutes for the reviewed real wheel; content validation remains real.
    monkeypatch.setattr(preparation, "COPIED_SCRIPT_WHEEL", expected["sha256"])
    assert verify_wheel(wheel, expected) == ["bin/py-spy"]
    cache, target = tmp_path / "cache", tmp_path / "target"
    cached = cache / preparation.COPIED_SCRIPT_MEMBER
    installed = target / preparation.COPIED_SCRIPT_TARGET
    cached.parent.mkdir(parents=True)
    installed.parent.mkdir(parents=True)
    cached.write_bytes(members[preparation.COPIED_SCRIPT_MEMBER])
    installed.write_bytes(cached.read_bytes())
    proof = preparation.audit_copied_script(wheel, expected, cache, target)
    assert proof["shared_inode"] is False
    installed.write_bytes(b"bad" + installed.read_bytes()[3:])
    with pytest.raises(ValueError, match="content"):
        preparation.audit_copied_script(wheel, expected, cache, target)
    members["other-1.data/scripts/py-spy"] = members.pop(preparation.COPIED_SCRIPT_MEMBER)
    expected = prepare()
    monkeypatch.setattr(preparation, "COPIED_SCRIPT_WHEEL", expected["sha256"])
    with pytest.raises(ValueError, match="relocat"):
        verify_wheel(wheel, expected)


def test_pending_stage_requires_exact_receipt_and_unmodified_owned_contents(tmp_path):
    import json

    from serving.baseline.prepare_runtime import validate_pending_stage

    stage = tmp_path / "wheel-stage-18"
    stage.mkdir()
    expected = {"filename": "locked.whl", "sha256": "1" * 64}
    pending = {"stage": "wheel-stage-18", "wheel_sha256": "1" * 64}
    owner = {"lock_sha256": "lock", "wheel": "locked.whl"}
    (stage / "owner.json").write_text(json.dumps(owner))
    (stage / "locked.whl").write_bytes(b"fixture")
    (stage / "cache").mkdir()
    validate_pending_stage(stage, expected, pending, "lock")
    (stage / "cache" / "unexpected").write_bytes(b"x")
    with pytest.raises(ValueError, match="contents"):
        validate_pending_stage(stage, expected, pending, "lock")
