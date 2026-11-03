import hashlib
import json

from scripts import verify_hf_snapshot as v
from scripts.resource_preflight import GB, SHARED_MODEL_CACHE_SHARE_BYTES


def test_git_blob_sha1_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert v.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def entries_for(tmp_path):
    big = b"x" * 1000
    (tmp_path / "w.safetensors").write_bytes(big)
    (tmp_path / "config.json").write_bytes(b"{}")
    return [
        {"type": "file", "path": "w.safetensors", "size": 1000, "oid": "unused",
         "lfs": {"oid": hashlib.sha256(big).hexdigest()}},
        {"type": "file", "path": "config.json", "size": 2, "oid": v.git_blob_sha1(b"{}")},
    ]


def test_verify_accepts_exact_files_and_flags_problems(tmp_path):
    entries = entries_for(tmp_path)
    assert v.verify(entries, tmp_path) == {}
    (tmp_path / "config.json").write_bytes(b"[]")
    (tmp_path / "w.safetensors").write_bytes(b"x" * 999)
    entries.append({"type": "file", "path": "gone.txt", "size": 1, "oid": "0"})
    assert v.verify(entries, tmp_path) == {
        "config.json": "digest mismatch", "w.safetensors": "size 999 != 1000", "gone.txt": "missing",
    }


def test_preflight_counts_only_missing_bytes_against_the_shared_share(tmp_path, monkeypatch):
    entries = entries_for(tmp_path) + [{"type": "file", "path": "next.safetensors", "size": 6 * GB, "oid": "0"}]
    monkeypatch.setattr(v, "project_bytes", lambda path: SHARED_MODEL_CACHE_SHARE_BYTES - 6 * GB)
    ok = v.preflight(entries, tmp_path, tmp_path, free=56 * GB)
    assert ok["missing_bytes"] == 6 * GB and ok["errors"] == []
    monkeypatch.setattr(v, "project_bytes", lambda path: SHARED_MODEL_CACHE_SHARE_BYTES - 6 * GB + 1)
    assert "80 GB" in v.preflight(entries, tmp_path, tmp_path, free=56 * GB)["errors"][0]
    assert v.preflight(entries, tmp_path, tmp_path, free=56 * GB - 1)["errors"]


def test_cli_reports_problems(tmp_path, capsys):
    entries = entries_for(tmp_path)
    tree = tmp_path / "tree.json"
    tree.write_text(json.dumps(entries + [{"type": "directory", "path": "d"}]))
    assert v.main(["--tree", str(tree), "--dest", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["files"] == 2
