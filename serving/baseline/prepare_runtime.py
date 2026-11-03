"""Verify a reviewed native runtime lock and install only its isolated wheel deltas."""

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from scripts.resource_preflight import PROJECT_SHARE_BYTES, project_bytes

ROOT = Path(__file__).resolve().parents[2]
CAP = PROJECT_SHARE_BYTES
CLEANUP_RESERVE = 50_000_000
INSTALL_OVERHEAD = 31_614_276
COPIED_SCRIPT_WHEEL = "aeb0323409199c785f730645e9f4bb7a7b9ca2c481f2c331a55642b5d13fa52f"
COPIED_SCRIPT_MEMBER = "py_spy-0.4.2.data/scripts/py-spy"
COPIED_SCRIPT_TARGET = "bin/py-spy"
COPIED_SCRIPT_BYTES = 8_079_336


def copied_script_bytes(expected):
    return COPIED_SCRIPT_BYTES if expected["sha256"] == COPIED_SCRIPT_WHEEL else 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_peak(current, additional, cleanup_reserve=CLEANUP_RESERVE):
    if (
        min(current, additional, cleanup_reserve) < 0
        or current + additional + cleanup_reserve > CAP
    ):
        raise ValueError("project cap would consume cleanup reserve")


def verify_record(site, record, boundary):
    """Check installed RECORD content without importing any installed package."""
    site, boundary = Path(site).resolve(), Path(boundary).resolve()
    count = size = 0
    unhashed = []
    with Path(record).open(newline="") as source:
        for name, encoded, expected_size in csv.reader(source):
            path = (site / name).resolve()
            if not path.is_relative_to(boundary):
                raise ValueError(f"RECORD path escapes owned environment: {name}")
            if not path.is_file():
                raise ValueError(f"RECORD file missing: {name}")
            if expected_size and path.stat().st_size != int(expected_size):
                raise ValueError(f"RECORD size mismatch: {name}")
            if not encoded:
                unhashed.append(name)
                continue
            algorithm, value = encoded.split("=", 1)
            if algorithm != "sha256":
                raise ValueError(f"unsupported RECORD hash: {name}")
            actual = base64.urlsafe_b64encode(bytes.fromhex(sha256(path))).rstrip(b"=").decode()
            if actual != value:
                raise ValueError(f"RECORD hash mismatch: {name}")
            count += 1
            size += path.stat().st_size
    return {"hashed_files": count, "hashed_bytes": size, "unhashed_files": unhashed}


def _files(directory):
    for parent, _, names in os.walk(directory, followlinks=False):
        for name in names:
            path = Path(parent) / name
            if path.is_file() and not path.is_symlink():
                yield path


def audit_hardlinks(cache, target, payloads):
    """Require actual cache/target inode identity for every selected payload file."""
    cache_inodes = {(p.stat().st_dev, p.stat().st_ino) for p in _files(cache)}
    evidence = []
    for relative in payloads:
        path = Path(target) / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"payload missing or symlink: {relative}")
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) not in cache_inodes or stat.st_nlink < 2:
            raise ValueError(f"payload inode not hardlinked from cache: {relative}")
        evidence.append(
            {
                "path": relative,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "links": stat.st_nlink,
                "bytes": stat.st_size,
            }
        )
    return {"linked_payload_bytes": sum(x["bytes"] for x in evidence), "files": evidence}


def clean_environment(cache, temporary):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        UV_CACHE_DIR=str(cache),
        TMPDIR=str(temporary),
        UV_NO_PROGRESS="1",
    )
    env.pop("PYTHONPATH", None)
    return env


def verify_wheel(wheel, expected):
    wheel = Path(wheel)
    if wheel.stat().st_size != expected["compressed_bytes"] or sha256(wheel) != expected["sha256"]:
        raise ValueError("wheel size/hash mismatch")
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        if sum(x.file_size for x in entries) != expected["expanded_bytes"]:
            raise ValueError("wheel expanded size mismatch")
        metadata = [x for x in entries if x.filename.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or metadata[0].file_size > 1 << 20:
            raise ValueError("wheel metadata missing or oversized")
        if (
            "metadata_sha256" in expected
            and hashlib.sha256(archive.read(metadata[0])).hexdigest() != expected["metadata_sha256"]
        ):
            raise ValueError("wheel metadata differs from resolved dependency graph")
        payloads = []
        names = set()
        for item in entries:
            path = Path(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or item.filename in names
                or item.flag_bits & 1
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("unsafe wheel member")
            names.add(item.filename)
            rewritten_record = path == Path(metadata[0].filename).parent / "RECORD"
            # uv regenerates RECORD with installed paths/hashes. Count that copied
            # metadata in the reserved overhead. The sole reviewed executable copy
            # has its own stage reserve; all other large files must link.
            if item.file_size >= 1 << 20 and not rewritten_record:
                if any(part.endswith(".data") for part in path.parts):
                    if (
                        not copied_script_bytes(expected)
                        or item.filename != COPIED_SCRIPT_MEMBER
                        or item.file_size != COPIED_SCRIPT_BYTES
                    ):
                        raise ValueError("large relocated payload lacks exact reviewed mapping")
                    payloads.append(COPIED_SCRIPT_TARGET)
                else:
                    payloads.append(item.filename)
    return payloads


def audit_copied_script(wheel, expected, cache, target):
    """One reviewed uv script copy, never represented as a hardlinked payload."""
    if not copied_script_bytes(expected):
        return None
    with zipfile.ZipFile(wheel) as archive:
        data = archive.read(COPIED_SCRIPT_MEMBER)
    if len(data) != COPIED_SCRIPT_BYTES or data[:4] != b"\x7fELF":
        raise ValueError("reviewed script archive content/size mismatch")
    digest = hashlib.sha256(data).hexdigest()
    cached = [p for p in _files(cache) if p.as_posix().endswith("/" + COPIED_SCRIPT_MEMBER)]
    if len(cached) != 1:
        raise ValueError("reviewed script cache identity ambiguous")
    paths = [cached[0], Path(target) / COPIED_SCRIPT_TARGET]
    evidence = []
    for path in paths:
        stat = path.stat()
        if path.is_symlink() or stat.st_size != COPIED_SCRIPT_BYTES or sha256(path) != digest:
            raise ValueError("reviewed copied script content/size mismatch")
        evidence.append(
            {
                "path": str(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "links": stat.st_nlink,
                "bytes": stat.st_size,
            }
        )
    return {
        "wheel_sha256": expected["sha256"],
        "archive_member": COPIED_SCRIPT_MEMBER,
        "payload_sha256": digest,
        "files": evidence,
        "shared_inode": (evidence[0]["device"], evidence[0]["inode"])
        == (evidence[1]["device"], evidence[1]["inode"]),
        "reserved_extra_copy_bytes": COPIED_SCRIPT_BYTES,
    }


def install_one(wheel, expected, target, cache, temporary, python):
    """Offline trusted installer; caller supplies one fully verified locked wheel."""
    target, cache, temporary = (Path(p).resolve() for p in (target, cache, temporary))
    for path in (target, cache, temporary):
        if not path.is_relative_to(ROOT / ".cache/task12-baseline"):
            raise ValueError("installation path outside task-owned scratch")
        path.mkdir(parents=True, exist_ok=True)
    if len({p.stat().st_dev for p in (target, cache, temporary)}) != 1:
        raise ValueError("hardlink staging must use one filesystem")
    payloads = verify_wheel(wheel, expected)
    requirement = temporary / "wheel-requirement.txt"
    requirement.write_text(f"{Path(wheel).resolve().as_uri()} --hash=sha256:{expected['sha256']}\n")
    command = [
        "uv",
        "--no-config",
        "--no-python-downloads",
        "--offline",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--require-hashes",
        "--only-binary",
        ":all:",
        "--link-mode",
        "hardlink",
        "--python",
        str(python),
        "--target",
        str(target),
        "-r",
        str(requirement),
    ]
    before = project_bytes(ROOT)
    check_peak(
        before, expected["expanded_bytes"] + INSTALL_OVERHEAD + copied_script_bytes(expected)
    )
    result = subprocess.run(
        command,
        check=False,
        env=clean_environment(cache, temporary),
        capture_output=True,
        text=True,
        timeout=180,
    )
    evidence = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "project_before": before,
        "project_after": project_bytes(ROOT),
    }
    if (
        result.returncode
        or "falling back" in result.stderr.lower()
        or "failed to hardlink" in result.stderr.lower()
    ):
        raise RuntimeError(json.dumps(evidence))
    evidence["copied_script"] = audit_copied_script(wheel, expected, cache, target)
    if evidence["copied_script"] is not None:
        payloads.remove(COPIED_SCRIPT_TARGET)
    evidence["hardlinks"] = audit_hardlinks(cache, target, payloads)
    return evidence


def prove_installer(directory, python):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("proof directory must be fresh")
    directory.mkdir(parents=True)
    wheel = directory / "crucible_link_proof-0.0.0-py3-none-any.whl"
    metadata = "crucible_link_proof-0.0.0.dist-info/"
    members = {
        "crucible_link_proof/payload.bin": bytes(range(256)) * (4 << 12),
        metadata
        + "METADATA": b"Metadata-Version: 2.1\nName: crucible-link-proof\nVersion: 0.0.0\n",
        metadata
        + "WHEEL": b"Wheel-Version: 1.0\nGenerator: task12-proof\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        metadata + "RECORD": b"",
    }
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    expected = {
        "compressed_bytes": wheel.stat().st_size,
        "sha256": sha256(wheel),
        "expanded_bytes": sum(map(len, members.values())),
    }
    evidence = install_one(
        wheel, expected, directory / "target", directory / "cache", directory / "tmp", python
    )
    payload = directory / "target/crucible_link_proof/payload.bin"
    original = payload.stat()
    shutil.rmtree(directory / "cache")
    evidence["target_survives_cache_cleanup"] = (
        payload.stat().st_ino == original.st_ino
        and payload.read_bytes() == members["crucible_link_proof/payload.bin"]
    )
    if not evidence["target_survives_cache_cleanup"]:
        raise ValueError("target changed after cache removal")
    evidence["wheel"] = expected
    shutil.rmtree(directory)
    evidence["proof_removed"] = not directory.exists()
    evidence["project_after_cleanup"] = project_bytes(ROOT)
    return evidence


def validate_lock(lock):
    names = set()
    for wheel in lock["wheels"]:
        url = urllib.parse.urlparse(wheel["url"])
        if (
            url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or url.username
            or url.password
        ):
            raise ValueError("unreviewed wheel host")
        if wheel["name"] in names:
            raise ValueError("duplicate wheel in runtime lock")
        names.add(wheel["name"])
        if (
            Path(wheel["filename"]).name != wheel["filename"]
            or not wheel["filename"].endswith(".whl")
            or not url.path.endswith("/" + wheel["filename"])
            or len(wheel["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in wheel["sha256"])
            or any(
                type(wheel[k]) is not int or wheel[k] <= 0
                for k in ("compressed_bytes", "expanded_bytes")
            )
        ):
            raise ValueError("invalid locked wheel identity/bounds")
    overlay = {x["name"]: x["version"] for x in lock["packages"] if x["origin"] == "overlay"}
    if overlay != {x["name"]: x["version"] for x in lock["wheels"]}:
        raise ValueError("wheel set differs from reviewed overlay graph")


def download_wheel(expected, destination, *, test_loopback=False, deadline=None):
    from serving.baseline.deadline_http import absolute_deadline, open_response, remaining

    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if os.path.lexists(destination) or os.path.lexists(partial):
        raise ValueError("download destination must be fresh")
    original = urllib.parse.urlparse(expected["url"])

    def allowed(url):
        if test_loopback:
            return (
                url.scheme == "http" and url.hostname == "127.0.0.1" and url.port == original.port
            )
        return (
            url.scheme == "https"
            and url.hostname == "files.pythonhosted.org"
            and url.port in (None, 443)
        )

    began = time.monotonic()
    deadline = deadline if deadline is not None else began + 300
    with absolute_deadline(deadline):
        digest = hashlib.sha256()
        count = 0
        with open_response(expected["url"], deadline, allowed) as response:
            if (
                response.status != 200
                or response.headers.get("Content-Length") != str(expected["compressed_bytes"])
                or response.headers.get("Content-Encoding", "identity") != "identity"
            ):
                raise ValueError("wheel response size/status/encoding mismatch")
            with partial.open("xb") as output:
                while count < expected["compressed_bytes"]:
                    remaining(deadline)
                    chunk = response.read1(min(1 << 20, expected["compressed_bytes"] - count))
                    remaining(deadline)
                    if not chunk:
                        raise ValueError("truncated wheel download")
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
        remaining(deadline)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("downloaded wheel hash mismatch")
        remaining(deadline)
        os.link(partial, destination)  # no-clobber; preserve owned partial on failure
        remaining(deadline)
        partial.unlink()
        remaining(deadline)
        return {"bytes": count, "sha256": digest.hexdigest(), "seconds": time.monotonic() - began}


def verify_completed_prefix(ordered_wheels, receipt, target):
    """Verify an explicit recovery receipt; never infer completion from a directory."""
    completed = receipt["completed"]
    if not completed or len(completed) > len(ordered_wheels):
        raise ValueError("invalid completed prefix length")
    verified = []
    for expected, item in zip(ordered_wheels, completed, strict=False):
        if item["name"] != expected["name"] or item["wheel_sha256"] != expected["sha256"]:
            raise ValueError("recovery is not the exact locked prefix")
        record = (Path(target) / item["record"]).resolve()
        if not record.is_relative_to(Path(target).resolve()) or record.name != "RECORD":
            raise ValueError("recovery RECORD escapes target")
        metadata = record.parent / "METADATA"
        if sha256(metadata) != expected["metadata_sha256"]:
            raise ValueError("recovery metadata identity changed")
        if sha256(record) != item["record_sha256"]:
            raise ValueError("recovery RECORD identity changed")
        audit = verify_record(target, record, target)
        verified.append({**item, "record_validation": audit, "recovered_without_reinstall": True})
    return verified


def validate_pending_stage(stage, expected, pending, lock_sha256):
    """Permit only a named failed-before-install stage, never an inferred retry."""
    stage = Path(stage)
    if (
        pending.get("stage") != stage.name
        or pending.get("wheel_sha256") != expected["sha256"]
        or json.loads((stage / "owner.json").read_text())
        != {"lock_sha256": lock_sha256, "wheel": expected["filename"]}
    ):
        raise ValueError("pending stage receipt/owner mismatch")
    for path in stage.iterdir():
        if path.is_symlink() or path.name not in {
            "owner.json",
            expected["filename"],
            "cache",
            "tmp",
        }:
            raise ValueError("pending stage contains unexpected contents")
        if path.name in {"cache", "tmp"} and (not path.is_dir() or any(path.iterdir())):
            raise ValueError("pending stage has unverified installer contents")
    if not (stage / expected["filename"]).is_file():
        raise ValueError("pending wheel missing")


def install_locked(lock_path, work, output, continuation=None):
    lock_path, work, output = Path(lock_path), Path(work).resolve(), Path(output)
    lock = json.loads(lock_path.read_text())
    validate_lock(lock)
    if len(lock["wheels"]) != 62:
        raise ValueError("this reviewed baseline requires exactly62 delta wheels")
    if (
        work != ROOT / ".cache/task12-baseline"
        or output.exists()
        or ((work / "overlay").exists() and continuation is None)
    ):
        raise ValueError("refuse to overwrite a prior installation/evidence")
    ordered = sorted(lock["wheels"], key=lambda w: w["compressed_bytes"], reverse=True)
    target = work / "overlay"
    completed = []
    pending = None
    if continuation is not None:
        receipt = json.loads(Path(continuation).read_text())
        if receipt["lock_sha256"] != sha256(lock_path):
            raise ValueError("continuation runtime lock changed")
        completed = verify_completed_prefix(ordered, receipt, target)
        pending = receipt.get("pending")
        remaining_peak = (
            sum(w["expanded_bytes"] for w in ordered[len(completed) :])
            + INSTALL_OVERHEAD
            + 600_000_000
            + max((copied_script_bytes(w) for w in ordered[len(completed) :]), default=0)
        )
        check_peak(project_bytes(ROOT), remaining_peak)
    else:
        check_peak(project_bytes(ROOT), 3_500_000_000)
        target.mkdir()
    evidence = {
        "lock_sha256": sha256(lock_path),
        "sources": {
            str(Path(__file__).relative_to(ROOT)): sha256(__file__),
            "scripts/resource_preflight.py": sha256(ROOT / "scripts/resource_preflight.py"),
        },
        "uv_sha256": sha256(shutil.which("uv")),
        "python_sha256": sha256(ROOT / ".venv-ml/bin/python"),
        "project_initial": project_bytes(ROOT),
        "wheels": completed,
        "continuation_sha256": sha256(continuation) if continuation else None,
        "status": "running",
    }
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    cumulative_expanded = sum(w["expanded_bytes"] for w in ordered[: len(completed)])
    try:
        for index, expected in enumerate(ordered[len(completed) :], start=len(completed)):
            for relative, digest in evidence["sources"].items():
                if sha256(ROOT / relative) != digest:
                    raise ValueError("installer source changed while running")
            if sha256(lock_path) != evidence["lock_sha256"]:
                raise ValueError("runtime lock changed while running")
            stage = work / f"wheel-stage-{index:02d}"
            if pending is not None and index == len(receipt["completed"]) and not stage.exists():
                raise ValueError("explicit pending wheel stage is missing; refuse redownload")
            if stage.exists():
                if pending is None or index != len(receipt["completed"]):
                    raise ValueError("unexpected previous wheel stage")
                validate_pending_stage(stage, expected, pending, evidence["lock_sha256"])
            check_peak(
                project_bytes(ROOT),
                expected["compressed_bytes"]
                + expected["expanded_bytes"]
                + INSTALL_OVERHEAD
                + copied_script_bytes(expected),
            )
            wheel = stage / expected["filename"]
            if stage.exists():
                verify_wheel(wheel, expected)
                downloaded = {
                    "bytes": 0,
                    "sha256": sha256(wheel),
                    "reused_pending_wheel": True,
                    "prior_compressed_bytes": expected["compressed_bytes"],
                }
                pending = None
            else:
                stage.mkdir()
                (stage / "owner.json").write_text(
                    json.dumps(
                        {"lock_sha256": evidence["lock_sha256"], "wheel": expected["filename"]}
                    )
                )
                downloaded = download_wheel(expected, wheel)
            installed = install_one(
                wheel,
                expected,
                target,
                stage / "cache",
                stage / "tmp",
                ROOT / ".venv-ml/bin/python",
            )
            with zipfile.ZipFile(wheel) as archive:
                records = [
                    name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
                ]
            if len(records) != 1:
                raise ValueError("wheel must have one RECORD")
            record = target / records[0]
            verified = verify_record(target, record, target)
            if installed["project_after"] - installed["project_before"] > expected[
                "expanded_bytes"
            ] + INSTALL_OVERHEAD + copied_script_bytes(expected):
                raise ValueError("unexpected installer expansion/metadata growth")
            cumulative_expanded += expected["expanded_bytes"]
            target_bytes = project_bytes(target)
            if target_bytes > cumulative_expanded + INSTALL_OVERHEAD:
                raise ValueError("cumulative installed metadata exceeds reserved bytes")
            item = {
                "name": expected["name"],
                "target_unique_bytes": target_bytes,
                "download": downloaded,
                "install": installed,
                "record": verified,
                "record_sha256": sha256(record),
            }
            evidence["wheels"].append(item)
            # Every directory below this fresh stage was created by this call. Target
            # payload hardlinks, the exact copied-script exception, and RECORD content
            # have been verified before unlinking.
            shutil.rmtree(stage)
            item["project_after_stage_cleanup"] = project_bytes(ROOT)
            check_peak(item["project_after_stage_cleanup"], 0)
            output.write_text(json.dumps(evidence, indent=2) + "\n")
            print(
                f"installed {index + 1}/62 {expected['name']} project_bytes={item['project_after_stage_cleanup']}",
                flush=True,
            )
        evidence["status"] = "installed-awaiting-native-validation"
    except BaseException as error:
        evidence["status"] = "failed"
        evidence["error"] = repr(error)
        raise
    finally:
        evidence["project_final"] = project_bytes(ROOT)
        output.write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prove-installer", type=Path)
    mode.add_argument("--install-lock", type=Path)
    parser.add_argument("--work", type=Path, default=ROOT / ".cache/task12-baseline")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-ml/bin/python")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-from", type=Path)
    args = parser.parse_args()
    if args.install_lock:
        install_locked(args.install_lock, args.work, args.output, args.continue_from)
    else:
        evidence = prove_installer(args.prove_installer, args.python)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
