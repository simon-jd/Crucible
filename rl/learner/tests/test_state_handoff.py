"""Tiny real CPU handoff evidence; no full-model, CUDA or quality readiness claim."""

import copy
import errno
import fcntl
import io
import json
import os
import struct
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")

from rl.learner import handoff_zip, state_handoff
from rl.learner.state_handoff import (
    F_ADD_SEALS,
    F_GET_SEALS,
    SEALS,
    CaptureBudget,
    Receipt,
    SnapshotOwner,
    _assert_unmapped,
    _BoundedWriter,
    _digest,
    _MetadataSink,
    _payload,
    _save,
    capture_ready_learner,
)
from rl.learner.tests.handoff_worker import BINDING, CAP, group, make_learner, summary


def command(mode):
    return [sys.executable, "-B", "-m", "rl.learner.tests.handoff_worker", mode]


@contextmanager
def artifact(payload, *, seal=True):
    fd = os.memfd_create("tiny-handoff-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        torch.save(payload, _BoundedWriter(CAP, fd))
        if seal:
            fcntl.fcntl(fd, F_ADD_SEALS, SEALS)
        size = os.fstat(fd).st_size
        receipt = Receipt(
            os.fstat(fd).st_dev,
            os.fstat(fd).st_ino,
            size,
            _digest(fd, size),
            SEALS,
            payload.get("version", 1),
            json.dumps(BINDING, sort_keys=True, separators=(",", ":")),
            CAP,
        )
        yield fd, receipt
    finally:
        os.close(fd)


def trained_payload(kind="AdamW"):
    learner = make_learner(optimizer=kind)
    learner.step([group(0)])
    return _payload(learner, BINDING)[0]


def assert_failed(learner):
    assert learner.phase_status == "failed"
    assert learner.failure_reason
    for name in ("policy", "optimizer", "policy_version"):
        with pytest.raises(RuntimeError, match="phase"):
            getattr(learner, name)
    with pytest.raises(RuntimeError, match="phase"):
        learner.step([])


def test_sealed_memfd_weights_only_mmap_support():
    assert not torch.cuda.is_initialized()
    payload = {"value": torch.arange(16, dtype=torch.float32)}
    with artifact(payload) as (fd, receipt):
        receipt.validate(fd)
        assert fcntl.fcntl(fd, F_GET_SEALS) == SEALS
        for operation in (
            lambda: os.pwrite(fd, b"x", 0),
            lambda: os.ftruncate(fd, receipt.size + 1),
            lambda: os.ftruncate(fd, receipt.size - 1),
        ):
            with pytest.raises(OSError) as caught:
                operation()
            assert caught.value.errno == errno.EPERM
        assert os.fstat(fd).st_size == receipt.size
        assert _digest(fd, receipt.size) == receipt.sha256
        loaded = torch.load(f"/proc/self/fd/{fd}", weights_only=True, mmap=True, map_location="cpu")
        torch.testing.assert_close(loaded["value"], payload["value"])
        loaded["value"].add_(1)
        assert _digest(fd, receipt.size) == receipt.sha256
        # The storage must occupy a private memfd mapping, not a copied heap record.
        address = loaded["value"].data_ptr()
        matches = []
        with open("/proc/self/maps") as maps:
            for line in maps:
                interval, permissions, *_ = line.split()
                low, high = (int(value, 16) for value in interval.split("-"))
                if low <= address < high:
                    matches.append(line)
                    assert permissions.endswith("p") and "/memfd:tiny-handoff-test" in line
        assert len(matches) == 1
        del loaded
    assert not torch.cuda.is_initialized()


def test_original_child_exit_then_fresh_process_adamw_resume():
    baseline = make_learner()
    baseline.step([group(0)])
    expected_restored = summary(baseline)
    next_loss = baseline.step([group(1)])["loss"]
    expected_final = summary(baseline)
    with SnapshotOwner(
        max_bytes=CAP, other_resident_bytes=1_000_000_000, ram_limit=2_000_000_000
    ) as owner:
        snapshot = owner.capture(command("capture"), version=1, binding=BINDING)
        original = owner.last_process
        assert original.returncode == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(original.pid, os.WNOHANG)
        next_snapshot = owner.capture(
            command("advance"), version=2, binding=BINDING, source=snapshot
        )
        resumed = json.loads(owner.last_output)
        assert resumed["pid"] != original.pid
        assert resumed["restored"] == expected_restored
        assert resumed["state"] == expected_final
        assert resumed["next_loss"] == next_loss
        assert all(event["reaped"] and event["returncode"] == 0 for event in owner.events)
        assert len(owner.snapshots) == 2
        with pytest.raises(ValueError, match="two"):
            owner.capture(command("advance"), version=3, binding=BINDING, source=next_snapshot)
        owner.release(snapshot)  # source consumer is now proved reaped; slot can be reused.
        inspection = json.loads(owner.consume(next_snapshot, command("inspect")))
        assert inspection["state"] == expected_final
        for state in resumed["state"]["optimizer"]["state"].values():
            assert state["step"]["tensor"]["scalar"] == 2
            assert state["exp_avg"]["tensor"]["absolute_sum"] > 0
            assert state["exp_avg_sq"]["tensor"]["absolute_sum"] > 0
        print(
            json.dumps(
                {
                    "kind": "actual-cpu-adamw-cross-process-resume",
                    "events": owner.events,
                    "receipt": asdict(next_snapshot.receipt),
                    "next_loss": next_loss,
                }
            )
        )
        fd = next_snapshot.fd
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize(
    "change",
    [
        "name",
        "order",
        "shape",
        "dtype",
        "config",
        "version",
        "schema",
        "aliases",
        "options",
        "cuda_rng",
        "state_key",
    ],
)
def test_mapping_rejected_before_mutation(change):
    payload = trained_payload()
    if change == "name":
        payload["optimizer"]["groups"][0]["names"][0] = "wrong"
    elif change == "order":
        payload["optimizer"]["groups"][0]["names"].reverse()
    elif change in ("shape", "dtype"):
        meta = payload["optimizer"]["state"]["embedding.weight"]["exp_avg"]["meta"]
        meta[change] = [4, 8] if change == "shape" else "torch.float64"
    elif change == "config":
        payload["config"]["clip_eps"] = 0.3
    elif change == "version":
        payload["version"] = 2
    elif change == "schema":
        payload["unknown"] = 1
    elif change == "aliases":
        payload["policy"]["aliases"] = {}
    elif change == "options":
        payload["optimizer"]["groups"][0]["options"]["lr"] = 0.4
    elif change == "cuda_rng":
        payload["rng"]["cuda"] = {"device": "cuda:1", "count": 2, "state": torch.zeros(8)}
    else:
        payload["optimizer"]["state"]["embedding.weight"]["unknown"] = {}
    learner = make_learner(1)
    policy, optimizer = learner.policy, learner.optimizer
    before = copy.deepcopy(policy.state_dict())
    with artifact(payload) as (fd, receipt):
        # Payload version disagreement is tested after a matching receipt passes.
        receipt = replace(receipt, version=1)
        with pytest.raises((ValueError, RuntimeError)):
            learner.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
    assert_failed(learner)
    assert not optimizer.state
    for name, value in policy.state_dict().items():
        assert torch.equal(value, before[name]), "validation must precede model mutation"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["wrong_fd", "unsealed", "digest", "truncated"])
def test_artifact_validation(fault):
    payload = trained_payload()
    with artifact(payload, seal=fault != "unsealed") as (fd, receipt):
        if fault == "wrong_fd":
            receipt = replace(receipt, inode=receipt.inode + 1)
        elif fault == "digest":
            receipt = replace(receipt, sha256="0" * 64)
        elif fault == "truncated":
            receipt = replace(receipt, size=receipt.size + 1)
        learner = make_learner(1)
        with pytest.raises(ValueError):
            learner.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
        assert_failed(learner)


def test_cap_and_snapshot_ownership():
    learner = make_learner()
    fd = os.memfd_create("cap-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with pytest.raises(ValueError, match="cap"):
            capture_ready_learner(learner, fd, binding=BINDING, budget=CaptureBudget(16, 0))
        assert os.fstat(fd).st_size == 0
        writer = _BoundedWriter(4, fd)
        writer.write(b"123")
        with pytest.raises(ValueError, match="before write"):
            writer.write(b"45")
        assert os.fstat(fd).st_size == 3
    finally:
        os.close(fd)
    with pytest.raises(ValueError, match="RAM"):
        CaptureBudget(CAP, 2_000_000_000, ram_limit=2_000_000_000).check(CAP, 8192)
    # Real stdlib-only child writes and seals a tiny owned artifact; no Torch loader.
    script = (
        "import os,fcntl; from rl.learner.state_handoff import F_ADD_SEALS,SEALS; "
        "f=int(os.environ['CRUCIBLE_HANDOFF_FD']); os.write(f,b'x'); "
        "fcntl.fcntl(f,F_ADD_SEALS,SEALS)"
    )
    with SnapshotOwner(max_bytes=CAP, other_resident_bytes=0) as owner:
        first = owner.capture([sys.executable, "-B", "-c", script], version=0, binding=BINDING)
        second = owner.capture([sys.executable, "-B", "-c", script], version=0, binding=BINDING)
        with pytest.raises(ValueError, match="two"):
            owner.capture(command("capture"), version=1, binding=BINDING)
        with (
            SnapshotOwner(max_bytes=CAP, other_resident_bytes=0) as foreign,
            pytest.raises(ValueError, match="owned"),
        ):
            foreign.release(first)
        owner.release(first)
        with pytest.raises(ValueError, match="owned"):
            owner.release(first)
        owner.release(second)


def test_metadata_cap_prevents_buffer_growth():
    metadata_only = io.BytesIO()
    sink = _MetadataSink(metadata_only, 8)
    sink.write(b"1234")
    with pytest.raises(ValueError, match="before buffer write"):
        sink.write(b"56789")
    assert metadata_only.getvalue() == b"1234"
    with pytest.raises(ValueError, match="metadata cap"):
        _save(torch, {"tensor": torch.ones(1)}, _BoundedWriter(CAP), metadata_cap=8)


def test_serial_stage_budget_requires_mapping_release():
    # Formula regression only: zero process/model baseline is not real fit evidence.
    raw, largest = 6_916_458_752, 544_538_624
    planned = CaptureBudget(
        raw,
        0,
        retained_snapshot_bytes=raw,
        mapped_bytes=raw,
        metadata_cap=1_000_000,
        storage_cap=largest,
    )
    assert planned.check(raw, largest) == 2 * raw + 2 * largest + planned.metadata_charge
    assert planned.check(raw, largest) < 16_000_000_000
    with artifact(trained_payload()) as (source_fd, receipt):
        mapped = torch.load(
            f"/proc/self/fd/{source_fd}", weights_only=True, mmap=True, map_location="cpu"
        )
        output = os.memfd_create("release-proof", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
        try:
            budget = CaptureBudget(
                CAP, 0, retained_snapshot_bytes=receipt.size, mapped_bytes=receipt.size
            )
            with pytest.raises(RuntimeError, match="mapping still live"):
                capture_ready_learner(
                    make_learner(2),
                    output,
                    binding=BINDING,
                    budget=budget,
                    source_fd=source_fd,
                    source_receipt=receipt,
                )
            assert os.fstat(output).st_size == 0
            del mapped
            _assert_unmapped(receipt)  # actual kernel maps proof, no GC inference
        finally:
            os.close(output)


@pytest.mark.parametrize(
    "script,timeout,match",
    [
        ("raise SystemExit(7)", 5.0, "exited 7"),
        ("import time; time.sleep(30)", 0.15, "deadline"),
        ("import os,signal; os.kill(os.getpid(), signal.SIGTERM)", 5.0, "exited"),
        ("import os; os.write(int(os.environ['CRUCIBLE_HANDOFF_FD']), b'bad')", 5.0, "seal"),
    ],
)
def test_child_failure_timeout_reaped_and_closed(script, timeout, match):
    before = set(os.listdir("/proc/self/fd"))
    with SnapshotOwner(max_bytes=CAP, other_resident_bytes=0, timeout=timeout) as owner:
        with pytest.raises((ValueError, RuntimeError, TimeoutError), match=match):
            owner.capture([sys.executable, "-B", "-c", script], version=0, binding=BINDING)
        assert owner.last_process.returncode is not None
        assert not owner.snapshots and owner.pending_fd is None
        with pytest.raises(ChildProcessError):
            os.waitpid(owner.last_process.pid, os.WNOHANG)
    assert set(os.listdir("/proc/self/fd")) == before


def test_restore_failure_poisoned_phase(monkeypatch):
    payload = trained_payload()
    learner = make_learner(1)
    optimizer = learner.optimizer
    original = RuntimeError("injected restore cleanup failure")

    def fail_cleanup(*args, **kwargs):
        assert learner.phase_status == "restoring"
        for name in ("policy", "optimizer", "policy_version"):
            with pytest.raises(RuntimeError, match="phase"):
                getattr(learner, name)
        with pytest.raises(RuntimeError, match="phase"):
            learner.step([])
        raise original

    monkeypatch.setattr(optimizer, "zero_grad", fail_cleanup)
    with artifact(payload) as (fd, receipt), pytest.raises(RuntimeError) as caught:
        learner.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
    assert caught.value is original
    assert_failed(learner)


def test_foreign_process_survives_owned_timeout():
    foreign = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"])
    try:
        with SnapshotOwner(max_bytes=CAP, other_resident_bytes=0, timeout=0.15) as owner:
            with pytest.raises(TimeoutError):
                owner.capture(
                    [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                    version=0,
                    binding=BINDING,
                )
            assert foreign.poll() is None
            assert owner.last_process.returncode is not None
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_restore_primary_and_cleanup_causes(monkeypatch):
    payload = trained_payload()
    payload["format"] = "invalid"
    learner = make_learner(1)
    cleanup = RuntimeError("cleanup after schema rejection")

    def fail(*args, **kwargs):
        raise cleanup

    monkeypatch.setattr(learner.optimizer, "zero_grad", fail)
    with artifact(payload) as (fd, receipt), pytest.raises(BaseExceptionGroup) as caught:
        learner.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
    assert isinstance(caught.value.exceptions[0], ValueError)
    assert caught.value.exceptions[1] is cleanup
    assert_failed(learner)


def test_sgd_resume_and_capture_phase_boundaries():
    baseline = make_learner(optimizer="SGD")
    baseline.step([group(0)])
    payload = _payload(baseline, BINDING)[0]
    resumed = make_learner(1, optimizer="SGD")
    with artifact(payload) as (fd, receipt):
        resumed.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
        assert summary(resumed) == summary(baseline)
        expected_loss = baseline.step([group(1)])["loss"]
        assert resumed.step([group(1)])["loss"] == expected_loss
        assert summary(resumed) == summary(baseline)
        with pytest.raises(ValueError, match="fresh"):
            resumed.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
        assert_failed(resumed)
        with pytest.raises(RuntimeError, match="phase"):
            capture_ready_learner(resumed, fd, binding=BINDING, budget=CaptureBudget(CAP, 0))
    # Capture must acquire guards before inspecting even a writable descriptor.
    active = make_learner()
    fd = os.memfd_create("phase-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)

    def executing(opt, args, kwargs):
        with pytest.raises(RuntimeError, match="phase"):
            capture_ready_learner(active, fd, binding=BINDING, budget=CaptureBudget(CAP, 0))

    hook = active.optimizer.register_step_pre_hook(executing)
    try:
        active.step([group(0)])
        assert os.fstat(fd).st_size == 0
    finally:
        hook.remove()
        os.close(fd)


def test_world1_cpu_dtensor_capture_restore():
    # Dedicated processes own Gloo and destroy it; never FSDP/GPU evidence.
    with SnapshotOwner(
        max_bytes=CAP, other_resident_bytes=1_000_000_000, ram_limit=2_000_000_000
    ) as owner:
        snapshot = owner.capture(command("dtensor-capture"), version=0, binding=BINDING)
        result = json.loads(owner.consume(snapshot, command("dtensor-resume")).splitlines()[-1])
        assert (
            result["state"]["optimizer"]["state"]["embedding.weight"]["step"]["tensor"]["scalar"]
            == 2
        )
        meta = result["restored"]["optimizer"]["state"]["embedding.weight"]["exp_avg"]["meta"]
        assert meta["distribution"]["placements"] == [["replicate"]]
        assert all(event["reaped"] for event in owner.events)
        print(json.dumps({"kind": "world1-cpu-gloo-dtensor-only", "events": owner.events}))


def test_consume_reserves_restore_staging_before_spawn():
    with SnapshotOwner(max_bytes=CAP, other_resident_bytes=0) as owner:
        first = owner.capture(command("capture"), version=1, binding=BINDING)
        second = owner.capture(command("capture"), version=1, binding=BINDING)
        retained = first.receipt.size + second.receipt.size
        required = (
            retained
            + first.receipt.size
            + owner.budget.metadata_charge
            + 2 * owner.budget.storage_cap
        )
        stage = replace(
            owner.budget, retained_snapshot_bytes=retained, mapped_bytes=first.receipt.size
        )
        assert stage.check(0, 0) == required
        assert owner.budget.restore_charge(retained, first.receipt.size) == required
        previous, events = owner.last_process, len(owner.events)
        owner.budget = replace(owner.budget, ram_limit=required - 1)
        with pytest.raises(ValueError, match="aggregate RAM"):
            owner.consume(first, command("inspect"))
        assert owner.last_process is previous and len(owner.events) == events
        assert not owner.busy and len(owner.snapshots) == 2
        owner.budget = replace(owner.budget, ram_limit=required)
        restored = json.loads(owner.consume(first, command("inspect")))
        assert restored["state"]["version"] == 1
        assert owner.last_process is not previous
        assert owner.events[-1]["reaped"] and owner.events[-1]["returncode"] == 0


def _tiny_directory_fixture(zip64=False, central64=False):
    """Only small in-memory ZIP bytes; no snapshot on disk."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(handoff_zip.FIXED_NAMES):
            archive.writestr("archive/" + name, b"0")
    data = bytearray(buffer.getvalue())
    end = list(struct.unpack("<4s4H2IH", data[-22:]))
    directory, length = end[6], end[5]
    if central64:
        # Exercise ZIP64 per-record size/offset sentinel resolution on a tiny file.
        original = struct.unpack_from("<4s6H3I5H2I", data, directory)
        extra_at = directory + 46 + original[10]
        extra = struct.pack("<HH3Q", 1, 24, original[9], original[8], original[16])
        struct.pack_into("<II", data, directory + 20, 0xFFFFFFFF, 0xFFFFFFFF)
        struct.pack_into("<H", data, directory + 30, len(extra))
        struct.pack_into("<I", data, directory + 42, 0xFFFFFFFF)
        data[extra_at:extra_at] = extra
        length += len(extra)
        end[5] = length
        data[-22:] = struct.pack("<4s4H2IH", *end)
    if zip64:
        end64_at = len(data) - 22
        end64 = struct.pack(
            "<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, end[3], end[4], length, directory
        )
        locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, end64_at, 1)
        end[3:7] = [65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF]
        data[-22:] = end64 + locator + struct.pack("<4s4H2IH", *end)
    assert len(data) < 8192
    return data, directory, length


@contextmanager
def _sealed_directory_fixture(data, metadata_cap):
    fd = os.memfd_create("tiny-directory-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        assert len(data) <= CAP
        assert os.write(fd, data) == len(data)
        fcntl.fcntl(fd, F_ADD_SEALS, SEALS)
        stat = os.fstat(fd)
        receipt = Receipt(
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            _digest(fd, stat.st_size),
            SEALS,
            1,
            json.dumps(BINDING, sort_keys=True, separators=(",", ":")),
            CAP,
            metadata_cap=metadata_cap,
        )
        yield fd, receipt
    finally:
        os.close(fd)


@pytest.mark.parametrize("zip64", [False, True])
def test_restore_rejects_oversized_zip_directory_before_buffering(monkeypatch, zip64):
    data, directory, length = _tiny_directory_fixture(zip64)
    learner = make_learner(1)
    policy, optimizer = learner.policy, learner.optimizer
    before = copy.deepcopy(policy.state_dict())
    reads = []

    def guarded_preflight(read_at, size, metadata_cap, storage_cap):
        def bounded_read(offset, count):
            reads.append((offset, count))
            assert count <= 128
            assert offset >= directory + length, "oversized directory must never be read"
            return read_at(offset, count)

        return handoff_zip.preflight(bounded_read, size, metadata_cap, storage_cap)

    def forbidden_parser(*args, **kwargs):
        raise AssertionError("eager parser reached before directory rejection")

    monkeypatch.setattr(state_handoff, "preflight", guarded_preflight)
    monkeypatch.setattr(zipfile, "ZipFile", forbidden_parser)
    monkeypatch.setattr(torch, "load", forbidden_parser)
    with _sealed_directory_fixture(data, length - 1) as (fd, receipt):
        receipt.validate(fd)  # valid envelope/digest/seals; not an envelope failure
        with pytest.raises(ValueError, match="directory byte allowance.*before read"):
            learner.restore_handoff(fd, receipt, expected_version=1, binding=BINDING)
    assert reads and sum(count for _, count in reads) <= 98
    assert_failed(learner)
    assert not optimizer.state
    for name, value in policy.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize(
    "fault",
    [
        "valid32",
        "valid64",
        "central64",
        "count",
        "extent",
        "name",
        "extra",
        "comment",
        "zip64_extent",
        "zip64_disagreement",
        "missing_locator",
    ],
)
def test_zip_preflight_structural_bounds(fault):
    use64 = fault in {
        "valid64",
        "central64",
        "zip64_extent",
        "zip64_disagreement",
        "missing_locator",
    }
    data, directory, length = _tiny_directory_fixture(use64, fault == "central64")
    if fault == "count":
        struct.pack_into("<HH", data, len(data) - 14, 65, 65)
    elif fault == "extent":
        struct.pack_into("<I", data, len(data) - 6, directory + 1)
    elif fault in {"name", "extra", "comment"}:
        position, value = {"name": (28, 129), "extra": (30, 29), "comment": (32, 1)}[fault]
        struct.pack_into("<H", data, directory + position, value)
    elif fault == "zip64_extent":
        struct.pack_into("<Q", data, len(data) - 94, 45)
    elif fault == "zip64_disagreement":
        struct.pack_into("<H", data, len(data) - 12, 1)
    elif fault == "missing_locator":
        data[-42:-38] = b"bad!"
    reads = []

    def read(offset, count):
        assert count <= 128
        reads.append((offset, count))
        return data[offset : offset + count]

    if fault.startswith("valid") or fault == "central64":
        result = handoff_zip.preflight(read, len(data), 32768, 8192)
        assert result == {"bytes": length, "records": 6, "zip64": use64}
    else:
        with pytest.raises(ValueError, match="ZIP"):
            handoff_zip.preflight(read, len(data), 32768, 8192)
    assert reads


def test_save_rejects_directory_allowance_before_output(monkeypatch):
    with pytest.raises(ValueError, match="directory record allowance.*before serialization"):
        handoff_zip.check_save_records(59, 32768)
    learner = make_learner()
    fd = os.memfd_create("tiny-save-directory-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    real_preflight = state_handoff.preflight
    reached = []

    def restricted_directory(read_at, size, metadata_cap, storage_cap):
        # Count pass's real ZIP tail, with an independently too-small allowance.
        reached.append(size)
        return real_preflight(read_at, size, 1, storage_cap)

    monkeypatch.setattr(state_handoff, "preflight", restricted_directory)
    try:
        with pytest.raises(ValueError, match="directory byte allowance.*before read"):
            capture_ready_learner(learner, fd, binding=BINDING, budget=CaptureBudget(CAP, 0))
        assert reached and os.fstat(fd).st_size == 0
        assert fcntl.fcntl(fd, F_GET_SEALS) == 0
    finally:
        os.close(fd)
