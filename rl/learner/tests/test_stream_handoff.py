"""Explicit consumed-source CPU path for the serial training-state handoff."""

import array
import contextlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")

from rl.learner import handoff_protocol as protocol
from rl.learner import handoff_stream as stream
from rl.learner import state_handoff as state
from rl.learner.tests.handoff_worker import BINDING, CAP, group, make_learner, summary
from rl.learner.tests.test_state_handoff import assert_failed, command


def budget(**changes):
    return replace(stream.SerialBudget(CAP, 1_000_000_000, CAP, CAP), **changes)


def owner(**changes):
    return state.SnapshotOwner(
        max_bytes=CAP, other_resident_bytes=1_000_000_000, ram_limit=2_000_000_000, **changes
    )


@contextlib.contextmanager
def artifact(data, version=1):
    fd = os.memfd_create("stream-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        stream._write(fd, data)
        state.fcntl.fcntl(fd, state.F_ADD_SEALS, state.SEALS)
        info = os.fstat(fd)
        receipt = state.Receipt(
            info.st_dev,
            info.st_ino,
            len(data),
            state._digest(fd, len(data)),
            state.SEALS,
            version,
            state._binding(BINDING),
            CAP,
        )
        yield fd, receipt
    finally:
        os.close(fd)


def fixture_bytes(change=None):
    learner = make_learner()
    learner.step([group(0)])
    payload = state._payload(learner, BINDING)[0]
    encoded, tensors = stream._encode(payload, budget())
    manifest = json.loads(encoded)
    if change:
        change(manifest)
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    raw = b"".join(bytes(stream._raw(t)) for t in tensors)
    return stream.HEADER.pack(stream.MAGIC, len(encoded)) + encoded + raw


def stdlib_worker(body):
    prelude = """import json, os, socket, time, signal
from rl.learner import state_handoff as state
from rl.learner import handoff_protocol as p
r = state.Receipt(**json.loads(os.environ['CRUCIBLE_HANDOFF_SOURCE_RECEIPT']))
fd = int(os.environ['CRUCIBLE_HANDOFF_SOURCE_FD'])
c = socket.socket(fileno=int(os.environ['CRUCIBLE_STREAM_CONTROL_FD']))
n = bytes.fromhex(os.environ['CRUCIBLE_STREAM_NONCE'])
assert 'CRUCIBLE_HANDOFF_FD' not in os.environ
"""
    return [sys.executable, "-B", "-c", prelude + body]


def install_source(owned):
    # Real capture in this test process, then a real original producer subprocess
    # writes those bytes to its inherited FD and seals; owner accepts only on reap.
    data = fixture_bytes()
    script = """import os, sys
from rl.learner import state_handoff as s
from rl.learner.handoff_stream import _write
fd = int(os.environ['CRUCIBLE_HANDOFF_FD'])
_write(fd, bytes.fromhex(sys.argv[1]))
s.fcntl.fcntl(fd, s.F_ADD_SEALS, s.SEALS)
"""
    return owned.capture_stream(
        [sys.executable, "-B", "-c", script, data.hex()],
        version=1,
        binding=BINDING,
        budget=budget(),
    )


def test_stream_real_adamw_consumption_and_fresh_inspect():
    baseline = make_learner()
    baseline.step([group(0)])
    expected_first = summary(baseline)
    next_loss = baseline.step([group(1)])["loss"]
    expected = summary(baseline)
    with owner() as owned:
        first = owned.capture_stream(
            command("stream-capture"), version=1, binding=BINDING, budget=budget()
        )
        producer = owned.last_process
        with pytest.raises(ChildProcessError):
            os.waitpid(producer.pid, os.WNOHANG)
        new = owned.advance_consuming(
            first,
            command("stream-advance"),
            version=2,
            binding=BINDING,
            budget=budget(weights_bytes=1024, weights_slots=1),
        )
        resumed = json.loads(owned.last_output)
        assert resumed["restored"] == expected_first
        assert resumed["state"] == expected
        assert resumed["next_loss"] == next_loss
        assert owned.snapshots == {new.fd: new}
        stages = [event["stage"] for event in owned.events if "stage" in event]
        assert (
            stages.index("source-proof")
            < stages.index("source-retired")
            < stages.index("output-granted")
        )
        proof = next(event for event in owned.events if event.get("stage") == "source-proof")
        assert proof["child"]["source_fds"] == []
        assert proof["parent"]["source_fds"] == [first.fd]
        grant = next(event for event in owned.events if event.get("stage") == "output-granted")
        assert grant["slots"] == 2
        assert owned.last_process.returncode == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(owned.last_process.pid, os.WNOHANG)
        # Fresh independent inspection, explicitly passed source only and its budget.
        text = owned._child(
            command("stream-inspect"),
            new.fd,
            {
                "CRUCIBLE_HANDOFF_SOURCE_FD": str(new.fd),
                "CRUCIBLE_HANDOFF_SOURCE_RECEIPT": json.dumps(asdict(new.receipt)),
                "CRUCIBLE_STREAM_BUDGET": json.dumps(
                    asdict(budget(weights_bytes=1024, weights_slots=1))
                ),
            },
        )
        assert json.loads(text)["state"] == expected
        print(
            json.dumps(
                {
                    "kind": "serial-stream-real-adamw",
                    "events": owned.events,
                    "receipt": asdict(new.receipt),
                    "loss": next_loss,
                }
            )
        )


def test_manifest_allocation_read_boundary_and_no_cuda_queries(monkeypatch):
    data = fixture_bytes()
    learner = make_learner(1)
    boundaries = []
    allocate, read = stream._allocate_record, stream._read_record

    def allocation(torch_module, record):
        boundaries.append(("allocate", record["bytes"]))
        return allocate(torch_module, record)

    def raw_read(fd, tensor, offset):
        boundaries.append(("read", tensor.numel() * tensor.element_size()))
        return read(fd, tensor, offset)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU path must not query or initialize CUDA")

    for name in ("is_available", "device_count", "current_device", "init", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(stream, "_allocate_record", allocation)
    monkeypatch.setattr(stream, "_read_record", raw_read)
    with artifact(data) as (fd, receipt):
        saved, records, _ = stream.read_manifest(fd, receipt, budget())
        assert saved["version"] == 1 and not boundaries
        learner.restore_handoff_stream(
            fd, receipt, expected_version=1, binding=BINDING, budget=budget()
        )
        assert len(boundaries) == 2 * len(records)
        assert all(x[0] == "allocate" for x in boundaries[::2])
        assert all(x[1] <= budget().storage_cap for x in boundaries)
        protocol.prove_absent(os.getpid(), receipt, fd)


@pytest.mark.parametrize(
    "fault", ["bytes", "offset", "dtype", "device", "shape", "stride", "alias", "extra", "cpu"]
)
def test_malformed_manifest_rejected_before_allocation(monkeypatch, fault):
    def change(manifest):
        record = manifest["records"][0]
        if fault == "bytes":
            record["bytes"] = 2**62
        elif fault == "offset":
            record["offset"] = 1
        elif fault == "dtype":
            record["dtype"] = "torch.complex128"
        elif fault == "device":
            record["device"] = "cuda:1"
        elif fault == "shape":
            record["shape"] = [2**62]
        elif fault == "stride":
            record["stride"] = [0] * len(record["stride"])
        elif fault == "alias":
            # Duplicate tensor reference, retaining all raw records.
            def walk(node):
                if type(node) is list:
                    if len(node) == 2 and node[0] == "tensor":
                        node[1] = 0
                    else:
                        for value in node:
                            walk(value)

            walk(manifest["state"])
        elif fault == "extra":
            manifest["unknown"] = 1

    data = fixture_bytes(change)
    learner = make_learner(1)
    before = [p.detach().clone() for p in learner.policy.parameters()]
    monkeypatch.setattr(
        stream, "_allocate_record", lambda *a: pytest.fail("allocated before validation")
    )
    with artifact(data) as (fd, receipt), pytest.raises((ValueError, TypeError)):
        learner.restore_handoff_stream(
            fd,
            receipt,
            expected_version=1,
            binding=BINDING,
            budget=budget(cpu_candidates=0) if fault == "cpu" else budget(),
        )
    assert_failed(learner)
    assert all(torch.equal(a, b) for a, b in zip(before, learner._policy.parameters()))


@pytest.mark.parametrize(
    "data", [b"[" * 80 + b"]" * 80, b'{"a":0,"a":1}', b'{"a":9223372036854775808}', b'{"a":NaN}']
)
def test_decoded_metadata_limits(data):
    with pytest.raises(ValueError):
        stream._json(data, 32768)


def test_tagged_primitives_roundtrip():
    values = {
        "none": None,
        "bool": True,
        "int": 17,
        "float": 0.25,
        "str": "state",
        "list": [1, (2,)],
        "tuple": (3, [4]),
        ("key", 1): False,
    }
    encoded, tensors = stream._encode(values, budget())
    assert not tensors
    manifest = stream._json(encoded, 32768)
    assert state._same(stream._decode(manifest["state"], [], 32768), values)


@pytest.mark.parametrize("fault", ["nan", "step", "rng", "config"])
def test_actual_validation_precedes_mutation(monkeypatch, fault):
    learner = make_learner()
    learner.step([group(0)])
    payload = state._payload(learner, BINDING)[0]
    encoded, tensors = stream._encode(payload, budget())
    if fault == "config":
        payload["config"]["group_size"] += 1
        encoded, tensors = stream._encode(payload, budget())
    raw = [bytearray(stream._raw(t)) for t in tensors]
    # Locate tensor records through decoded metadata, then alter actual bytes.
    manifest = stream._json(encoded, 32768)
    saved = stream._decode(manifest["state"], manifest["records"], 32768)
    if fault == "nan":
        record = next(iter(saved["policy"]["tensors"].values()))
        raw[record.index][:4] = bytes.fromhex("0000c07f")
    elif fault == "step":
        record = next(iter(saved["optimizer"]["state"].values()))["step"]["tensor"]
        raw[record.index][:4] = bytes.fromhex("000080bf")
    elif fault == "rng":
        raw[saved["rng"]["cpu"].index][:] = b"\xff" * len(raw[saved["rng"]["cpu"].index])
    data = stream.HEADER.pack(stream.MAGIC, len(encoded)) + encoded + b"".join(raw)
    fresh = make_learner(1)
    before = [p.detach().clone() for p in fresh.policy.parameters()]
    with artifact(data) as (fd, receipt), pytest.raises((ValueError, RuntimeError)):
        fresh.restore_handoff_stream(
            fd, receipt, expected_version=1, binding=BINDING, budget=budget()
        )
    assert_failed(fresh)
    assert all(torch.equal(a, b) for a, b in zip(before, fresh._policy.parameters()))


@pytest.mark.parametrize(
    "body,retired",
    [
        ("os._exit(7)", False),
        ("os.kill(os.getpid(), signal.SIGTERM)", False),
        ("time.sleep(5)", False),
        ("os.close(fd); p._send(c, p._packet(p.ACK, b'x'*16, r)); time.sleep(5)", False),
        ("p._send(c, p._packet(p.ACK, n, r)); time.sleep(5)", False),
        (
            "import mmap\nm=mmap.mmap(fd,0,access=mmap.ACCESS_READ)\nos.close(fd)\np._send(c,p._packet(p.ACK,n,r)); time.sleep(5)",
            False,
        ),
        ("p._send(c,p._packet(p.ACK,n,r),fd); time.sleep(5)", False),
        ("os.close(fd); c.send(b'x'); time.sleep(5)", False),
        ("os.close(fd); p._send(c,p._packet(p.ACK,n,r)); p._receive(c); os._exit(7)", True),
        ("os.close(fd); p._send(c,p._packet(p.ACK,n,r)); p._receive(c); time.sleep(5)", True),
        (
            "os.close(fd); p._send(c,p._packet(p.ACK,n,r)); p._receive(c); p._send(c,p._packet(p.ACK,n,r)); time.sleep(5)",
            True,
        ),
    ],
)
def test_protocol_rejections_no_replacement(body, retired):
    owned = owner(timeout=1)
    source = install_source(owned)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        owned.advance_consuming(
            source, stdlib_worker(body), version=2, binding=BINDING, budget=budget()
        )
    assert owned.last_process.returncode is not None
    assert len(owned.snapshots) == (0 if retired else 1)
    assert not any(e.get("accepted") is True for e in owned.events)
    if retired:
        assert owned.failed and owned.pending_fd is None
    else:
        owned.close()


def test_grant_transfer_failure_is_irreversible(monkeypatch):
    owned = owner(timeout=2)
    source = install_source(owned)
    send = protocol._send

    def uncertain(channel, data, fd=None):
        send(channel, data, fd)
        raise OSError("injected partial/uncertain grant")

    monkeypatch.setattr(protocol, "_send", uncertain)
    body = "os.close(fd); p._send(c,p._packet(p.ACK,n,r)); p._receive(c); time.sleep(5)"
    with pytest.raises(OSError, match="uncertain"):
        owned.advance_consuming(
            source, stdlib_worker(body), version=2, binding=BINDING, budget=budget()
        )
    assert owned.failed and owned.snapshots == {} and owned.pending_fd is None
    assert owned.last_process.returncode is not None


def test_source_close_uncertainty_retains_reservation_and_reused_fd(monkeypatch):
    owned = owner(timeout=2)
    source = install_source(owned)
    close = protocol._close_identity
    replacement = []

    def uncertain(fd, identity):
        close(fd, identity)
        if identity == (source.receipt.device, source.receipt.inode):
            replacement.append(os.open("/dev/null", os.O_RDONLY))
            raise OSError("source close uncertain")

    monkeypatch.setattr(protocol, "_close_identity", uncertain)
    body = "os.close(fd); p._send(c,p._packet(p.ACK,n,r)); time.sleep(5)"
    try:
        with pytest.raises(OSError, match="uncertain"):
            owned.advance_consuming(
                source, stdlib_worker(body), version=2, binding=BINDING, budget=budget()
            )
        assert owned.failed and source.fd not in owned.snapshots
        assert owned._serial_retirement.source is source
        assert not owned._serial_retirement.close_confirmed
        assert owned._serial_retirement.reserved_bytes == source.receipt.size
        assert owned._serial_retirement.reserved_slots == 1
        assert owned.pending_fd is None
        assert os.fstat(replacement[0])
    finally:
        for fd in replacement:
            os.close(fd)


def test_primary_and_reap_uncertainty_preserved(monkeypatch):
    owned = owner(timeout=2)
    source = install_source(owned)
    cleanup = protocol._cleanup_process

    def uncertain(process):
        cleanup(process)
        raise RuntimeError("reap uncertainty")

    monkeypatch.setattr(protocol, "_cleanup_process", uncertain)
    with pytest.raises(BaseExceptionGroup) as caught:
        owned.advance_consuming(
            source, stdlib_worker("os._exit(9)"), version=2, binding=BINDING, budget=budget()
        )
    assert "reap uncertainty" in str(caught.value.exceptions[-1])
    assert owned.failed and owned.snapshots.get(source.fd) is source
    # Test owns exact still-open source and proved real cleanup before injected error.
    os.close(source.fd)


def test_budget_weights_slots_and_nonzero_baselines():
    with pytest.raises(ValueError):
        budget(host_baseline=0).check()
    with pytest.raises(ValueError):
        budget(weights_bytes=1).check()
    with pytest.raises(ValueError):
        budget(gpu_candidates=24_000_000_000, gpu_destinations=1).check()
    owned = owner()
    first = install_source(owned)
    with pytest.raises(ValueError, match="two"):
        owned.capture_stream(
            command("stream-capture"),
            version=1,
            binding=BINDING,
            budget=budget(retained_bytes=first.receipt.size, weights_bytes=1, weights_slots=1),
        )
    owned.close()


def test_foreign_process_untouched_by_serial_timeout():
    foreign = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(10)"])
    owned = owner(timeout=0.5)
    try:
        source = install_source(owned)
        with pytest.raises(TimeoutError):
            owned.advance_consuming(
                source, stdlib_worker("time.sleep(5)"), version=2, binding=BINDING, budget=budget()
            )
        assert foreign.poll() is None
        owned.close()
    finally:
        foreign.terminate()
        foreign.wait(timeout=3)


def test_stream_restore_missing_budget_fails_phase():
    fresh = make_learner(1)
    with pytest.raises(TypeError, match="SerialBudget"):
        fresh.restore_handoff_stream(-1, None, expected_version=1, binding=BINDING, budget=None)
    assert_failed(fresh)


@pytest.mark.parametrize("kind", ["extra", "wrong", "truncated"])
def test_actual_scm_rights_grant_rejected_and_closed(kind):
    source = os.memfd_create("grant-source", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    info = os.fstat(source)
    receipt = state.Receipt(
        info.st_dev, info.st_ino, 1, "00" * 32, state.SEALS, 1, state._binding(BINDING), CAP
    )
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    output = os.memfd_create("grant-output", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    out = os.fstat(output)
    nonce = b"n" * 16
    data = protocol._packet(protocol.GRANT, nonce, receipt, (out.st_dev, out.st_ino))
    if kind == "wrong":
        data = protocol._packet(protocol.GRANT, b"x" * 16, receipt, (out.st_dev, out.st_ino))
    if kind == "truncated":
        data += b"extra"
    descriptors = [output, output] if kind == "extra" else [output]
    try:
        sender.sendmsg(
            [data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", descriptors))]
        )
        with pytest.raises(ValueError):
            protocol.retire_input_and_receive_output(source, receipt, receiver.detach(), nonce)
        # Only original output remains: all installed SCM_RIGHTS aliases closed.
        check = replace(receipt, device=out.st_dev, inode=out.st_ino)
        protocol.prove_absent(os.getpid(), check, output)
    finally:
        sender.close()
        receiver.close()
        os.close(output)


def test_ancillary_cleanup_collects_rights_after_unknown_item():
    fd = os.open("/dev/null", os.O_RDONLY)

    class Channel:
        def recvmsg(self, *args):
            return (
                b"x" * protocol.PACKET.size,
                [
                    (999, 999, b"unknown"),
                    (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]).tobytes()),
                ],
                0,
                None,
            )

    with pytest.raises(ValueError):
        protocol._receive(Channel())
    with pytest.raises(OSError):
        os.fstat(fd)


def test_destination_allowance_is_checked_before_allocation(monkeypatch):
    data = fixture_bytes()
    fresh = make_learner(1)
    monkeypatch.setattr(
        stream, "_allocate_record", lambda *a: pytest.fail("destination allowance bypassed")
    )
    with artifact(data) as (fd, receipt), pytest.raises(ValueError, match="aggregate"):
        fresh.restore_handoff_stream(
            fd, receipt, expected_version=1, binding=BINDING, budget=budget(cpu_destinations=0)
        )
    assert_failed(fresh)


def test_post_accept_cleanup_failure_retains_pending_handle(monkeypatch):
    owned = owner(timeout=2)
    source = install_source(owned)
    accept = protocol._accept
    # A valid version2 payload lets acceptance finish before injected cleanup fails.
    data = fixture_bytes()
    header = stream.HEADER.size
    length = stream.HEADER.unpack(data[:header])[1]
    manifest = json.loads(data[header : header + length])
    for key, value in manifest["state"][1]:
        if key == ["atom", "version"]:
            value[1] = 2
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    newdata = stream.HEADER.pack(stream.MAGIC, len(encoded)) + encoded + data[header + length :]

    class BadPipe:
        def __init__(self, pipe):
            self.pipe = pipe

        def close(self):
            self.pipe.close()
            raise OSError("injected pipe close uncertainty")

    def wrapped(*args, **kwargs):
        result = accept(*args, **kwargs)
        owned.last_process.stdout = BadPipe(owned.last_process.stdout)
        return result

    monkeypatch.setattr(protocol, "_accept", wrapped)
    body = "os.close(fd); p._send(c,p._packet(p.ACK,n,r)); _,rights=p._receive(c)\n"
    body += "from rl.learner.handoff_stream import _write\n"
    body += "_write(rights[0], bytes.fromhex(" + repr(newdata.hex()) + "))\n"
    body += "state.fcntl.fcntl(rights[0],state.F_ADD_SEALS,state.SEALS)\nc.close()\n"
    try:
        with pytest.raises(OSError, match="close uncertainty"):
            owned.advance_consuming(
                source, stdlib_worker(body), version=2, binding=BINDING, budget=budget()
            )
        assert owned.failed and not owned.snapshots
        assert owned.pending_fd is not None and os.fstat(owned.pending_fd).st_size
    finally:
        if owned.pending_fd is not None:
            os.close(owned.pending_fd)


def test_received_right_close_uncertainty_poison_retains_handles(monkeypatch):
    owned = owner(timeout=2)
    source = install_source(owned)
    close = protocol._close_identity

    def fail_received(fd, identity):
        if identity == (source.receipt.device, source.receipt.inode) and fd != source.fd:
            raise OSError("received right close uncertain")
        close(fd, identity)

    monkeypatch.setattr(protocol, "_close_identity", fail_received)
    try:
        with pytest.raises(protocol.UncertainRights):
            owned.advance_consuming(
                source,
                stdlib_worker("p._send(c,p._packet(p.ACK,n,r),fd); time.sleep(5)"),
                version=2,
                binding=BINDING,
                budget=budget(),
            )
        assert owned.failed and owned.snapshots.get(source.fd) is source
        assert len(owned._uncertain_rights) == 1
        assert os.fstat(owned._uncertain_rights[0][0])
    finally:
        for fd, _ in getattr(owned, "_uncertain_rights", []):
            os.close(fd)
        os.close(source.fd)


def assert_owner_permanently_failed(owned, source=None):
    assert owned.failed
    operations = [
        lambda: owned.capture_stream([], version=1, binding=BINDING, budget=budget()),
        lambda: owned.capture([], version=1, binding=BINDING),
        lambda: owned.advance_consuming(source, [], version=2, binding=BINDING, budget=budget()),
        lambda: owned.consume(source, []),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError, match="closed/failed/busy"):
            operation()


@pytest.mark.parametrize("when", ["before-delete", "after-delete"])
def test_retirement_registry_interrupt_quarantines_consumed_source(monkeypatch, when):
    owned = owner(timeout=2)
    source = install_source(owned)

    class InterruptedRegistry(dict):
        def __delitem__(self, key):
            if when == "after-delete":
                super().__delitem__(key)
            raise KeyboardInterrupt("injected source registry interruption")

    owned.snapshots = InterruptedRegistry(owned.snapshots)
    monkeypatch.setattr(
        protocol.os,
        "memfd_create",
        lambda *a, **k: pytest.fail("output created before retirement committed"),
    )
    body = "os.close(fd); p._send(c,p._packet(p.ACK,n,r)); time.sleep(5)"
    with pytest.raises(KeyboardInterrupt, match="registry interruption"):
        owned.advance_consuming(
            source, stdlib_worker(body), version=2, binding=BINDING, budget=budget()
        )
    assert_owner_permanently_failed(owned, source)
    assert not owned.snapshots and owned.pending_fd is None
    retirement = owned._serial_retirement
    assert retirement.source is source and retirement.close_confirmed
    assert retirement.phase == "closed"
    assert retirement.reserved_slots == retirement.reserved_bytes == 0
    protocol.prove_absent(os.getpid(), source.receipt)
    with pytest.raises(OSError):
        os.fstat(source.fd)
    assert owned.last_process.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(owned.last_process.pid, os.WNOHANG)
    assert not any(e.get("stage") == "output-granted" for e in owned.events)


@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt])
def test_initial_output_identity_failure_retains_owned_pending(monkeypatch, exception):
    owned = owner()
    create, fstat = protocol.os.memfd_create, protocol.os.fstat
    created = []

    def actual_create(*args, **kwargs):
        fd = create(*args, **kwargs)
        created.append(fd)
        return fd

    def fail_identity(fd):
        if created and fd == created[-1]:
            raise exception("injected initial identity failure")
        return fstat(fd)

    monkeypatch.setattr(protocol.os, "memfd_create", actual_create)
    monkeypatch.setattr(protocol.os, "fstat", fail_identity)
    try:
        with pytest.raises(exception, match="identity failure"):
            owned.capture_stream(
                [], version=1, binding=BINDING, budget=budget(weights_bytes=1, weights_slots=1)
            )
        assert_owner_permanently_failed(owned)
        lease = owned._serial_pending
        assert owned.pending_fd == lease.fd == created[0]
        assert lease.identity is None and lease.phase == "identity-unknown"
        assert lease.reserved_bytes == CAP and lease.reserved_slots == 1
        assert lease.reserved_slots + owned._serial_weights[1] == 2
        assert fstat(created[0]).st_size == 0 and not owned.snapshots
    finally:
        for fd in created:
            os.close(fd)  # exact test-owned descriptor, no production cleanup claim


def test_output_acquisition_unknown_result_retains_reservation(monkeypatch):
    owned = owner()
    create = protocol.os.memfd_create
    created = []

    def unknown_return(*args, **kwargs):
        created.append(create(*args, **kwargs))
        raise KeyboardInterrupt("injected unknown syscall return")

    monkeypatch.setattr(protocol.os, "memfd_create", unknown_return)
    try:
        with pytest.raises(KeyboardInterrupt, match="unknown syscall"):
            owned.capture_stream([], version=1, binding=BINDING, budget=budget())
        assert_owner_permanently_failed(owned)
        lease = owned._serial_pending
        assert lease.fd is lease.identity is None and lease.phase == "acquiring"
        assert lease.reserved_slots == 1 and lease.reserved_bytes == CAP
        assert owned.pending_fd is None and not owned.snapshots
        assert os.fstat(created[0]).st_size == 0
    finally:
        for fd in created:
            os.close(fd)


def test_initial_output_registration_interrupt_tracks_fd():
    class InterruptedOwner(state.SnapshotOwner):
        armed = False

        def __setattr__(self, name, value):
            if name == "pending_fd" and value is not None and self.armed:
                self.armed = False
                raise KeyboardInterrupt("injected pending registration interruption")
            super().__setattr__(name, value)

    owned = InterruptedOwner(max_bytes=CAP, other_resident_bytes=1_000_000_000)
    owned.armed = True
    try:
        with pytest.raises(KeyboardInterrupt, match="registration interruption"):
            owned.capture_stream([], version=1, binding=BINDING, budget=budget())
        assert_owner_permanently_failed(owned)
        lease = owned._serial_pending
        assert lease.fd == owned.pending_fd and lease.identity is None
        assert lease.reserved_slots == 1 and lease.reserved_bytes == CAP
        assert os.fstat(lease.fd).st_size == 0
    finally:
        if owned.pending_fd is not None:
            os.close(owned.pending_fd)


def test_continuation_output_identity_failure_retains_pending(monkeypatch):
    owned = owner(timeout=2)
    source = install_source(owned)
    create, fstat = protocol.os.memfd_create, protocol.os.fstat
    created = []

    def actual_create(*args, **kwargs):
        fd = create(*args, **kwargs)
        created.append(fd)
        return fd

    def fail_identity(fd):
        if fd in created:
            raise OSError("continuation output identity failure")
        return fstat(fd)

    monkeypatch.setattr(protocol.os, "memfd_create", actual_create)
    monkeypatch.setattr(protocol.os, "fstat", fail_identity)
    body = "os.close(fd); p._send(c,p._packet(p.ACK,n,r)); time.sleep(5)"
    try:
        with pytest.raises(OSError, match="output identity failure"):
            owned.advance_consuming(
                source, stdlib_worker(body), version=2, binding=BINDING, budget=budget()
            )
        assert_owner_permanently_failed(owned, source)
        assert not owned.snapshots and owned._serial_retirement.close_confirmed
        lease = owned._serial_pending
        assert owned.pending_fd == lease.fd == created[0] and lease.identity is None
        assert lease.reserved_slots == 1 and lease.reserved_bytes == CAP
        assert fstat(lease.fd).st_size == 0
        assert owned.last_process.returncode is not None
        assert not any(e.get("stage") == "output-granted" for e in owned.events)
    finally:
        for fd in created:
            os.close(fd)


def test_initial_output_close_uncertainty_preserves_primary(monkeypatch):
    owned = owner()
    close = protocol._close_identity
    replacement = []

    def failed_child(*args, **kwargs):
        raise ValueError("injected initial capture failure")

    def uncertain_release(fd, identity):
        close(fd, identity)
        replacement.append(os.open("/dev/null", os.O_RDONLY))
        raise OSError("injected initial close uncertainty")

    monkeypatch.setattr(owned, "_child", failed_child)
    monkeypatch.setattr(protocol, "_close_identity", uncertain_release)
    try:
        with pytest.raises(BaseExceptionGroup) as caught:
            owned.capture_stream([], version=1, binding=BINDING, budget=budget())
        assert [type(e) for e in caught.value.exceptions] == [ValueError, OSError]
        assert_owner_permanently_failed(owned)
        assert not owned.snapshots and owned.pending_fd == owned._serial_pending.fd
        assert owned._serial_pending.phase == "releasing"
        assert owned._serial_pending.reserved_slots == 1
        assert os.fstat(replacement[0])
    finally:
        for fd in replacement:
            os.close(fd)
