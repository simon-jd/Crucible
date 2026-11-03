"""Fixed local serial consumption protocol. Parent imports no Torch.

The exact unreaped Popen child is the ownership anchor. Workers are trusted,
non-spawning and forbidden to transfer/reacquire source capabilities externally.
"""

import array
import json
import os
import secrets
import selectors
import socket
import struct
import subprocess
import time
from dataclasses import asdict, dataclass

from rl.learner import state_handoff as state
from rl.learner.handoff_stream import SerialBudget, read_manifest

PACKET = struct.Struct("<8s16sQQQQ32sQQ")
ACK = b"RETIRED1"
GRANT = b"OUTPUT01"
MAX_FDS = 4096
MAX_MAPS = 100000


def _packet(kind, nonce, receipt, output=(0, 0)):
    return PACKET.pack(
        kind,
        nonce,
        receipt.device,
        receipt.inode,
        receipt.size,
        receipt.version,
        bytes.fromhex(receipt.sha256),
        *output,
    )


class UncertainRights(RuntimeError):
    def __init__(self, descriptors, errors):
        super().__init__("received descriptor release uncertain")
        self.descriptors, self.errors = descriptors, errors


def _retain_uncertain(owner, error):
    if isinstance(error, UncertainRights):
        owner.failed = True
        owner._uncertain_rights = getattr(owner, "_uncertain_rights", []) + error.descriptors
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            _retain_uncertain(owner, child)


def _close_rights(fds):
    errors, uncertain = [], []
    for fd in fds:
        identity = None
        try:
            identity = os.fstat(fd)
            _close_identity(fd, (identity.st_dev, identity.st_ino))
        except BaseException as error:  # noqa: BLE001 - retain uncertain descriptor releases
            errors.append(error)
            uncertain.append((fd, None if identity is None else (identity.st_dev, identity.st_ino)))
    if errors:
        raise UncertainRights(uncertain, errors) from BaseExceptionGroup(
            "descriptor cleanup failures", errors
        )


def _ancillary(ancillary):
    # Collect ALL installed rights before rejecting any earlier malformed item.
    fds, invalid = [], False
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i")
            complete = len(value) - len(value) % values.itemsize
            values.frombytes(value[:complete])
            fds.extend(values)
            invalid |= complete != len(value)
        else:
            invalid = True
    return fds, invalid


def _receive(channel):
    data, ancillary, flags, _ = channel.recvmsg(
        PACKET.size + 1, socket.CMSG_SPACE(16), socket.MSG_CMSG_CLOEXEC
    )
    fds, invalid = _ancillary(ancillary)
    try:
        if invalid or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(data) != PACKET.size:
            raise ValueError("invalid/truncated/closed/oversized protocol packet")
        return data, fds
    except BaseException as error:
        try:
            _close_rights(fds)
        except BaseException as cleanup:  # noqa: BLE001 - preserve primary and cleanup causes
            raise BaseExceptionGroup("protocol and descriptor cleanup failed", [error, cleanup])
        raise


def _send(channel, data, fd=None):
    ancillary = (
        [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    )
    if channel.sendmsg([data], ancillary) != len(data):
        raise OSError("partial protocol grant; transfer is uncertain")


def _starttime(pid):
    with open(f"/proc/{pid}/stat", "rb") as source:
        data = source.read(8193)
    if len(data) > 8192:
        raise RuntimeError("process stat allowance")
    return int(data[data.rfind(b")") + 2 :].split()[19])


def _live(process, starttime):
    if process.returncode is not None:
        raise RuntimeError("original process ownership anchor was reaped")
    result = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if result is not None or _starttime(process.pid) != starttime:
        raise RuntimeError("original child exited/changed before retirement")


def prove_absent(pid, receipt, allowed_fd=None):
    """Kernel fd+maps proof; no fd-close/GC inference, no PID ownership inference."""
    count, matches = 0, []
    with os.scandir(f"/proc/{pid}/fd") as entries:
        for entry in entries:
            count += 1
            if count > MAX_FDS:
                raise RuntimeError("fd proof allowance exceeded")
            # Our own scandir FD may disappear only after this iterator closes.
            info = os.stat(entry.path)
            if (info.st_dev, info.st_ino) == (receipt.device, receipt.inode):
                matches.append(int(entry.name))
    if matches != ([] if allowed_fd is None else [allowed_fd]):
        raise RuntimeError("source descriptor aliases remain")
    lines = 0
    with open(f"/proc/{pid}/maps", "rb") as source:
        while line := source.readline(16385):
            lines += 1
            if lines > MAX_MAPS or len(line) > 16384:
                raise RuntimeError("maps proof allowance exceeded")
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                raise RuntimeError("malformed maps proof")
            major, minor = (int(v, 16) for v in fields[3].split(b":"))
            if (os.makedev(major, minor), int(fields[4])) == (receipt.device, receipt.inode):
                raise RuntimeError("source mapping remains")
    return {"fd_count": count, "map_count": lines, "source_fds": matches}


def retire_input_and_receive_output(source_fd, receipt, control_fd, nonce):
    """Called after full successful restore; consumes source even on later failure."""
    _close_identity(source_fd, (receipt.device, receipt.inode))
    prove_absent(os.getpid(), receipt)
    with socket.socket(fileno=control_fd) as channel:
        _send(channel, _packet(ACK, nonce, receipt))
        data, fds = _receive(channel)
        try:
            if len(fds) != 1:
                raise ValueError("grant requires exactly one output descriptor")
            info = os.fstat(fds[0])
            if data != _packet(GRANT, nonce, receipt, (info.st_dev, info.st_ino)):
                raise ValueError("wrong output grant")
            if info.st_size or state.fcntl.fcntl(fds[0], state.F_GET_SEALS):
                raise ValueError("grant output is not empty and unsealed")
            return fds.pop()
        finally:
            _close_rights(fds)


def _close_identity(fd, identity):
    info = os.fstat(fd)
    if (info.st_dev, info.st_ino) != identity:
        raise RuntimeError("descriptor identity changed; refuse cleanup of reused FD")
    os.close(fd)


def _accept(owner, fd, identity, version, binding, budget):
    info = os.fstat(fd)
    if not 0 < info.st_size <= budget.max_bytes or (info.st_dev, info.st_ino) != identity:
        raise ValueError("stream output identity/size")
    receipt = state.Receipt(
        *identity,
        info.st_size,
        state._digest(fd, info.st_size),
        state.fcntl.fcntl(fd, state.F_GET_SEALS),
        version,
        state._binding(binding),
        budget.max_bytes,
        budget.metadata_cap,
        budget.storage_cap,
    )
    payload, _, _ = read_manifest(fd, receipt, budget)
    if (
        type(payload) is not dict
        or payload.get("format") != state.FORMAT
        or type(payload.get("version")) is not int
        or payload["version"] != version
        or payload.get("binding") != receipt.binding
    ):
        raise ValueError("stream accepted format/version/binding mismatch")
    result = state.Snapshot(fd, receipt)
    owner.snapshots[fd] = result
    return result


def _admit(owner, budget, source=None):
    if len(owner.events) > 1000:
        raise ValueError("serial audit event allowance exceeded")
    if type(budget) is not SerialBudget:
        raise TypeError("explicit SerialBudget required")
    previous_weights = getattr(owner, "_serial_weights", (0, 0))
    declared_weights = (budget.weights_bytes, budget.weights_slots)
    if previous_weights != (0, 0) and previous_weights != declared_weights:
        raise ValueError("existing weights reservation cannot be omitted/changed")
    actual_retained = sum(s.receipt.size for s in owner.snapshots.values() if s is not source)
    if budget.retained_bytes != actual_retained:
        raise ValueError("serial retained-byte declaration mismatch")
    count = owner.disk_count + len(owner.snapshots) + budget.weights_slots
    if count + (source is None) > 2:
        raise ValueError("at most two checkpoints including weights/pending")
    budget.check(0 if source is None else source.receipt.size)
    owner._serial_weights = declared_weights
    owner._serial_budget = budget


@dataclass
class OutputLease:
    budget: SerialBudget
    fd: int | None = None
    identity: tuple | None = None
    phase: str = "reserved"
    reserved_slots: int = 1

    @property
    def reserved_bytes(self):
        return self.budget.max_bytes if self.reserved_slots else 0


@dataclass
class SourceRetirement:
    source: state.Snapshot
    phase: str = "entered"
    close_confirmed: bool = False

    @property
    def reserved_slots(self):
        return 0 if self.close_confirmed else 1

    @property
    def reserved_bytes(self):
        return 0 if self.close_confirmed else self.source.receipt.size


def _acquire_output(owner, budget):
    # Arm failure BEFORE the acquisition boundary. An exception even before the
    # returned fd can be assigned leaves a poisoned, explicitly reserved unknown
    # acquisition; do not discover/close arbitrary fds by integer or /proc diff.
    owner.failed = True
    lease = OutputLease(budget)
    owner._serial_pending = lease
    fd = None
    try:
        lease.phase = "acquiring"
        fd = os.memfd_create("crucible-stream", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        lease.fd = fd
        owner.pending_fd = fd
        lease.phase = "identity-unknown"
        info = os.fstat(fd)
        lease.identity = (info.st_dev, info.st_ino)
        lease.phase = "owned"
        return lease
    except BaseException:
        # Also covers interruption between local assignment and owner tracking.
        # If the syscall outcome itself is unknown, fd=None is explicit; the
        # reserved slot/bytes remain charged and owner can never admit another.
        if fd is not None:
            lease.fd = fd
            owner.pending_fd = fd
        raise


def _release_output(owner, lease):
    if lease.fd is None or lease.identity is None:
        return  # unknown acquisition/identity: retain poisoned reservation
    lease.phase = "releasing"
    _close_identity(lease.fd, lease.identity)
    lease.phase = "closed"
    lease.reserved_slots = 0
    owner.pending_fd = None


def _retire_source(owner, source):
    # This guard precedes every destructive operation and is cleared ONLY after
    # successful replacement acceptance and all process/control cleanup.
    owner.failed = True
    retirement = SourceRetirement(source)
    owner._serial_retirement = retirement
    try:
        _close_identity(source.fd, (source.receipt.device, source.receipt.inode))
        retirement.close_confirmed = True
        retirement.phase = "closed"
        del owner.snapshots[source.fd]
        retirement.phase = "retired"
        return retirement
    except BaseException:
        # A closed or close-uncertain source is quarantined, never advertised as
        # an accepted resume. Base dict.pop deliberately bypasses a failing
        # registry __delitem__; identity/reservation remain in retirement.
        dict.pop(owner.snapshots, source.fd, None)
        if not retirement.close_confirmed:
            retirement.phase = "close-uncertain"
        raise


def capture_initial(owner, command, *, version, binding, budget):
    owner._ready()
    state._integer(version, "version")
    _admit(owner, budget)
    owner.busy = True
    previous_process = owner.last_process
    previous_lease = getattr(owner, "_serial_pending", None)
    lease = None
    try:
        lease = _acquire_output(owner, budget)
        owner.last_output = owner._child(
            command,
            lease.fd,
            {
                "CRUCIBLE_HANDOFF_VERSION": str(version),
                "CRUCIBLE_HANDOFF_BINDING": state._binding(binding),
                "CRUCIBLE_STREAM_BUDGET": json.dumps(asdict(budget)),
            },
        )
        result = _accept(owner, lease.fd, lease.identity, version, binding, budget)
        lease.phase = "accepted"
        lease.reserved_slots = 0  # accepted registry now carries the slot
        owner.pending_fd = None
        owner.failed = False
        return result
    except BaseException as error:
        owner.failed = True
        current = getattr(owner, "_serial_pending", None)
        if current is not previous_lease:
            lease = current
        if lease is not None:
            # Re-arm reservation if acceptance itself was interrupted.
            lease.reserved_slots = 1
            if lease.fd is not None:
                owner.pending_fd = lease.fd
                dict.pop(owner.snapshots, lease.fd, None)
            process = owner.last_process
            if process is previous_process or process.returncode is not None:
                try:
                    _release_output(owner, lease)
                except BaseException as cleanup:  # noqa: BLE001 - retain primary and cleanup uncertainty
                    raise BaseExceptionGroup("capture and cleanup failed", [error, cleanup])
        raise
    finally:
        owner.busy = False


def _cleanup_process(process):
    # returncode is only set by this function or the successful final wait below.
    if process.returncode is None:
        pending = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if pending is None:
            process.terminate()
        try:
            process.wait(timeout=2.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.5)


def advance_consuming(owner, source, command, *, version, binding, budget):
    owner._ready()
    if owner.snapshots.get(source.fd) is not source:
        raise ValueError("source is not owned")
    state._integer(version, "version")
    if version <= source.receipt.version or state._binding(binding) != source.receipt.binding:
        raise ValueError("serial version/binding mismatch")
    _admit(owner, budget, source)
    read_manifest(source.fd, source.receipt, budget)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
    nonce = secrets.token_bytes(16)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CRUCIBLE_HANDOFF_") and not k.startswith("CRUCIBLE_STREAM_")
    }
    env.update(
        {
            "CRUCIBLE_HANDOFF_SOURCE_FD": str(source.fd),
            "CRUCIBLE_HANDOFF_SOURCE_RECEIPT": json.dumps(asdict(source.receipt)),
            "CRUCIBLE_HANDOFF_VERSION": str(version),
            "CRUCIBLE_HANDOFF_BINDING": state._binding(binding),
            "CRUCIBLE_STREAM_BUDGET": json.dumps(asdict(budget)),
            "CRUCIBLE_STREAM_CONTROL_FD": str(child.fileno()),
            "CRUCIBLE_STREAM_NONCE": nonce.hex(),
        }
    )
    owner.busy = True
    process, output_fd, retired, accepted = None, None, False, False
    errors, output = [], bytearray()
    deadline = time.monotonic() + owner.timeout
    try:
        process = subprocess.Popen(
            command,
            env=env,
            pass_fds=(source.fd, child.fileno()),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        owner.last_process = process
        child.close()
        starttime = _starttime(process.pid)
        owner.events.append(
            {
                "stage": "stream-start",
                "pid": process.pid,
                "starttime": starttime,
                "source": [source.receipt.device, source.receipt.inode],
                "charge": budget.check(source.receipt.size),
            }
        )
        with selectors.DefaultSelector() as selector:
            selector.register(parent, selectors.EVENT_READ)
            selector.register(process.stdout, selectors.EVENT_READ)
            acknowledged, channel_eof, stdout_eof = False, False, False
            while not (channel_eof and stdout_eof):
                if time.monotonic() >= deadline:
                    raise TimeoutError("serial handoff deadline exceeded")
                for key, _ in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                    if key.fileobj is process.stdout:
                        block = os.read(process.stdout.fileno(), 4096)
                        if not block:
                            stdout_eof = True
                            selector.unregister(process.stdout)
                        elif len(output) + len(block) > 65536:
                            raise ValueError("serial child output cap")
                        else:
                            output.extend(block)
                        continue
                    if acknowledged:
                        # After GRANT the only legal control event is EOF.
                        data, anc, flags, _ = parent.recvmsg(
                            PACKET.size + 1, socket.CMSG_SPACE(16), socket.MSG_CMSG_CLOEXEC
                        )
                        rights, _ = _ancillary(anc)
                        _close_rights(rights)
                        if data or anc or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                            raise ValueError("duplicate/extra protocol message")
                        channel_eof = True
                        selector.unregister(parent)
                        continue
                    data, fds = _receive(parent)
                    if fds:
                        _close_rights(fds)
                        raise ValueError("ACK must not contain descriptors")
                    if data != _packet(ACK, nonce, source.receipt):
                        raise ValueError("wrong retirement ACK")
                    _live(process, starttime)
                    child_proof = prove_absent(process.pid, source.receipt)
                    parent_proof = prove_absent(os.getpid(), source.receipt, source.fd)
                    _live(process, starttime)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("deadline before source retirement")
                    owner.events.append(
                        {
                            "stage": "source-proof",
                            "pid": process.pid,
                            "child": child_proof,
                            "parent": parent_proof,
                        }
                    )
                    _retire_source(owner, source)
                    retired = True
                    owner.events.append(
                        {
                            "stage": "source-retired",
                            "pid": process.pid,
                            "accepted": len(owner.snapshots),
                            "weights_slots": budget.weights_slots,
                        }
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("deadline after source retirement")
                    _live(process, starttime)
                    # Source capability is irreversibly gone BEFORE output exists.
                    lease = _acquire_output(owner, budget)
                    output_fd = lease.fd
                    identity = lease.identity
                    if time.monotonic() >= deadline:
                        raise TimeoutError("deadline before output grant")
                    _live(process, starttime)
                    _send(
                        parent,
                        _packet(GRANT, nonce, source.receipt, identity),
                        output_fd,
                    )
                    acknowledged = True
                    owner.events.append(
                        {
                            "stage": "output-granted",
                            "pid": process.pid,
                            "output": list(identity),
                            "slots": owner.disk_count
                            + len(owner.snapshots)
                            + budget.weights_slots
                            + 1,
                        }
                    )
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if code != 0 or not acknowledged:
                raise RuntimeError(f"serial child exited {code}; no accepted replacement")
        owner.last_output = output.decode()
        result = _accept(owner, output_fd, identity, version, binding, budget)
        if time.monotonic() >= deadline:
            owner.snapshots.pop(output_fd, None)
            raise TimeoutError("deadline during output acceptance")
        accepted = True
        return result
    except BaseException as error:
        errors.append(error)
        _retain_uncertain(owner, error)
        retirement = getattr(owner, "_serial_retirement", None)
        if retirement is not None and retirement.source is source:
            owner.failed = True
            dict.pop(owner.snapshots, source.fd, None)
        lease = getattr(owner, "_serial_pending", None)
        if lease is not None and lease.phase != "accepted" and lease.fd is not None:
            dict.pop(owner.snapshots, lease.fd, None)
        raise
    finally:
        uncertain = False
        if process is not None:
            try:
                _cleanup_process(process)
            except BaseException as error:  # noqa: BLE001 - retain reap and interrupt uncertainty
                errors.append(error)
                uncertain = True
                owner.failed = True
            owner.events.append(
                {
                    "stage": "stream-exit",
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "reaped": process.returncode is not None,
                    "retired": retired,
                    "retirement_phase": (
                        getattr(owner, "_serial_retirement", None).phase
                        if getattr(owner, "_serial_retirement", None) is not None
                        else None
                    ),
                    "source_close_confirmed": (
                        getattr(owner, "_serial_retirement", None).close_confirmed
                        if getattr(owner, "_serial_retirement", None) is not None
                        else False
                    ),
                    "accepted": accepted,
                }
            )
        lease = getattr(owner, "_serial_pending", None)
        if lease is not None and lease.phase != "accepted" and not accepted and not uncertain:
            try:
                _release_output(owner, lease)
            except BaseException as error:  # noqa: BLE001 - retain output release uncertainty
                errors.append(error)
                owner.failed = True
        for handle in (parent, child, None if process is None else process.stdout):
            if handle is not None:
                try:
                    handle.close()
                except BaseException as error:  # noqa: BLE001 - retain control and pipe cleanup failures
                    errors.append(error)
                    owner.failed = True
        owner.busy = False
        if errors and accepted:
            owner.snapshots.pop(output_fd, None)
            owner.pending_fd = output_fd
            owner.failed = True
        elif accepted:
            lease.phase = "accepted"
            lease.reserved_slots = 0
            owner.pending_fd = None
            owner.failed = False
        if len(errors) > 1:
            raise BaseExceptionGroup("serial handoff and cleanup failed", errors)
        if errors and accepted:
            raise errors[0]
