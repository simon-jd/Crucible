"""Real tiny production phase ownership and failure transitions; CPU only."""

import hashlib
import json
import os

import pytest

pytest.importorskip("torch")
from rl.learner import handoff_protocol as protocol
from rl.learner.tests.handoff_worker import BINDING, group, summary
from rl.rollout.sampler import Prompt, SamplingConfig
from rl.sync.phase_bridge import PhaseBridge
from rl.sync.tests.ram_worker import CPUWorker, command, contract, make_learner, serial_budget
from rl.sync.tests.test_ram_weights import budget


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    name = "phase-" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    return tmp_path_factory.mktemp(name, numbered=False)


def bridge(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    content = b'{"model_type":"qwen2"}'
    (assets / "config.json").write_bytes(content)
    return PhaseBridge(
        budget=budget(),
        contract=contract(),
        model_path=assets,
        base_revision="tiny-fixture",
        assets=(("config.json", hashlib.sha256(content).hexdigest()),),
        runtime_root=tmp_path / "runtime",
    )


def initial(owned):
    return owned.capture_initial(
        command("initial"), version=1, binding=BINDING, budget=serial_budget()
    )


def advance(owned, version, *, mode="advance", headers=None):
    return owned.advance_learner(
        command(mode),
        version=version,
        binding=BINDING,
        budget=serial_budget(),
        sample_versions=[[version - 1] * 2] if headers is None else headers,
    )


def test_two_real_cycles_adamw_bf16_reference_packed_receiver(tmp_path):
    expected = make_learner()
    expected.step([group(0)])
    owned = bridge(tmp_path)
    try:
        receipt = initial(owned)
        for version in (1, 2):
            producer = owned._source_producer
            with pytest.raises(ChildProcessError):
                os.waitpid(producer.pid, os.WNOHANG)
            checkpoint = owned.export_inference()
            assert owned._source_producer is producer
            assert owned.audit["slots"] == 2
            assert checkpoint.policy_version == version
            with pytest.raises(RuntimeError, match="retire"):
                advance(owned, version + 1)
            with pytest.raises(RuntimeError, match="retire"):
                owned.export_inference()
            assert owned.start_rollout(SamplingConfig(), _backend_factory=CPUWorker) == version
            samples = owned.generate([Prompt("p", "prompt")], 2, version)
            assert len(samples) == 2 and all(s.policy_version == version for s in samples)
            receipt.validate(owned._source.fd)
            worker = owned._engine._backend._process
            owned.retire_rollout()
            assert owned.audit["slots"] == 1
            with pytest.raises(ChildProcessError):
                os.waitpid(worker.pid, os.WNOHANG)
            expected_loss = expected.step([group(version)])["loss"]
            receipt = advance(owned, version + 1)
            assert json.loads(owned._owner.last_output)["state"] == summary(expected)
            assert json.loads(owned._owner.last_output)["loss"] == expected_loss
            assert receipt.version == version + 1
        assert owned.audit["events"][0]["exitcode"] == 0
    finally:
        owned.close()
    assert owned.audit["closed"] and owned.audit["slots"] == 0


def assert_noop_retained(owned, version, headers, monkeypatch):
    source, producer = owned._source, owned._source_producer
    process, highwater = owned._owner.last_process, owned._highwater
    snapshots = dict(owned._owner.snapshots)
    before = owned.audit
    before.pop("noop_dropped_stale", None)
    with monkeypatch.context() as blocked:

        def unexpected(*args, **kwargs):
            pytest.fail("known no-op must not consume or spawn")

        blocked.setattr(owned._owner, "advance_consuming", unexpected)
        blocked.setattr(owned._owner, "_child", unexpected)
        assert advance(owned, version, headers=headers) == source.receipt
    assert owned._source is source and owned._source_producer is producer
    assert owned._owner.last_process is process and owned._highwater == highwater
    assert owned._owner.snapshots == snapshots
    source.receipt.validate(source.fd)  # same sealed optimizer/model/RNG bytes
    after = owned.audit
    after.pop("noop_dropped_stale", None)
    assert after == before


def test_known_noop_does_not_spawn_or_retire(tmp_path, monkeypatch):
    owned = bridge(tmp_path)
    try:
        initial(owned)
        counters = [owned.audit.get("noop_dropped_stale")]
        assert_noop_retained(owned, 2, [], monkeypatch)
        counters.append(owned.audit.get("noop_dropped_stale"))
        producer = owned._source_producer
        with pytest.raises(ValueError, match="group"):
            advance(owned, 2, headers=[[2, 2]])
        # Age exactly1 is accepted by the existing learner's group filter.
        advance(owned, 2, mode="advance-stale", headers=[[0, 0]])
        assert owned._source_producer is not producer
        assert json.loads(owned._owner.last_output)["metrics"]["dropped_stale"] == 0
        counters.append(owned.audit.get("noop_dropped_stale"))
        # No batch IDs in headers: each valid resubmission counts its four drops.
        for _ in range(2):
            assert_noop_retained(owned, 3, [[0, 0], [0, 0]], monkeypatch)
            counters.append(owned.audit.get("noop_dropped_stale"))
        assert_noop_retained(owned, 3, [], monkeypatch)
        counters.append(owned.audit.get("noop_dropped_stale"))
        assert counters == [0, 0, 0, 4, 8, 8]
    finally:
        owned.close()


def test_mixed_stale_accounting_owned_by_learner_and_submission_retry(tmp_path, monkeypatch):
    owned = bridge(tmp_path)
    try:
        initial(owned)
        advance(owned, 2)
        assert_noop_retained(owned, 3, [[0, 0]], monkeypatch)
        counters = [owned.audit.get("noop_dropped_stale")]
        expected = make_learner()
        expected.step([group(0)])
        expected.step([group(1)])
        learner_drops = 0
        # Same headers are two submissions, with valid new target versions.
        for version in (3, 4):
            advance(owned, version, mode="advance-mixed", headers=[[2, 2], [0, 0]])
            result = json.loads(owned._owner.last_output)
            learner_drops += result["metrics"]["dropped_stale"]
            assert result["metrics"]["dropped_stale"] == 2
            assert result["metrics"]["accepted_samples"] == 2
            assert result["metrics"]["optimizer_steps"] == 1
            expected.step([group(2)])
            assert result["state"] == summary(expected)
            before = owned.audit
            with pytest.raises(ValueError, match="strict new"):
                advance(owned, version, headers=[[2, 2], [0, 0]])
            assert owned.audit == before  # invalid target retry is not a submission
            counters.append(owned.audit.get("noop_dropped_stale"))
        assert_noop_retained(owned, 5, [[2, 2], [0, 0]], monkeypatch)
        counters.append(owned.audit.get("noop_dropped_stale"))
        assert counters == [2, 2, 2, 6]
        assert learner_drops + owned.audit["noop_dropped_stale"] == 10
    finally:
        owned.close()


def test_invalid_headers_cannot_partially_account_stale_drops(tmp_path, monkeypatch):
    owned = bridge(tmp_path)
    try:
        initial(owned)
        advance(owned, 2)
        assert_noop_retained(owned, 3, [[0, 0]], monkeypatch)
        before, source, producer = owned.audit, owned._source, owned._source_producer
        invalid = [
            None,
            [[0, 0]] * 1025,
            *[[[0, 0], tail] for tail in ([3, 3], [1], [1, 2], [True, True], [-1, -1], ["0", "0"])],
        ]
        with monkeypatch.context() as blocked:

            def unexpected(*args, **kwargs):
                pytest.fail("invalid headers must not consume or spawn")

            blocked.setattr(owned._owner, "advance_consuming", unexpected)
            blocked.setattr(owned._owner, "_child", unexpected)
            for headers in invalid:
                with pytest.raises(ValueError):
                    owned.advance_learner(
                        command("advance"),
                        version=3,
                        binding=BINDING,
                        budget=serial_budget(),
                        sample_versions=headers,
                    )
                assert owned.audit == before
                assert owned._source is source and owned._source_producer is producer
                source.receipt.validate(source.fd)
            for version in (3.0, True, "3"):
                with pytest.raises(ValueError, match="strict new"):
                    owned.advance_learner(
                        command("advance"),
                        version=version,
                        binding=BINDING,
                        budget=serial_budget(),
                        sample_versions=[[0, 0]],
                    )
                assert owned.audit == before
                assert owned._source is source and owned._source_producer is producer
                source.receipt.validate(source.fd)
        assert owned.audit.get("noop_dropped_stale") == 2
    finally:
        owned.close()


def test_unexpected_worker_noop_keeps_irreversible_failure(tmp_path):
    owned = bridge(tmp_path)
    initial(owned)
    with pytest.raises(ValueError):
        advance(owned, 2, mode="noop")
    assert owned.audit["failed"] and owned._owner.failed
    assert not owned._owner.snapshots
    with pytest.raises(RuntimeError):
        owned.export_inference()


@pytest.mark.parametrize("fault", ["acquire", "identity", "child"])
def test_export_unknown_reservation_poison(tmp_path, monkeypatch, fault):
    owned = bridge(tmp_path)
    initial(owned)
    real_create, real_stat = os.memfd_create, os.fstat
    allocated = []

    def create(*args, **kwargs):
        fd = real_create(*args, **kwargs)
        allocated.append(fd)
        if fault == "acquire":
            raise KeyboardInterrupt("unknown acquisition")
        return fd

    def stat(fd):
        if fault == "identity" and fd in allocated:
            raise OSError("unknown identity")
        return real_stat(fd)

    def child(*args, **kwargs):
        raise OSError("ambiguous process creation")

    monkeypatch.setattr(os, "memfd_create", create)
    monkeypatch.setattr(os, "fstat", stat)
    if fault == "child":
        monkeypatch.setattr(owned._owner, "_child", child)
    try:
        with pytest.raises((KeyboardInterrupt, OSError)):
            owned.export_inference()
        assert owned.audit["failed"] and owned.audit["weight_reserved"] == budget().max_bytes
        with pytest.raises(RuntimeError):
            advance(owned, 2)
        with pytest.raises(RuntimeError):
            owned.start_rollout(_backend_factory=CPUWorker)
    finally:
        monkeypatch.undo()
        for fd in allocated:
            os.close(fd)  # exact fixture-owned capability, not a production cleanup claim
        os.close(owned._source.fd)


def test_source_close_uncertain_retains_failure_and_reused_fd(tmp_path, monkeypatch):
    owned = bridge(tmp_path)
    initial(owned)
    close = protocol._close_identity
    replacements = []

    def uncertain(fd, identity):
        close(fd, identity)
        replacements.append(os.open("/dev/null", os.O_RDONLY))
        raise OSError("close uncertain")

    monkeypatch.setattr(protocol, "_close_identity", uncertain)
    try:
        with pytest.raises(OSError):
            owned.close()
        assert owned.audit["failed"] and not owned._source_retirement.close_confirmed
        assert os.fstat(replacements[0])
        with pytest.raises(RuntimeError):
            owned.export_inference()
    finally:
        for fd in replacements:
            os.close(fd)


def test_extra_parent_weight_holder_blocks_retirement(tmp_path):
    owned = bridge(tmp_path)
    initial(owned)
    owned.export_inference()
    duplicate = os.dup(owned._weight.fd)
    try:
        with pytest.raises(RuntimeError):
            owned.retire_rollout()
        assert owned.audit["failed"] and owned.audit["slots"] == 2
    finally:
        os.close(duplicate)
        owned.close()


def test_post_source_close_registration_interrupt_is_poisoned(tmp_path, monkeypatch):
    owned = bridge(tmp_path)
    receipt = initial(owned)
    original = PhaseBridge.__setattr__
    armed = [True]

    def interrupted(self, name, value):
        if self is owned and name == "_source" and value is None and armed[0]:
            armed[0] = False
            raise KeyboardInterrupt("post-close registry interruption")
        original(self, name, value)

    monkeypatch.setattr(PhaseBridge, "__setattr__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        owned.close()
    assert owned.audit["failed"] and owned._source_retirement.close_confirmed
    protocol.prove_absent(os.getpid(), receipt)
    with pytest.raises(RuntimeError):
        owned.export_inference()
    owned.close()  # retry never closes the retired/reusable FD integer


def test_changed_assets_before_start_cannot_relabel_checkpoint(tmp_path):
    owned = bridge(tmp_path)
    initial(owned)
    owned.export_inference()
    (tmp_path / "assets/config.json").write_text("{}")
    try:
        with pytest.raises(ValueError, match="asset"):
            owned.start_rollout(_backend_factory=CPUWorker)
    finally:
        owned.close()
