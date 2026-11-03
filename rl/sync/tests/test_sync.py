"""Atomic publication, complete tensors, and real two-process transport."""

import json
import multiprocessing as mp
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from rl.rollout.engine import RolloutEngine
from rl.rollout.sampler import SamplingConfig
from rl.sync.nccl_broadcast import WeightSync
from rl.sync.version import VersionStore


def make_store(path, dtype=torch.float32):
    assets = path / "assets"
    assets.mkdir(parents=True)
    (assets / "config.json").write_text(json.dumps({"vocab_size": 3}))
    return VersionStore(path / "versions", {"weight": ((3, 2), dtype)}, assets)


class Backend:
    live = 0
    failures = frozenset()

    def __init__(self, checkpoint, config):
        from safetensors.torch import load_file

        assert Backend.live == 0
        if checkpoint.policy_version in self.failures:
            error = RuntimeError("injected load failure")
            error.cleanup_confirmed = True
            raise error
        self.weight = load_file(str(Path(checkpoint.model_path) / "tensor-00000.safetensors"))[
            "weight"
        ]
        Backend.live += 1

    def close(self):
        Backend.live -= 1


def test_logits_match_after_push(tmp_path):
    store = make_store(tmp_path)
    weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    stats = WeightSync(store).push([("weight", weights)], 7)
    assert stats.bytes == 24 and stats.seconds >= 0
    weights.add_(100)  # publication owns bytes, not caller tensor aliases
    with RolloutEngine(
        store.checkpoint(7), SamplingConfig(), _backend_factory=Backend, weight_store=store
    ) as engine:
        x = torch.tensor([0.3, -0.2])
        torch.testing.assert_close(engine._backend.weight @ x, (weights - 100) @ x)
        newer = weights + 1
        WeightSync(store).push([("weight", newer)], 8)
        engine.load_weights(8)
        torch.testing.assert_close(engine._backend.weight @ x, newer @ x)
        assert engine.policy_version == 8


def test_version_increments(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    sync.push([("weight", torch.ones(3, 2))], 1)
    sync.push([("weight", torch.zeros(3, 2))], 2)
    for bad in (2, 1, True, -1):
        with pytest.raises(ValueError):
            sync.push([("weight", torch.ones(3, 2))], bad)
    assert store.checkpoint(1).revision != store.checkpoint(2).revision


def test_failed_push_keeps_previous_version(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    sync.push([("weight", torch.ones(3, 2))], 1)
    with RolloutEngine(store.checkpoint(1), _backend_factory=Backend, weight_store=store) as engine:

        def broken():
            yield "weight", torch.zeros(3, 2)
            raise RuntimeError("interrupted producer")

        with pytest.raises(RuntimeError, match="interrupted"):
            sync.push(broken(), 2)
        assert engine.policy_version == 1
        assert len(list(store.root.iterdir())) == 1
        sync.push([("weight", torch.zeros(3, 2))], 2)
        Backend.failures = {2}
        try:
            with pytest.raises(RuntimeError, match="injected load"):
                engine.load_weights(2)
            assert engine.policy_version == 1
            assert torch.equal(engine._backend.weight, torch.ones(3, 2))
        finally:
            Backend.failures = set()


@pytest.mark.parametrize(
    "shards",
    [
        [],
        [("other", torch.ones(3, 2))],
        [("weight", torch.ones(3, 3))],
        [("weight", torch.ones(3, 2, dtype=torch.bfloat16))],
        [("weight", torch.full((3, 2), float("nan")))],
        [("weight", torch.ones(3, 2)), ("weight", torch.ones(3, 2))],
    ],
)
def test_incomplete_or_changed_schema_rejected(tmp_path, shards):
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        WeightSync(store).push(shards, 1)
    assert list(store.root.iterdir()) == []


def test_modified_owned_bytes_fail_closed(tmp_path):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    checkpoint = store.checkpoint(1)
    file = Path(checkpoint.model_path) / "config.json"
    file.chmod(0o600)
    file.write_text("{}")
    with pytest.raises(ValueError, match="content"):
        store.checkpoint(1)


def _distributed_rank(rank, root, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        store = make_store(Path(root) / str(rank))
        sync = WeightSync(store, group=dist.group.WORLD)
        data = (
            [("weight", torch.arange(6, dtype=torch.float32).reshape(3, 2))] if rank == 0 else None
        )
        sync.push(data, 1)
        assert store.checkpoint(1).policy_version == 1
        # Receiver schema failure must be acknowledged before payload, on BOTH ranks.
        if rank == 1:
            store.schema["weight"] = ((3, 3), torch.float32)
        try:
            sync.push(data, 2)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("receiver rejection did not abort sender")
        assert not (store.root / "v2").exists()
        store.schema["weight"] = ((3, 2), torch.float32)

        def broken_source():
            yield "weight", torch.ones(3, 2)
            raise RuntimeError("actual source stream interruption")

        with pytest.raises(RuntimeError):
            sync.push(broken_source() if rank == 0 else None, 2)
        from rl.sync.version import Stage

        original = Stage.add

        def fail_write(self, name, tensor):
            if rank == 1:
                raise OSError("actual receiver write rejection")
            return original(self, name, tensor)

        Stage.add = fail_write
        try:
            with pytest.raises(RuntimeError):
                sync.push(data, 2)
        finally:
            Stage.add = original
        assert len(list(store.root.iterdir())) == 1
        sync.push(data, 2)
        assert store.checkpoint(2).policy_version == 2

    finally:
        dist.destroy_process_group()


def test_actual_two_process_gloo(tmp_path):
    context = mp.get_context("spawn")
    children = [
        context.Process(
            target=_distributed_rank, args=(rank, str(tmp_path), str(tmp_path / "rendezvous"))
        )
        for rank in range(2)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(60)
        assert [child.exitcode for child in children] == [0, 0]
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join()


def test_unconfirmed_candidate_start_never_starts_rollback(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    sync.push([("weight", torch.ones(3, 2))], 1)
    sync.push([("weight", torch.zeros(3, 2))], 2)
    calls = []

    class UnknownBackend(Backend):
        def __init__(self, checkpoint, config):
            calls.append(checkpoint.policy_version)
            if checkpoint.policy_version == 2:
                raise RuntimeError("unknown startup ownership")
            super().__init__(checkpoint, config)

    engine = RolloutEngine(store.checkpoint(1), weight_store=store, _backend_factory=UnknownBackend)
    with pytest.raises(RuntimeError, match="unknown startup"):
        engine.load_weights(2)
    assert calls == [1, 2] and engine.policy_version == 1
    with pytest.raises(RuntimeError, match="live owned"):
        engine.load_weights(2)
    with pytest.raises(RuntimeError, match="without a cleanup handle"):
        engine.close()
    assert store.active_version == 1


def test_two_snapshot_bound_and_active_prune(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    sync.push([("weight", torch.ones(3, 2))], 1)
    sync.push([("weight", torch.zeros(3, 2))], 2)
    store.active_version = 1
    with pytest.raises(ValueError, match="active"):
        store.discard(1)
    with pytest.raises(ValueError, match="two snapshots"):
        sync.push([("weight", torch.zeros(3, 2))], 3)
    store.discard(2)
    sync.push([("weight", torch.zeros(3, 2))], 3)
    assert store.checkpoint(1).policy_version == 1


def test_whole_transaction_deadline_cleans_stage(tmp_path):
    import time

    store = make_store(tmp_path)

    def slow():
        time.sleep(0.02)
        yield "weight", torch.ones(3, 2)

    with pytest.raises(TimeoutError, match="transaction deadline"):
        WeightSync(store, timeout=0.01).push(slow(), 1)
    assert list(store.root.iterdir()) == []


def test_export_fresh_ready_guard_and_explicit_cast(tmp_path):
    from rl.sync.export import export_ready_learner_bf16

    store = make_store(tmp_path, torch.bfloat16)

    class Learner:
        ready = True
        version = 1
        model = torch.nn.Linear(2, 3, bias=False)

        @property
        def policy(self):
            if not self.ready:
                raise RuntimeError("failed phase")
            return self.model

        @property
        def policy_version(self):
            if not self.ready:
                raise RuntimeError("failed phase")
            return self.version

    learner = Learner()
    export_ready_learner_bf16(learner, WeightSync(store))
    assert store.checkpoint(1).policy_version == 1
    learner.ready = False
    with pytest.raises(RuntimeError, match="failed phase"):
        export_ready_learner_bf16(learner, WeightSync(store))
    assert len(list(store.root.iterdir())) == 1


def test_export_intervening_failure_never_publishes(tmp_path):
    from rl.sync.export import export_ready_learner_bf16

    store = make_store(tmp_path, torch.bfloat16)

    class Learner:
        calls = 0
        model = torch.nn.Linear(2, 3, bias=False)

        @property
        def policy(self):
            return self.model

        @property
        def policy_version(self):
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("failed phase during export")
            return 1

    with pytest.raises(RuntimeError, match="failed phase"):
        export_ready_learner_bf16(Learner(), WeightSync(store))
    assert list(store.root.iterdir()) == []


def test_discard_cannot_retag_used_version(tmp_path):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    store.discard(1)
    reopened = VersionStore(store.root, store.schema, store.assets)
    with pytest.raises(ValueError, match="increase"):
        WeightSync(reopened).push([("weight", torch.zeros(3, 2))], 1)


def _lost_ack_rank(rank, root, rendezvous):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    from rl.sync.version import Stage

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=3),
    )
    try:
        store = make_store(Path(root) / str(rank))
        sync = WeightSync(store, group=dist.group.WORLD, timeout=5)
        data = [("weight", torch.ones(3, 2))] if rank == 0 else None
        sync.push(data, 1)
        original = Stage.commit

        def die_after_catalog(self):
            original(self)
            if rank == 1:
                os._exit(73)  # own test child: real lost final ACK, not a mocked collective

        Stage.commit = die_after_catalog
        with pytest.raises((RuntimeError, TimeoutError)):
            sync.push(data, 2)
        assert sync.failed and store._in_doubt
        with pytest.raises(RuntimeError, match="in doubt"):
            store.checkpoint(2)
        with pytest.raises(RuntimeError, match="owner failed"):
            sync.push(data, 3)
        # Previous published bytes remain intact despite unknown candidate outcome.
        from rl.sync.version import verify_checkpoint

        verify_checkpoint(store._entries()[1])
    finally:
        dist.destroy_process_group()


def test_actual_gloo_lost_ack_fails_closed(tmp_path):
    context = mp.get_context("spawn")
    children = [
        context.Process(
            target=_lost_ack_rank, args=(rank, str(tmp_path), str(tmp_path / "rendezvous"))
        )
        for rank in range(2)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(20)
        assert [child.exitcode for child in children] == [0, 73]
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join()


def test_midstream_contract_change_rejected(tmp_path):
    store = make_store(tmp_path)

    def shards():
        yield "weight", torch.ones(3, 2)
        store.schema["weight"] = ((3, 3), torch.float32)

    with pytest.raises(ValueError, match="contract changed"):
        WeightSync(store).push(shards(), 1)
    assert list(store.root.iterdir()) == []


def test_cleanup_failure_preserves_cause_and_blocks_reuse(tmp_path, monkeypatch):
    from rl.sync.version import Stage

    store = make_store(tmp_path)
    sync = WeightSync(store)

    def fail_cleanup(_stage):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(Stage, "abort", fail_cleanup)
    with pytest.raises(BaseExceptionGroup) as caught:
        sync.push([], 1)
    assert "incomplete" in str(caught.value.exceptions[0])
    assert "cleanup failure" in str(caught.value.exceptions[1])
    assert sync.failed and store._in_doubt


def test_active_store_pin_released_only_after_confirmed_close(tmp_path):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    with RolloutEngine(store.checkpoint(1), _backend_factory=Backend, weight_store=store):
        with pytest.raises(RuntimeError, match="already has an active"):
            RolloutEngine(store.checkpoint(1), _backend_factory=Backend, weight_store=store)
        with pytest.raises(ValueError, match="active"):
            store.discard(1)
    store.discard(1)


def test_candidate_cleanup_unknown_retains_handle_and_blocks_rollback(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    for version in (1, 2):
        sync.push([("weight", torch.ones(3, 2) * version)], version)

    class Pending:
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("cleanup still unconfirmed")

    pending, starts = Pending(), []

    def factory(checkpoint, config):
        starts.append(checkpoint.policy_version)
        if checkpoint.policy_version == 2:
            error = RuntimeError("candidate startup uncertain")
            error.cleanup_confirmed, error.worker = False, pending
            raise error
        return Backend(checkpoint, config)

    engine = RolloutEngine(store.checkpoint(1), _backend_factory=factory, weight_store=store)
    with pytest.raises(RuntimeError, match="startup uncertain"):
        engine.load_weights(2)
    assert starts == [1, 2] and engine._pending_backend is pending
    with pytest.raises(RuntimeError, match="still unconfirmed"):
        engine.close()
    assert engine._pending_backend is pending and store.active_version == 1
    engine.close()
    assert engine._pending_backend is None
    with pytest.raises(RuntimeError, match="live owned"):
        engine.load_weights(2)


def test_old_worker_close_failure_never_starts_candidate(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    for version in (1, 2):
        sync.push([("weight", torch.ones(3, 2) * version)], version)

    class CloseFailure(Backend):
        failed = False

        def close(self):
            if not self.failed:
                self.failed = True
                raise RuntimeError("old worker ownership uncertain")
            super().close()

    engine = RolloutEngine(store.checkpoint(1), _backend_factory=CloseFailure, weight_store=store)
    with pytest.raises(RuntimeError, match="old worker"):
        engine.load_weights(2)
    assert Backend.live == 1 and engine.policy_version == 1
    engine.close()
    assert Backend.live == 0


def test_post_load_content_change_closes_candidate_before_rollback(tmp_path):
    store = make_store(tmp_path)
    sync = WeightSync(store)
    for version in (1, 2):
        sync.push([("weight", torch.ones(3, 2) * version)], version)
    starts = []

    def factory(checkpoint, config):
        starts.append(checkpoint.policy_version)
        worker = Backend(checkpoint, config)
        if checkpoint.policy_version == 2:
            file = Path(checkpoint.model_path) / "config.json"
            file.chmod(0o600)
            file.write_text("{}")
        return worker

    with RolloutEngine(store.checkpoint(1), _backend_factory=factory, weight_store=store) as engine:
        with pytest.raises(ValueError, match="content"):
            engine.load_weights(2)
        assert starts == [1, 2, 1] and engine.policy_version == 1 and Backend.live == 1
        torch.testing.assert_close(engine._backend.weight, torch.ones(3, 2))


def test_reopened_store_must_match_snapshot_contract(tmp_path):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    mismatched = VersionStore(store.root, {"weight": ((3, 2), torch.bfloat16)}, store.assets)
    with pytest.raises(ValueError, match="schema/config contract"):
        mismatched.checkpoint(1)


def test_stale_close_preserves_successor_reservation(tmp_path):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    checkpoint = store.checkpoint(1)
    first = RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store)
    first.close()
    with RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store):
        first.close()
        assert store.active_version == 1
        with pytest.raises(ValueError, match="active"):
            store.discard(1)
        with pytest.raises(RuntimeError, match="already has an active"):
            RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store)
    assert Backend.live == 0 and store.active_version is None


@pytest.mark.parametrize("confirmed", [False, True])
def test_initial_start_reserves_until_confirmed_cleanup(tmp_path, confirmed):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    checkpoint = store.checkpoint(1)
    calls = []

    class Partial:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("still unresolved")

    partial = Partial()

    def fail_start(checkpoint, config):
        calls.append(checkpoint.policy_version)
        # Reservation must already block a reentrant constructor inside the factory.
        with pytest.raises(RuntimeError, match="already has an active"):
            RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store)
        error = RuntimeError("initial start failed")
        error.worker, error.cleanup_confirmed = partial, confirmed
        raise error

    with pytest.raises(RuntimeError, match="initial start failed") as caught:
        RolloutEngine(checkpoint, _backend_factory=fail_start, weight_store=store)
    recovery = caught.value.recovery_engine
    if not confirmed:
        assert store.active_version == 1
        with pytest.raises(RuntimeError, match="already has an active"):
            RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store)
        with pytest.raises(RuntimeError, match="still unresolved"):
            recovery.close()
        assert store.active_version == 1
        recovery.close()
        assert partial.close_calls == 2
    else:
        assert partial.close_calls == 0
    assert store.active_version is None
    with RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store):
        recovery.close()  # stale failed owner cannot unpin its successor either
        assert store.active_version == 1
    assert calls == [1] and Backend.live == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_real_loader_rejects_owned_non_bf16_before_spawn(tmp_path, monkeypatch, dtype):
    from rl.rollout import _worker

    store = make_store(tmp_path, dtype)
    original = torch.ones(3, 2, dtype=dtype)
    WeightSync(store).push([("weight", original)], 1)
    checkpoint = store.checkpoint(1)
    spawned = []

    def forbidden_context(*args):
        spawned.append(args)
        raise AssertionError("must reject before worker construction")

    monkeypatch.setattr(_worker.mp, "get_context", forbidden_context)
    with pytest.raises(ValueError, match="BF16"):
        RolloutEngine(checkpoint, weight_store=store)
    assert spawned == [] and store.active_version is None
    with RolloutEngine(checkpoint, weight_store=store, _backend_factory=Backend) as engine:
        assert engine._backend.weight.dtype == dtype
        torch.testing.assert_close(engine._backend.weight, original)


@pytest.mark.parametrize("direct_stage", [False, True])
def test_final_directory_fsync_deadline_preserves_committed_identity(
    tmp_path, monkeypatch, direct_stage
):
    import stat

    from rl.sync import version as version_module

    store = make_store(tmp_path)
    sync = WeightSync(store, timeout=30)
    sync.push([("weight", torch.ones(3, 2))], 1)
    previous = store.checkpoint(1)
    store.active_version = 1
    now = [0.0]
    monkeypatch.setattr(version_module.time, "monotonic", lambda: now[0])
    original_fsync = version_module.os.fsync

    def slow_directory(fd):
        original_fsync(fd)
        if stat.S_ISDIR(version_module.os.fstat(fd).st_mode):
            now[0] = 60.0

    monkeypatch.setattr(version_module.os, "fsync", slow_directory)
    with pytest.raises(TimeoutError, match="transaction deadline"):
        if direct_stage:
            stage = store.begin(2, 30)
            stage.add("weight", torch.zeros(3, 2))
            stage.prepare()
            stage.commit()
        else:
            sync.push([("weight", torch.zeros(3, 2))], 2)
    assert store._in_doubt and (direct_stage or sync.failed)
    assert store.active_version == 1
    version_module.verify_checkpoint(previous)
    committed = store._entries()[2]
    version_module.verify_checkpoint(committed)
    with pytest.raises(RuntimeError, match="in doubt"):
        store.checkpoint(2)
    reopened = VersionStore(store.root, store.schema, store.assets)
    assert reopened.checkpoint(2) == committed
    reopened.discard(2)
    with pytest.raises(ValueError, match="increase"):
        WeightSync(reopened).push([("weight", torch.zeros(3, 2))], 2)


def test_initial_unknown_without_handle_keeps_failed_owner(tmp_path):
    store = make_store(tmp_path)
    WeightSync(store).push([("weight", torch.ones(3, 2))], 1)
    checkpoint = store.checkpoint(1)

    def unknown_start(*_args):
        raise RuntimeError("no trustworthy cleanup handle")

    with pytest.raises(RuntimeError, match="no trustworthy") as caught:
        RolloutEngine(checkpoint, _backend_factory=unknown_start, weight_store=store)
    recovery = caught.value.recovery_engine
    assert store._engine_owner is recovery
    with pytest.raises(RuntimeError, match="without a cleanup handle"):
        recovery.close()
    assert store.active_version == 1
    with pytest.raises(RuntimeError, match="already has an active"):
        RolloutEngine(checkpoint, _backend_factory=Backend, weight_store=store)
    assert Backend.live == 0  # this injected unknown factory never actually started a worker


def test_actual_loader_bf16_schema_validation_accepts_exact_snapshot(tmp_path):
    from rl.rollout._worker import _validate_owned_bf16

    store = make_store(tmp_path, torch.bfloat16)
    WeightSync(store).push([("weight", torch.ones(3, 2, dtype=torch.bfloat16))], 1)
    assert _validate_owned_bf16(store.checkpoint(1)) is True


def test_push_return_deadline_checked_after_commit(tmp_path, monkeypatch):
    from rl.sync import version as version_module

    store = make_store(tmp_path)
    sync = WeightSync(store, timeout=30)
    now = [0.0]
    monkeypatch.setattr(version_module.time, "monotonic", lambda: now[0])
    original_commit = version_module.Stage.commit

    def late_return(stage):
        original_commit(stage)
        now[0] = 60.0

    monkeypatch.setattr(version_module.Stage, "commit", late_return)
    with pytest.raises(TimeoutError, match="transaction deadline"):
        sync.push([("weight", torch.ones(3, 2))], 1)
    assert sync.failed and store._in_doubt
    version_module.verify_checkpoint(store._entries()[1])
