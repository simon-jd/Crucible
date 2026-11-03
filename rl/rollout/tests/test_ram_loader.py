"""Shared loader verifier on real CPU packed tensors, never actual vLLM."""

import contextlib
import hashlib
import multiprocessing as mp
import os
import time
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from rl.rollout._rpc import FramedSocket
from rl.rollout._worker import VLLMWorker, WorkerStartupError
from rl.rollout.ram_loader import SpawnFD, iter_weights, target_views, verify_model
from rl.rollout.sampler import SamplingConfig
from rl.sync.ram_weights import digest, read_weights, verify_qwen_assets
from rl.sync.tests.ram_worker import TinyQwen, contract, source_artifact
from rl.sync.tests.test_ram_weights import budget, checkpoint_without_contract, export


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    name = "loader-" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()
    return tmp_path_factory.mktemp(name, numbered=False)


@contextlib.contextmanager
def weights():
    with source_artifact() as (source, receipt):
        fd = os.memfd_create("loader-weights", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            expected = contract()
            export(source, receipt, fd, budget(), digest(expected))
            checkpoint = checkpoint_without_contract(fd, receipt, expected)
            yield fd, checkpoint
        finally:
            os.close(fd)


def loaded_model(fd, checkpoint):
    model = TinyQwen(packed=True).to(torch.bfloat16)
    records, _ = read_weights(fd, checkpoint)
    views, _ = target_views(model, records)
    with torch.no_grad():
        for name, tensor in iter_weights(fd, checkpoint):
            views[name].copy_(tensor)
    return model


@pytest.mark.parametrize(
    "fault",
    ["none", "q_bias", "k_weight", "v_weight", "gate", "up", "buffer", "tie", "dtype", "missing"],
)
def test_all_coordinate_packed_verification(fault):
    with weights() as (fd, checkpoint):
        model = loaded_model(fd, checkpoint)
        layer = model.model.layers[0]
        with torch.no_grad():
            if fault == "q_bias":
                layer.self_attn.qkv_proj.bias[0] += 1
            elif fault == "k_weight":
                layer.self_attn.qkv_proj.weight[4, 0] += 1
            elif fault == "v_weight":
                layer.self_attn.qkv_proj.weight[8, 0] += 1
            elif fault in ("gate", "up"):
                layer.mlp.gate_up_proj.weight[0 if fault == "gate" else 6, 0] += 1
            elif fault == "buffer":
                model.offset[0] += 1
            elif fault == "tie":
                model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
            elif fault == "dtype":
                model.offset = model.offset.float()
            elif fault == "missing":
                model.register_buffer("extra", torch.ones(1))
        if fault == "none":
            assert verify_model(model, fd, checkpoint) == checkpoint.ack()
        else:
            with pytest.raises(ValueError):
                verify_model(model, fd, checkpoint)


def test_missing_packed_input_shard_rejected():
    with weights() as (fd, checkpoint):
        model = loaded_model(fd, checkpoint)
        records, _ = read_weights(fd, checkpoint)
        records = [r for r in records if ".k_proj.bias" not in r["name"]]
        with pytest.raises(ValueError, match="missing qkv"):
            target_views(model, records)


@pytest.mark.parametrize(
    "field", ["base_revision", "model_path", "assets", "source_sha256", "contract"]
)
def test_sealed_checkpoint_relabel_rejected(field):
    with weights() as (fd, checkpoint):
        value = (("config.json", "0" * 64),) if field == "assets" else "/changed"
        altered = replace(checkpoint, **{field: value})
        with pytest.raises(ValueError):
            read_weights(fd, altered)


def test_spawn_reduction_outside_start_forbids_broker(monkeypatch):
    with weights() as (fd, checkpoint):
        import multiprocessing.reduction

        monkeypatch.setattr(
            multiprocessing.reduction, "DupFd", lambda *_: pytest.fail("broker duplication")
        )
        capability = SpawnFD(fd, checkpoint.receipt)
        with pytest.raises(RuntimeError, match="spawn"):
            capability.__reduce__()


def test_actual_loader_rejects_incomplete_tokenizer_pins():
    with weights() as (_, checkpoint):
        revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        path = (
            "/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
            + revision
        )
        incomplete = replace(
            checkpoint, model_path=path, base_revision=revision, assets=(("config.json", "0" * 64),)
        )
        with pytest.raises(ValueError, match="complete Qwen"):
            verify_qwen_assets(incomplete)


def test_ambiguous_start_without_pid_retains_original_handle(monkeypatch, tmp_path):
    with weights() as (fd, checkpoint):

        def fail_start(self):
            raise OSError("ambiguous start")

        monkeypatch.setattr(mp.context.SpawnProcess, "start", fail_start)
        with pytest.raises(WorkerStartupError) as caught:
            VLLMWorker(checkpoint, SamplingConfig(), ram_fd=fd, runtime_root=tmp_path)
        worker = caught.value.worker
        assert not caught.value.cleanup_confirmed and worker._process.pid is None
        assert worker._ram_capability.fd == fd
        with pytest.raises(RuntimeError, match="ambiguous"):
            worker.close()
        checkpoint.receipt.validate(fd)


def test_real_spawn_serialization_failure_retains_capability(monkeypatch, tmp_path):
    with weights() as (fd, checkpoint):

        def interrupted(self):
            self.phase = "serialization-uncertain"
            raise KeyboardInterrupt("serialization interrupted")

        monkeypatch.setattr(SpawnFD, "__reduce__", interrupted)
        with pytest.raises(WorkerStartupError) as caught:
            VLLMWorker(checkpoint, SamplingConfig(), ram_fd=fd, runtime_root=tmp_path)
        worker = caught.value.worker
        assert not caught.value.cleanup_confirmed
        assert worker._ram_capability.phase == "serialization-uncertain"
        with pytest.raises(RuntimeError):
            worker.close()
        checkpoint.receipt.validate(fd)


def bad_ack(sock, checkpoint, config, diagnostic, fd, runtime):
    channel = FramedSocket(sock)
    try:
        checkpoint.receipt.validate(fd)
        if str(runtime).endswith("late"):
            time.sleep(5)
        elif str(runtime).endswith("partial"):
            channel.socket.send(b"\0")
            time.sleep(5)
        else:
            ack = checkpoint.ack()
            ack["weights"] = "0" * 64
            channel.send(["ready", [ack, {}]], time.monotonic() + 1)
    finally:
        os.close(fd)
        channel.close()


@pytest.mark.parametrize("kind", ["wrong", "partial", "late"])
def test_bad_ack_reaps_original_spawn(kind, tmp_path):
    class ShortWorker(VLLMWorker):
        STARTUP_SECONDS = 3
        GRACE_SECONDS = 0.1
        TERM_SECONDS = 0.2
        KILL_SECONDS = 0.2

    with weights() as (fd, checkpoint):
        with pytest.raises(WorkerStartupError) as caught:
            ShortWorker(
                checkpoint,
                SamplingConfig(),
                ram_fd=fd,
                runtime_root=tmp_path / kind,
                _serve_target=bad_ack,
            )
        worker = caught.value.worker
        assert worker._process.exitcode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(worker._process.pid, os.WNOHANG)
        checkpoint.receipt.validate(fd)
