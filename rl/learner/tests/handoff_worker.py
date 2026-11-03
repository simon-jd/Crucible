"""Explicit trusted CPU subprocess modes for the RAM handoff tests only."""

import copy
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from rl.learner.state_handoff import (
    CaptureBudget,
    Receipt,
    _payload,
    capture_ready_learner,
)
from rl.learner.train import Learner, LearnerConfig, RolloutGroup
from rl.rollout.sampler import Prompt, Sample, SamplingConfig

BINDING = {
    "model": "tiny-cpu-fixture-v1",
    "tokenizer": "tiny-tokenizer-v1",
    "external_state": "fixed-lr:no-scheduler:no-scaler:no-loader:no-numpy",
}
CAP = 131072


class Tokenizer:
    def encode(self, text):
        assert text == "prompt"
        return [1, 2]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)
        self.head.weight = self.embedding.weight
        self.register_buffer("offset", torch.arange(8, dtype=torch.float32) * 0.01)
        self.register_buffer("gain", torch.tensor(0.8), persistent=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(
            logits=self.head(self.embedding(input_ids)) * self.gain + self.offset
        )


def make_learner(version=0, optimizer="AdamW", distributed=False):
    # Seed CPU directly: no CUDA initialization or hardware query is needed.
    torch.random.default_generator.manual_seed(914)
    random.seed(2026)
    policy = TinyLM()
    reference = copy.deepcopy(policy)
    if distributed:
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import Replicate, distribute_tensor

        mesh = DeviceMesh("cpu", [0])
        weight = torch.nn.Parameter(
            distribute_tensor(policy.embedding.weight.detach(), mesh, [Replicate()])
        )
        bias = torch.nn.Parameter(distribute_tensor(policy.head.bias.detach(), mesh, [Replicate()]))
        policy.embedding.weight = policy.head.weight = weight
        policy.head.bias = bias
    if optimizer == "AdamW":
        opt = torch.optim.AdamW(
            policy.parameters(), lr=0.02, betas=(0.8, 0.95), amsgrad=True, foreach=False
        )
    else:
        opt = torch.optim.SGD(policy.parameters(), lr=0.03, momentum=0.9, foreach=False)
    return Learner(
        policy,
        reference,
        Tokenizer(),
        opt,
        LearnerConfig(group_size=2, max_length=16),
        policy_version=version,
    )


def group(version):
    return RolloutGroup(
        "batch",
        "group",
        Prompt("p", "prompt"),
        (
            Sample("p", "", (3, 4, 0), (-2.0, -2.0, -2.0), version),
            Sample("p", "", (5,), (-2.0,), version),
        ),
        (1.0, 0.0),
        SamplingConfig(),
    )


def compact(value):
    if type(value) is torch.Tensor:
        data = value.detach().cpu().reshape(-1).view(torch.uint8).tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(bytes(data)).hexdigest(),
            "absolute_sum": float(value.abs().sum()),
            "scalar": value.item() if value.numel() == 1 else None,
        }
    if type(value) is dict:
        return {key: compact(child) for key, child in value.items()}
    if type(value) in (list, tuple):
        return [compact(child) for child in value]
    return value


def summary(learner):
    payload = _payload(learner, BINDING)[0]
    # Keep Python RNG comparison exact but bounded in subprocess output.
    payload["rng"]["python"] = hashlib.sha256(repr(payload["rng"]["python"]).encode()).hexdigest()
    return compact(payload)


def dtensor_step(learner):
    # Real CPU DTensor optimizer/autograd, deliberately no FSDP or LM claims.
    learner.optimizer.zero_grad(set_to_none=True)
    loss = sum((parameter * parameter).sum() for parameter in learner.policy.parameters())
    loss.backward()
    learner.optimizer.step()
    learner.optimizer.zero_grad(set_to_none=True)
    return float(loss.to_local())


def stream_main(mode):
    from rl.learner.handoff_protocol import retire_input_and_receive_output
    from rl.learner.handoff_stream import SerialBudget, capture

    budget = SerialBudget(**json.loads(os.environ["CRUCIBLE_STREAM_BUDGET"]))
    if mode == "stream-capture":
        fd = int(os.environ["CRUCIBLE_HANDOFF_FD"])
        learner = make_learner()
        learner.step([group(0)])
        result = capture(learner, fd, binding=BINDING, budget=budget)
        print(json.dumps({"pid": os.getpid(), "capture": result}), flush=True)
        return
    receipt = Receipt(**json.loads(os.environ["CRUCIBLE_HANDOFF_SOURCE_RECEIPT"]))
    fd = int(os.environ["CRUCIBLE_HANDOFF_SOURCE_FD"])
    learner = make_learner(receipt.version)
    learner.restore_handoff_stream(
        fd, receipt, expected_version=receipt.version, binding=BINDING, budget=budget
    )
    restored = summary(learner)
    if mode == "stream-inspect":
        print(json.dumps({"pid": os.getpid(), "state": restored}), flush=True)
        return
    if mode != "stream-advance":
        raise ValueError("unknown stream worker mode")
    assert "CRUCIBLE_HANDOFF_FD" not in os.environ, "output capability must not be inherited"
    output = retire_input_and_receive_output(
        fd,
        receipt,
        int(os.environ["CRUCIBLE_STREAM_CONTROL_FD"]),
        bytes.fromhex(os.environ["CRUCIBLE_STREAM_NONCE"]),
    )
    try:
        loss = learner.step([group(receipt.version)])["loss"]
        result = capture(learner, output, binding=BINDING, budget=budget)
        print(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "restored": restored,
                    "next_loss": loss,
                    "state": summary(learner),
                    "capture": result,
                }
            ),
            flush=True,
        )
    finally:
        os.close(output)


def main():
    mode = sys.argv[1]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("worker requires CPU-only uninitialized CUDA")
    if mode.startswith("stream-"):
        stream_main(mode)
        return
    fd = int(os.environ["CRUCIBLE_HANDOFF_FD"])
    distributed = mode.startswith("dtensor-")
    if distributed:
        rendezvous = Path(tempfile.gettempdir()) / f"gloo-{os.getpid()}"
        torch.distributed.init_process_group(
            "gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1
        )
    try:
        if mode in ("capture", "dtensor-capture"):
            learner = make_learner(distributed=distributed)
            before = summary(learner)
            if distributed:
                loss = dtensor_step(learner)
            else:
                loss = learner.step([group(0)])["loss"]
            assert any(
                before["policy"]["tensors"][name] != value
                for name, value in summary(learner)["policy"]["tensors"].items()
            ), "real optimizer update required"
            declared = int(os.environ["CRUCIBLE_HANDOFF_VERSION"])
            assert learner.policy_version == declared
            result = capture_ready_learner(
                learner,
                fd,
                binding=json.loads(os.environ["CRUCIBLE_HANDOFF_BINDING"]),
                budget=CaptureBudget(**json.loads(os.environ["CRUCIBLE_HANDOFF_BUDGET"])),
            )
            print(json.dumps({"pid": os.getpid(), "loss": loss, "capture": result}), flush=True)
        elif mode == "advance":
            source_fd = int(os.environ["CRUCIBLE_HANDOFF_SOURCE_FD"])
            receipt = Receipt(**json.loads(os.environ["CRUCIBLE_HANDOFF_SOURCE_RECEIPT"]))
            learner = make_learner(receipt.version)
            learner.restore_handoff(
                source_fd, receipt, expected_version=receipt.version, binding=BINDING
            )
            restored = summary(learner)
            loss = learner.step([group(receipt.version)])["loss"]
            assert learner.policy_version == int(os.environ["CRUCIBLE_HANDOFF_VERSION"])
            capture = capture_ready_learner(
                learner,
                fd,
                binding=BINDING,
                budget=CaptureBudget(**json.loads(os.environ["CRUCIBLE_HANDOFF_BUDGET"])),
                source_fd=source_fd,
                source_receipt=receipt,
            )
            print(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "restored": restored,
                        "next_loss": loss,
                        "state": summary(learner),
                        "capture": capture,
                    }
                ),
                flush=True,
            )
        elif mode in ("resume", "inspect", "dtensor-resume"):
            receipt = Receipt(**json.loads(os.environ["CRUCIBLE_HANDOFF_RECEIPT"]))
            learner = make_learner(receipt.version, distributed=distributed)
            learner.restore_handoff(fd, receipt, expected_version=receipt.version, binding=BINDING)
            restored = summary(learner)
            if mode == "inspect":
                print(json.dumps({"pid": os.getpid(), "state": restored}), flush=True)
                return
            if distributed:
                loss = dtensor_step(learner)
            else:
                loss = learner.step([group(1)])["loss"]
            final = summary(learner)
            if distributed:
                uninterrupted = make_learner(distributed=True)
                dtensor_step(uninterrupted)
                assert summary(uninterrupted) == restored
                assert dtensor_step(uninterrupted) == loss
                assert summary(uninterrupted) == final
            print(
                json.dumps(
                    {"pid": os.getpid(), "restored": restored, "next_loss": loss, "state": final}
                ),
                flush=True,
            )
        else:
            raise ValueError("unknown worker mode")
        assert not torch.cuda.is_initialized(), "CPU test must not initialize CUDA"
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
