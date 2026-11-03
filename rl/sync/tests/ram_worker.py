"""Trusted tiny CPU workers; no vLLM import or actual-GPU evidence."""

import contextlib
import copy
import json
import os
import random
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import torch

from rl.learner import handoff_stream as stream
from rl.learner import state_handoff as state
from rl.learner.handoff_protocol import retire_input_and_receive_output
from rl.learner.tests.handoff_worker import BINDING, Tokenizer, group, summary
from rl.learner.train import Learner, LearnerConfig
from rl.rollout._rpc import FramedSocket
from rl.rollout._worker import VLLMWorker
from rl.rollout.ram_loader import iter_weights, target_views, verify_model
from rl.sync.ram_weights import describe_source, read_weights

CAP = 131072
META = 65536


class Attention(torch.nn.Module):
    def __init__(self, packed=False):
        super().__init__()
        self.q_size = self.kv_size = 4
        self.packed = packed
        if packed:
            self.qkv_proj = torch.nn.Linear(4, 12)
        else:
            for name in ("q", "k", "v"):
                setattr(self, name + "_proj", torch.nn.Linear(4, 4))
        self.o_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        if self.packed:
            q, k, v = self.qkv_proj(x).split(4, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return self.o_proj(torch.tanh(q + k + v))


class MLP(torch.nn.Module):
    def __init__(self, packed=False):
        super().__init__()
        self.packed = packed
        if packed:
            self.gate_up_proj = torch.nn.Linear(4, 12, bias=False)
        else:
            self.gate_proj = torch.nn.Linear(4, 6, bias=False)
            self.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 4, bias=False)

    def forward(self, x):
        if self.packed:
            gate, up = self.gate_up_proj(x).split(6, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(torch.sigmoid(gate) * up)


class Layer(torch.nn.Module):
    def __init__(self, packed):
        super().__init__()
        self.self_attn = Attention(packed)
        self.mlp = MLP(packed)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class TinyQwen(torch.nn.Module):
    def __init__(self, packed=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True, intermediate_size=6, vocab_size=8)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.model.layers = torch.nn.ModuleList([Layer(packed)])
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.register_buffer("offset", torch.arange(8, dtype=torch.float32) / 100)
        self.register_buffer("gain", torch.tensor(0.8), persistent=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(x) * self.gain + self.offset)


def make_learner(version=0):
    torch.random.default_generator.manual_seed(914)
    random.seed(2026)
    policy = TinyQwen()
    reference = copy.deepcopy(policy).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.02, betas=(0.8, 0.95), foreach=False)
    return Learner(
        policy,
        reference,
        Tokenizer(),
        optimizer,
        LearnerConfig(group_size=2, max_length=16, max_staleness=1),
        policy_version=version,
    )


def serial_budget():
    return stream.SerialBudget(
        CAP, 1_000_000_000, CAP, CAP, metadata_cap=META, ram_limit=2_000_000_000
    )


def contract():
    learner = make_learner()
    learner.step([group(0)])
    payload = state._payload(learner, BINDING)[0]
    encoded, _ = stream._encode(payload, serial_budget())
    manifest = stream._json(encoded, META)
    saved = stream._decode(manifest["state"], manifest["records"], META)
    return describe_source(saved)


def command(mode):
    return [sys.executable, "-B", "-m", "rl.sync.tests.ram_worker", mode]


@contextlib.contextmanager
def source_artifact():
    learner = make_learner()
    learner.step([group(0)])
    fd = os.memfd_create("unit-training", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        stream.capture(learner, fd, binding=BINDING, budget=serial_budget())
        del learner
        info = os.fstat(fd)
        receipt = state.Receipt(
            info.st_dev,
            info.st_ino,
            info.st_size,
            state._digest(fd, info.st_size),
            state.SEALS,
            1,
            state._binding(BINDING),
            CAP,
            META,
        )
        yield fd, receipt
    finally:
        os.close(fd)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("tiny worker must be CPU-only")
    mode = sys.argv[1]
    budget = serial_budget()
    if mode == "initial":
        learner = make_learner()
        metrics = learner.step([group(0)])
        fd = int(os.environ["CRUCIBLE_HANDOFF_FD"])
    else:
        receipt = state.Receipt(**json.loads(os.environ["CRUCIBLE_HANDOFF_SOURCE_RECEIPT"]))
        source = int(os.environ["CRUCIBLE_HANDOFF_SOURCE_FD"])
        learner = make_learner(receipt.version)
        learner.restore_handoff_stream(
            source, receipt, expected_version=receipt.version, binding=BINDING, budget=budget
        )
        fd = retire_input_and_receive_output(
            source,
            receipt,
            int(os.environ["CRUCIBLE_STREAM_CONTROL_FD"]),
            bytes.fromhex(os.environ["CRUCIBLE_STREAM_NONCE"]),
        )
        if mode in ("advance", "advance-stale"):
            metrics = learner.step([group(receipt.version - int(mode == "advance-stale"))])
        elif mode == "advance-mixed":
            if receipt.version not in (2, 3):
                raise ValueError("mixed fixture expects source version2 or3")
            metrics = learner.step([group(2), replace(group(0), group_id="stale")])
        elif mode != "noop":
            raise ValueError("unknown worker mode")
        else:
            metrics = learner.step([])
    stream.capture(learner, fd, binding=BINDING, budget=budget)
    print(
        json.dumps(
            {
                "state": summary(learner),
                "loss": metrics.get("loss"),
                "metrics": metrics,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )


def cpu_serve(sock, checkpoint, config, diagnostic, fd, runtime_root):
    connection = FramedSocket(sock)
    try:
        checkpoint.receipt.validate(fd)
        model = TinyQwen(packed=True).to(torch.bfloat16)
        records, _ = read_weights(fd, checkpoint)
        views, _ = target_views(model, records)
        with torch.no_grad():
            for name, tensor in iter_weights(fd, checkpoint):
                views[name].copy_(tensor)
        ack = verify_model(model, fd, checkpoint)
        connection.send(["ready", [ack, {}]], time.monotonic() + 10)
        while True:
            request = connection.receive(time.monotonic() + 20)
            if request == ["close"]:
                break
            if request[0] != "generate":
                raise ValueError("CPU receiver request")
            with torch.no_grad():
                logp = torch.log_softmax(model(torch.tensor([[1, 2]])).logits[0, -1].float(), -1)
            outputs = [
                {
                    "outputs": [
                        {
                            "index": i,
                            "text": "",
                            "token_ids": [3],
                            "logprobs": [{"3": float(logp[3])}],
                        }
                        for i in range(request[2])
                    ]
                }
                for _ in request[1]
            ]
            connection.send(["outputs", outputs], time.monotonic() + 10)
    finally:
        os.close(fd)
        connection.close()


class CPUWorker(VLLMWorker):
    STARTUP_SECONDS = 15
    GRACE_SECONDS = 1
    TERM_SECONDS = 0.2
    KILL_SECONDS = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _serve_target=cpu_serve, **kwargs)


if __name__ == "__main__":
    main()
