"""Real autograd/optimizer CPU coverage of the shared training loop."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from rl.learner.train import Learner, LearnerConfig, RolloutGroup, encode_group
from rl.rollout.sampler import Prompt, Sample, SamplingConfig


class Tokenizer:
    def encode(self, text):
        assert text == "exact prompt"
        return [1, 2]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def group(version=2, group_id="group", rewards=(1.0, 0.0)):
    return RolloutGroup(
        "batch",
        group_id,
        Prompt("p", "exact prompt"),
        (
            Sample("p", "not tokenizable as generated IDs", (3, 4, 0), (-1.0, -2.0, -1.0), version),
            Sample("p", "", (5,), (-1.5,), version),
        ),
        rewards,
        SamplingConfig(),
    )


def trainer(model, reference, microbatch):
    return Learner(
        model,
        reference,
        Tokenizer(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        LearnerConfig(group_size=2, microbatch_size=microbatch, max_length=16),
        policy_version=2,
    )


def test_nonuniform_token_accumulation_matches_whole_batch():
    torch.manual_seed(123)
    policy = TinyLM()
    ref = copy.deepcopy(policy)
    split = trainer(copy.deepcopy(policy), copy.deepcopy(ref), 1)
    whole = trainer(copy.deepcopy(policy), copy.deepcopy(ref), 2)
    before = copy.deepcopy(whole.policy.state_dict())
    split_metrics = split.step([group(), group(0, "stale")])
    whole_metrics = whole.step([group()])
    assert split_metrics["dropped_stale"] == 2
    assert split_metrics["effective_tokens"] == 4
    assert split_metrics["policy_version"] == 3
    assert split_metrics["loss"] == pytest.approx(whole_metrics["loss"], abs=1e-6)
    for key, tensor in whole.policy.state_dict().items():
        torch.testing.assert_close(tensor, split.policy.state_dict()[key], atol=1e-7, rtol=1e-6)
    assert any(
        not torch.equal(tensor, before[key]) for key, tensor in whole.policy.state_dict().items()
    )
    for key, tensor in whole.reference.state_dict().items():
        torch.testing.assert_close(tensor, ref.state_dict()[key])
    assert all(p.grad is None and not p.requires_grad for p in whole.reference.parameters())


def test_causal_token_alignment_includes_eos():
    records = encode_group(group(), Tokenizer(), 16)
    assert records[0].input_ids == (1, 2, 3, 4, 0)
    assert records[0].prompt_length == 2
    assert records[0].old_logprobs == (-1.0, -2.0, -1.0)
    with pytest.raises(ValueError, match="length"):
        encode_group(group(), Tokenizer(), 4)


def test_group_contract_and_empty_stale_step():
    model = TinyLM()
    learner = trainer(model, copy.deepcopy(model), 1)
    result = learner.step([group(0)])
    assert result["dropped_stale"] == 2
    assert result["policy_version"] == 2
    assert result["optimizer_steps"] == 0
    with pytest.raises(ValueError, match="future"):
        learner.step([group(3)])
    with pytest.raises(ValueError, match="duplicate"):
        learner.step([group(), group()])
    # Same prompt IDs in distinct batches are not merged.
    different_batch = RolloutGroup(
        "other", "group", group().prompt, group().samples, group().rewards, SamplingConfig()
    )
    assert learner.step([group(), different_batch])["accepted_samples"] == 4


@pytest.mark.parametrize(
    "change,match",
    [
        ({"samples": group().samples[:1], "rewards": (1.0,)}, "complete"),
        ({"samples": (group().samples[0], group(1).samples[1])}, "version"),
        ({"sampling": SamplingConfig(temperature=0.8)}, "neutral"),
        ({"rewards": (float("nan"), 0.0)}, "finite"),
    ],
)
def test_reject_invalid_groups(change, match):
    from dataclasses import replace

    bad = replace(group(), **change)
    policy = TinyLM()
    learner = trainer(policy, copy.deepcopy(policy), 1)
    with pytest.raises(ValueError, match=match):
        learner.step([bad])
    assert learner.policy_version == 2


def test_nonfinite_gradient_cancels_step_and_version():
    policy = TinyLM()
    learner = trainer(policy, copy.deepcopy(policy), 1)
    before = copy.deepcopy(policy.state_dict())
    hook = policy.head.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(FloatingPointError, match="gradient"):
        learner.step([group()])
    hook.remove()
    assert_failed_phase(learner)
    assert all(p.grad is None for p in policy.parameters())
    for key, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[key])


def test_independent_fixed_reference_and_optimizer_ownership():
    policy = TinyLM()
    with pytest.raises(ValueError, match="share"):
        trainer(policy, policy, 1)
    with pytest.raises(ValueError, match="exactly"):
        Learner(
            policy,
            copy.deepcopy(policy),
            Tokenizer(),
            torch.optim.SGD(policy.head.parameters(), lr=0.1),
            LearnerConfig(group_size=2),
        )


def assert_failed_phase(learner):
    assert learner.phase_status == "failed"
    assert learner.last_committed_version == 2
    assert learner.failure_reason
    for attribute in ("policy", "optimizer", "policy_version"):
        with pytest.raises(RuntimeError, match="phase"):
            getattr(learner, attribute)
    for batch in ([], [group()]):
        with pytest.raises(RuntimeError, match="phase"):
            learner.step(batch)


def test_partial_optimizer_failure_poisoned_phase():
    class MutateThenRaise(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            raise RuntimeError("after real SGD update")

    torch.manual_seed(123)
    policy = TinyLM()
    before = copy.deepcopy(policy.state_dict())
    learner = Learner(
        policy,
        copy.deepcopy(policy),
        Tokenizer(),
        MutateThenRaise(policy.parameters(), lr=0.1),
        LearnerConfig(group_size=2),
        policy_version=2,
    )
    with pytest.raises(RuntimeError, match="after real SGD update"):
        learner.step([group()])
    assert any(not torch.equal(value, before[key]) for key, value in policy.state_dict().items())
    assert_failed_phase(learner)


def test_native_sgd_finite_gradient_overflow_poisoned_phase():
    policy = TinyLM()
    learner = Learner(
        policy,
        copy.deepcopy(policy),
        Tokenizer(),
        torch.optim.SGD(policy.parameters(), lr=1e20),
        LearnerConfig(group_size=2),
        policy_version=2,
    )
    hook = policy.head.weight.register_hook(lambda grad: torch.full_like(grad, 1e20))
    try:
        with pytest.raises(FloatingPointError, match="policy"):
            learner.step([group()])
    finally:
        hook.remove()
    assert not torch.isfinite(policy.head.weight).all()
    assert_failed_phase(learner)


def test_nonfinite_optimizer_state_poisoned_phase():
    class CorruptState(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            for state in self.state.values():
                state["momentum_buffer"].fill_(float("inf"))

    policy = TinyLM()
    learner = Learner(
        policy,
        copy.deepcopy(policy),
        Tokenizer(),
        CorruptState(policy.parameters(), lr=0.1, momentum=0.9),
        LearnerConfig(group_size=2),
        policy_version=2,
    )
    with pytest.raises(FloatingPointError, match="optimizer"):
        learner.step([group()])
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    assert_failed_phase(learner)


@pytest.mark.parametrize("target", ["policy", "reference", "optimizer"])
def test_reject_nonfinite_initial_state(target):
    policy, reference = TinyLM(), TinyLM()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1, momentum=0.9)
    with torch.no_grad():
        if target == "policy":
            policy.head.weight.fill_(float("inf"))
        elif target == "reference":
            reference.head.weight.fill_(float("nan"))
        else:
            optimizer.state[policy.head.weight]["momentum_buffer"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match=target):
        Learner(policy, reference, Tokenizer(), optimizer, LearnerConfig(group_size=2))


def test_pre_execution_rejection_is_reusable():
    policy = TinyLM()
    learner = trainer(policy, copy.deepcopy(policy), 1)
    with pytest.raises(ValueError, match="future"):
        learner.step([group(3)])
    assert learner.phase_status == "ready"
    assert learner.policy_version == 2
    assert learner.step([group()])["policy_version"] == 3


def test_cleanup_failure_poisoned_phase_and_attempts_all_restores(monkeypatch):
    policy = TinyLM()
    learner = trainer(policy, copy.deepcopy(policy), 2)
    # Fault-inject the control callbacks only; this is not FSDP/CUDA evidence.
    learner.fsdp = True
    sync_calls, last_calls = [], []

    def sync(value):
        sync_calls.append(value)
        if len(sync_calls) == 2:
            raise RuntimeError("restore sync failed")

    def last(value):
        last_calls.append(value)

    monkeypatch.setattr(policy, "set_requires_gradient_sync", sync, raising=False)
    monkeypatch.setattr(policy, "set_is_last_backward", last, raising=False)
    with pytest.raises(RuntimeError, match="restore sync failed"):
        learner.step([group()])
    assert sync_calls == [True, True]
    assert last_calls == [True, True], "one failed restore must not skip other cleanup"
    assert_failed_phase(learner)


def test_execution_and_cleanup_errors_are_both_preserved():
    class DoubleFailure(torch.optim.SGD):
        clear_calls = 0

        def step(self, closure=None):
            super().step(closure)
            raise RuntimeError("update failed")

        def zero_grad(self, *args, **kwargs):
            self.clear_calls += 1
            super().zero_grad(*args, **kwargs)
            if self.clear_calls == 2:
                raise RuntimeError("cleanup failed")

    policy = TinyLM()
    learner = Learner(
        policy,
        copy.deepcopy(policy),
        Tokenizer(),
        DoubleFailure(policy.parameters(), lr=0.1),
        LearnerConfig(group_size=2),
        policy_version=2,
    )
    with pytest.raises(BaseExceptionGroup) as caught:
        learner.step([group()])
    assert [str(exc) for exc in caught.value.exceptions] == ["update failed", "cleanup failed"]
    assert_failed_phase(learner)


def test_executing_phase_cannot_be_reentered_or_published():
    policy = TinyLM()
    learner = trainer(policy, copy.deepcopy(policy), 1)
    optimizer = learner.optimizer

    def check_boundary(_optimizer, _args, _kwargs):
        assert learner.phase_status == "executing"
        assert learner.last_committed_version == 2
        for attribute in ("policy", "optimizer", "policy_version"):
            with pytest.raises(RuntimeError, match="phase"):
                getattr(learner, attribute)
        with pytest.raises(RuntimeError, match="phase"):
            learner.step([])

    hook = optimizer.register_step_pre_hook(check_boundary)
    try:
        assert learner.step([group()])["policy_version"] == 3
    finally:
        hook.remove()
    assert learner.phase_status == "ready"
    assert learner.policy is policy


def test_initial_gradient_cleanup_failure_poisoned_phase():
    class FailFirstClear(torch.optim.SGD):
        clear_calls = 0

        def zero_grad(self, *args, **kwargs):
            self.clear_calls += 1
            if self.clear_calls == 1:
                raise RuntimeError("initial clear failed")
            super().zero_grad(*args, **kwargs)

    policy = TinyLM()
    optimizer = FailFirstClear(policy.parameters(), lr=0.1)
    learner = Learner(
        policy,
        copy.deepcopy(policy),
        Tokenizer(),
        optimizer,
        LearnerConfig(group_size=2),
        policy_version=2,
    )
    with pytest.raises(RuntimeError, match="initial clear failed"):
        learner.step([group()])
    assert optimizer.clear_calls == 2
    assert_failed_phase(learner)


def test_pre_step_state_validation_also_covers_empty_batches():
    policy = TinyLM()
    learner = trainer(policy, copy.deepcopy(policy), 1)
    learner.optimizer.param_groups[0]["lr"] = float("inf")
    with pytest.raises(FloatingPointError, match="optimizer options"):
        learner.step([])
    assert_failed_phase(learner)


def test_unsupported_optimizer_state_is_not_silently_accepted():
    policy = TinyLM()
    with pytest.raises(TypeError, match="SGD/AdamW"):
        Learner(
            policy,
            copy.deepcopy(policy),
            Tokenizer(),
            torch.optim.RMSprop(policy.parameters()),
            LearnerConfig(group_size=2),
        )
