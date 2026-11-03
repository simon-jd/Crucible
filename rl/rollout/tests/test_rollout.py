"""CPU contract tests; real vLLM tests opt in under the shared GPU guard."""

from types import SimpleNamespace as NS

import pytest

from rl.rollout.engine import PolicyCheckpoint, RolloutEngine
from rl.rollout.sampler import Prompt, SamplingConfig, samples_from_outputs


class FakeBackend:
    def __init__(self, checkpoint, config):
        self.checkpoint, self.config, self.closed = checkpoint, config, False

    def generate(self, prompts, group_size):
        return [
            NS(
                outputs=[
                    NS(
                        index=j,
                        text=f"{p.text}:{self.config.seed}:{j}",
                        token_ids=[11, 22],
                        logprobs=[
                            {33: NS(logprob=-9.0), 11: NS(logprob=-0.5)},
                            {22: NS(logprob=-1.5)},
                        ],
                    )
                    for j in range(group_size)
                ]
            )
            for p in prompts
        ]

    def close(self):
        self.closed = True


@pytest.fixture
def engine():
    with RolloutEngine(
        PolicyCheckpoint("/fixture/checkpoint", "fixture-revision", 7),
        SamplingConfig(seed=123),
        _backend_factory=FakeBackend,
    ) as result:
        yield result


def test_group_size_respected(engine):
    samples = engine.generate([Prompt("a", "Hello"), Prompt("b", "World")], 3, 7)
    assert [s.prompt_id for s in samples] == ["a"] * 3 + ["b"] * 3


def test_policy_version_tagged(engine):
    assert {s.policy_version for s in engine.generate([Prompt("a", "Hello")], 2, 7)} == {7}


def test_policy_version_mismatch_rejected(engine):
    with pytest.raises(ValueError, match="loaded policy"):
        engine.generate([Prompt("a", "Hello")], 2, 8)


def test_seeded_sampling_reproducible(engine):
    assert engine.config.temperature > 0
    assert engine.generate([Prompt("a", "Hello")], 3, 7) == engine.generate(
        [Prompt("a", "Hello")], 3, 7
    )


def test_chosen_token_logprobs_order(engine):
    sample = engine.generate([Prompt("a", "Hello")], 1, 7)[0]
    assert sample.token_ids == (11, 22)
    assert sample.logprobs == (-0.5, -1.5)


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [NS(outputs=[])],
        [NS(outputs=[NS(index=0, text="", token_ids=[11], logprobs=[])])],
        [NS(outputs=[NS(index=0, text="", token_ids=[11], logprobs=[{22: NS(logprob=-1.0)}])])],
    ],
)
def test_malformed_outputs_rejected(outputs):
    with pytest.raises(ValueError):
        samples_from_outputs([Prompt("a", "Hello")], outputs, 1, 7)


@pytest.mark.parametrize("group", [0, -1, True, 1.5])
def test_invalid_group_size(engine, group):
    with pytest.raises(ValueError):
        engine.generate([Prompt("a", "Hello")], group, 7)


def test_close_and_empty_batch(engine):
    assert engine.generate([], 2, 7) == []
    backend = engine._backend
    engine.close()
    assert backend.closed
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([Prompt("a", "Hello")], 1, 7)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        SamplingConfig(seed=seed)


def test_failed_generation_closes_engine(engine):
    def fail(*_args):
        raise RuntimeError("backend failed")

    engine._backend.generate = fail
    with pytest.raises(RuntimeError, match="backend failed"):
        engine.generate([Prompt("a", "Hello")], 1, 7)
    assert engine._backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([Prompt("a", "Hello")], 1, 7)


def test_policy_identity_readonly(engine):
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        engine.checkpoint.policy_version = 8
    with pytest.raises(AttributeError):
        engine.checkpoint = PolicyCheckpoint("/different", "different", 8)


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": 0},
        {"max_tokens": 0},
        {"max_tokens": 256},
        {"max_model_len": True},
    ],
)
def test_invalid_sampling_settings(values):
    with pytest.raises(ValueError):
        SamplingConfig(**values)


def test_eos_included_and_completion_order():
    outputs = [
        NS(
            outputs=[
                NS(index=1, text="", token_ids=[151645], logprobs=[{151645: NS(logprob=-0.25)}]),
                NS(
                    index=0,
                    text="answer",
                    token_ids=[17, 151645],
                    logprobs=[{17: NS(logprob=-1.0)}, {151645: NS(logprob=-0.5)}],
                ),
            ]
        )
    ]
    samples = samples_from_outputs([Prompt("a", "Hello")], outputs, 2, 7)
    assert samples[0].token_ids == (17, 151645)
    assert samples[0].logprobs == (-1.0, -0.5)
    assert samples[1].token_ids == (151645,)
    assert samples[1].text == ""
