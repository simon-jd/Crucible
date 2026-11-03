import pytest

from rl.local import grpo_run as gr
from rl.verl import reward as rw

EOS = {151645}


def test_is_truncated_follows_vllm_finish_reason():
    assert rw.is_truncated([1, 2, 3], 3, EOS)  # hit the limit without EOS -> "length"
    assert not rw.is_truncated([1, 2, 151645], 3, EOS)  # EOS as the last allowed token -> "stop"
    assert not rw.is_truncated([1, 151645], 3, EOS)  # stopped early
    assert not rw.is_truncated([], 3, EOS)


def test_is_truncated_ignores_other_special_tokens():
    # generation_config="vllm": <|endoftext|> (151643) does not stop Qwen2.5-Instruct.
    assert rw.is_truncated([1, 2, 151643], 3, EOS)


@pytest.mark.parametrize(
    ("text", "truncated", "score"),
    [
        ("so \\boxed{4}", False, 1.0),
        ("so \\boxed{4} and then on and on", True, 0.0),
        ("so \\boxed{5}", False, 0.0),
        ("so \\boxed{5}", True, 0.0),
    ],
)
def test_compute_score_is_grpo_run_eos_reward(text, truncated, score):
    out = rw.compute_score("gsm8k", text, "4", extra_info={"truncated": truncated})
    assert out["score"] == score
    assert out["acc"] == float("\\boxed{4}" in text)
    assert out["truncated"] == float(truncated)
    assert out["score"] == gr.shaped_reward(bool(out["acc"]), "length" if truncated else "stop", "eos")


def test_compute_score_uses_the_project_verifier(monkeypatch):
    calls = []

    class V:
        correct, reason = True, "stub"

    monkeypatch.setattr(rw, "verify_math", lambda p, r: calls.append((p, r)) or V)
    assert rw.compute_score("gsm8k", "text", "ref", extra_info={"truncated": False})["score"] == 1.0
    assert calls == [("text", "ref")]


def test_compute_score_requires_the_truncation_flag():
    with pytest.raises(ValueError):
        rw.compute_score("gsm8k", "\\boxed{4}", "4", extra_info={})
    with pytest.raises(ValueError):
        rw.compute_score("gsm8k", "\\boxed{4}", "4")
