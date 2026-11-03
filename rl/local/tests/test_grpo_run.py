import pytest

from rl.local import grpo_run as gr


def test_sync_queue_trains_each_batch_immediately():
    q = gr.StaleQueue(0)
    assert q.push("b0") == "b0" and q.push("b1") == "b1"


def test_async_k1_trains_the_previous_batch_one_version_behind():
    q = gr.StaleQueue(1)
    assert q.push("b0") is None
    assert q.push("b1") == "b0"
    assert q.push("b2") == "b1"
    assert q.pending() == 1


def test_math_flags():
    assert gr.math_flags("so \\boxed{3}", truncated=False) == []
    assert gr.math_flags("\\boxed{3} or \\boxed{4}", truncated=False) == ["multiple_boxed_answers"]
    assert gr.math_flags("\\boxed{3} so \\boxed{3}", truncated=False) == []
    assert gr.math_flags("no answer", truncated=True) == ["no_boxed_answer", "truncated"]


def test_idle_share_is_time_outside_gpu_phases():
    assert gr.idle_share({"rollout_s": 4, "learn_s": 3, "sync_s": 1}, wall_s=10) == pytest.approx(0.2)


def test_eval_summary_uses_unbiased_pass_at_1_per_problem():
    outcomes = {"p1": [True, False, False, False], "p2": [True, True, True, True]}
    s = gr.eval_summary(outcomes)
    assert s["problems"] == 2 and s["samples_per_problem"] == 4
    assert s["pass_at_1"] == pytest.approx((0.25 + 1.0) / 2)
    assert s["stderr"] > 0


def test_prompt_asks_for_boxed_answer():
    messages = gr.prompt_messages("What is 2+2?")
    assert messages[-1]["role"] == "user" and "\\boxed{}" in messages[-1]["content"]


def test_reward_none_grades_correctness_only():
    assert gr.shaped_reward(True, "length", "none") == 1.0
    assert gr.shaped_reward(False, "stop", "none") == 0.0


def test_reward_eos_zeroes_truncated_completions():
    # The sync k=0 exploit: a right last box followed by run-on text that never stops.
    assert gr.shaped_reward(True, "length", "eos") == 0.0
    assert gr.shaped_reward(True, "stop", "eos") == 1.0
    assert gr.shaped_reward(False, "stop", "eos") == 0.0


def test_reward_mode_rejects_unknown():
    with pytest.raises(ValueError):
        gr.shaped_reward(True, "stop", "length")
