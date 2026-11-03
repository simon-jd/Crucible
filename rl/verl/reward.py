"""T11 reward for the verl comparison: grpo_run's --reward eos, computed inside verl.

Score = rl.local.grpo_run.shaped_reward(correct, finish_reason, "eos"), where
correct = rl.verifiers.math.verify_math(text, answer).correct, exactly as
grpo_run.grade does it (same verifier, same 3 s worker timeout).

Truncation. grpo_run reads vLLM's finish_reason ("length" -> reward 0). verl's
agent loop drops finish_reason before the reward (vllm_async_server maps both
"stop" and "length" to stop_reason "completed"), so the reward manager
(rl.verl.reward_manager.T11RewardManager) recomputes it from the response
token ids, which verl keeps verbatim from vLLM (EOS included):

    truncated  <=>  len(ids) >= max_response_tokens  and  ids[-1] not in eos_ids

This is vLLM's rule: a sequence whose last allowed token is EOS finishes with
"stop", not "length". There are no stop strings in either run, so EOS and the
token limit are the only ways a completion ends. The manager passes the result
in extra_info["truncated"]; compute_score refuses to guess when it is missing.

No verl import here: Ray workers import this with PYTHONPATH=<repo root>, and
the CPU tests import it from .venv.
"""

from rl.local.grpo_run import shaped_reward
from rl.verifiers.math import verify_math

REWARD_MODE = "eos"


def is_truncated(response_ids, max_response_tokens, eos_ids):
    ids = list(response_ids)
    return len(ids) >= max_response_tokens and (not ids or ids[-1] not in eos_ids)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not extra_info or "truncated" not in extra_info:
        raise ValueError("extra_info['truncated'] missing: use reward.reward_manager T11RewardManager (rl.verl.reward_manager)")
    truncated = bool(extra_info["truncated"])
    verdict = verify_math(solution_str, ground_truth)
    finish = "length" if truncated else "stop"
    return {
        "score": shaped_reward(verdict.correct, finish, REWARD_MODE),
        "acc": float(verdict.correct),
        "truncated": float(truncated),
    }
