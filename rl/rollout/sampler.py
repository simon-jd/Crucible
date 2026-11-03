"""Rollout data contract and strict vLLM chosen-token extraction.

Logprobs are V1 RAW_LOGPROBS: FP32 log-softmax of model logits before any
sampling temperature, truncation or penalties. They are model logps, not a
non-neutral sampler's behavior logps. Learner/ref scoring must use the same
raw model distribution. Every generated token (including emitted EOS/stop
IDs) has one entry; text skips special tokens and need not round-trip to IDs.
Only EOS and max_tokens stop generation; custom string stops are not used.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str

    def __post_init__(self):
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("prompt_id must be a nonempty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("prompt text must be a nonempty string")


@dataclass(frozen=True)
class Sample:
    prompt_id: str
    text: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    policy_version: int


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 32
    max_model_len: int = 256

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.temperature) or self.temperature < 0.01:
            raise ValueError("temperature must be >=0.01 for stochastic sampling")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ValueError("top_k must be -1 or positive")
        if any(type(v) is not int or v <= 0 for v in (self.max_tokens, self.max_model_len)):
            raise ValueError("token limits must be positive integers")
        if self.max_tokens >= self.max_model_len:
            raise ValueError("max_tokens must leave room for prompt tokens")


def samples_from_outputs(prompts, outputs, group_size, policy_version):
    """Preserve prompt order, completion index, and each chosen token's logprob."""
    if len(outputs) != len(prompts):
        raise ValueError("backend returned wrong prompt count")
    samples = []
    for prompt, request in zip(prompts, outputs, strict=True):
        completions = sorted(request.outputs, key=lambda output: output.index)
        if [c.index for c in completions] != list(range(group_size)):
            raise ValueError("backend returned wrong completion count/indices")
        for output in completions:
            ids = tuple(output.token_ids)
            if output.logprobs is None or len(ids) != len(output.logprobs):
                raise ValueError("missing per-token logprobs")
            logps = []
            for token, mapping in zip(ids, output.logprobs, strict=True):
                if mapping is None or token not in mapping:
                    raise ValueError("missing chosen-token logprob")
                value = float(mapping[token].logprob)
                if not math.isfinite(value) or value > 1e-6:
                    raise ValueError("invalid chosen-token logprob")
                logps.append(value)
            samples.append(Sample(prompt.prompt_id, output.text, ids, tuple(logps), policy_version))
    return samples
