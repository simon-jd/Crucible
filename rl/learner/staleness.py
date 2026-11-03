"""Bounded policy age; future versions are invalid, never silently accepted."""

from rl.rollout.sampler import Sample


def filter_stale(samples, current_version: int, k: int) -> tuple[list[Sample], int]:
    if type(current_version) is not int or current_version < 0 or type(k) is not int or k < 0:
        raise ValueError("current_version and k must be nonnegative integers")
    accepted = []
    dropped = 0
    for sample in samples:
        version = sample.policy_version
        if type(version) is not int or version < 0:
            raise ValueError("sample policy_version must be a nonnegative integer")
        if version > current_version:
            raise ValueError("sample has a future policy version")
        if current_version - version > k:
            dropped += 1
        else:
            accepted.append(sample)
    return accepted, dropped
