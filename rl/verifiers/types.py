"""Trusted verifier boundary types. Sandbox implementations are trusted code."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Verdict:
    correct: bool
    reason: str

    @property
    def reward(self) -> float:
        return 1.0 if self.correct else 0.0


@dataclass(frozen=True)
class Case:
    input: Any
    expected: Any


@dataclass(frozen=True)
class TestBundle:
    """JSON cases for `solve(value)`. Hidden expectations must never enter prompts."""

    __test__ = False
    public: tuple[Case, ...]
    hidden: tuple[Case, ...]


@dataclass(frozen=True)
class RunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    completed: bool = False
    cleanup_confirmed: bool = False
    timed_out: bool = False
    output_truncated: bool = False
    error: str = ""
    run_id: str | None = None
    idempotency_key: str | None = None


class Sandbox(Protocol):
    def run(self, solution: str, case_input: Any, public_tests: str, timeout_s: float) -> RunResult:
        """Run one fresh isolated process. Never execute candidate code on host.

        public_tests is canonical JSON (public cases only). Return stdout as
        untrusted answer data; completion/cleanup fields come from the runtime.
        """
        ...
