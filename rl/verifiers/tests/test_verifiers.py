"""CPU contracts. FakeSandbox is evidence of orchestration, not isolation."""

import pytest

from rl.rewards.hacks import detect_hacks
from rl.verifiers.code import Case, RunResult, TestBundle, verify_code
from rl.verifiers.math import verify_math


class FakeSandbox:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def run(self, solution, case_input, public_tests, timeout_s):
        self.requests.append((solution, case_input, public_tests, timeout_s))
        return self.result


def bundle():
    return TestBundle(public=(Case(1, 2),), hidden=(Case(2, 3),))


@pytest.mark.parametrize("prediction", ["1/2", "0.5", r"\frac{1}{2}", r"work \boxed{\frac{1}{2}}"])
def test_math_equivalent_forms(prediction):
    assert verify_math(prediction, "1/2").correct


def test_math_wrong():
    assert not verify_math("2/3", "1/2").correct


def test_code_timeout_reward_zero():
    result = verify_code(
        "def solve(x): return x+1", bundle(), FakeSandbox(RunResult(timed_out=True))
    )
    assert not result.correct and result.reward == 0 and "timeout" in result.reason


def test_code_cannot_modify_tests():
    candidate = 'def solve(x):\n open("/tests/public.json", "w").write("hacked")\n return x+1'
    sandbox = FakeSandbox(RunResult())
    result = verify_code(candidate, bundle(), sandbox)
    assert not result.correct and "write_tests" in result.reason
    assert (
        not sandbox.requests
    )  # Static rejection; real mount enforcement needs runtime acceptance.


def test_detect_sys_exit_hack():
    assert "early_exit" in detect_hacks("import sys\nsys.exit(0)", "")


def test_stdout_and_exit_zero_are_not_success():
    result = verify_code(
        "def solve(x): return x",
        bundle(),
        FakeSandbox(RunResult(stdout="PASS", exit_code=0, completed=True, cleanup_confirmed=True)),
    )
    assert not result.correct


@pytest.mark.parametrize(
    "prediction",
    [
        '__import__("os").system("echo unsafe")',
        "1/2 garbage",
        "1/2)",
        r"\frac{1}{2}x",
        r"\boxed{\frac{1}{2}",
        "0/0",
        "1/0",
        "0^0",
        "9^999999",
        "(x^8)^8^8",
        "1e-1",
        "sin(4)",
        "1" * 4097,
        "(" * 30 + "1" + ")" * 30,
    ],
)
def test_math_rejects_unsupported_and_malicious(prediction):
    assert not verify_math(prediction, "1/2").correct


def test_math_nested_fraction_and_symbolic_equivalence():
    assert verify_math(r"\boxed{\frac{\frac{1}{2}}{2}}", "1/4").correct
    assert verify_math("(x+1)*(x-1)", "x^2-1").correct
    assert not verify_math("-2^2", "4").correct


@pytest.mark.parametrize(
    "candidate,flag",
    [
        ("import sys as s\ns.exit()", "early_exit"),
        ("from os import _exit as done\ndone(0)", "early_exit"),
        ("raise SystemExit(0)", "early_exit"),
        ("import unittest as u\nu.TestCase.assertEqual = lambda *a: None", "monkeypatch_tests"),
        ('import unittest\nsetattr(unittest, "TestCase", object)', "monkeypatch_tests"),
        ('print("expected output")', "literal_output"),
        ('open("/tests/public.json", mode="w")', "write_tests"),
    ],
)
def test_hack_signals(candidate, flag):
    assert flag in detect_hacks(candidate, "")


@pytest.mark.parametrize(
    "prediction,reference",
    [
        ("sqrt(8)", "2*sqrt(2)"),
        (r"\boxed{\sqrt{\frac{1}{4}}}", "0.5"),
        (r"2*\pi/2", "pi"),
        ("sqrt(x^2)", "sqrt(x*x)"),
    ],
)
def test_math_radicals_and_pi(prediction, reference):
    assert verify_math(prediction, reference).correct


def test_code_trusted_grader_runs_every_case_with_hidden_expected_not_sent():
    class AnswerFixture:
        def __init__(self):
            self.requests = []

        def run(self, solution, case_input, public_tests, timeout_s):
            import json

            self.requests.append((case_input, public_tests))
            return RunResult(
                stdout=json.dumps(case_input + 1),
                exit_code=0,
                completed=True,
                cleanup_confirmed=True,
            )

    sandbox = AnswerFixture()
    result = verify_code("def solve(x): return x+1", bundle(), sandbox)
    assert result.correct and result.reward == 1
    assert [request[0] for request in sandbox.requests] == [1, 2]
    assert all("3" not in request[1] for request in sandbox.requests)


def test_hidden_case_defeats_fixed_public_answer():
    result = verify_code(
        "def solve(x): return 2",
        bundle(),
        FakeSandbox(RunResult(stdout="2", exit_code=0, completed=True, cleanup_confirmed=True)),
    )
    assert not result.correct and result.reason == "wrong_answer:case=1"


@pytest.mark.parametrize("output", ["true", "2\nPASS", "NaN", '{"a":1,"a":2}', "2" * 65537])
def test_code_malformed_or_wrong_output(output):
    assert not verify_code(
        "def solve(x): return x",
        bundle(),
        FakeSandbox(RunResult(stdout=output, exit_code=0, completed=True, cleanup_confirmed=True)),
    ).correct


def test_code_completion_and_cleanup_are_required():
    for result in [
        RunResult(stdout="2", exit_code=0),
        RunResult(stdout="2", exit_code=0, completed=True),
        RunResult(
            stdout="2", exit_code=0, completed=True, cleanup_confirmed=True, output_truncated=True
        ),
    ]:
        assert not verify_code("def solve(x): return x", bundle(), FakeSandbox(result)).correct


def test_code_requires_hidden_cases_and_bounded_json():
    assert not verify_code(
        "def solve(x): return x", TestBundle(public=(Case(1, 1),), hidden=()), FakeSandbox(None)
    ).correct
    assert not verify_code(
        "def solve(x): return x",
        TestBundle(public=(Case(float("nan"), 1),), hidden=(Case(2, 2),)),
        FakeSandbox(None),
    ).correct


def test_math_reasons_and_worker_timeout(monkeypatch):
    import subprocess

    assert verify_math("sin(1)", "1").reason == "invalid_math:unsupported_syntax"
    assert verify_math(r"\boxed{1/2} \boxed", "1/2").reason == "invalid_math:malformed_box"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("rl.verifiers.math.subprocess.run", timeout)
    verdict = verify_math("1/2", "0.5")
    assert not verdict.correct and verdict.reason == "math_timeout"


def test_ast_and_unicode_source_bounds():
    assert detect_hacks("a" + ".b" * 1000 + "()", "") == ["source_complexity"]
    assert not verify_code("\ud800", bundle(), FakeSandbox(None)).correct


@pytest.mark.parametrize(
    "result",
    [
        None,
        RunResult(stdout="2", exit_code=False, completed=True, cleanup_confirmed=True),
        RunResult(stdout="2", exit_code=0, completed="yes", cleanup_confirmed=True),
    ],
)
def test_invalid_sandbox_result_fails_closed(result):
    assert not verify_code("def solve(x): return x", bundle(), FakeSandbox(result)).correct


def test_uncertain_post_key_survives_verdict():
    verdict = verify_code(
        "def solve(x): return x",
        bundle(),
        FakeSandbox(RunResult(timed_out=True, idempotency_key="retry-key")),
    )
    assert not verdict.correct and "idempotency_key=retry-key" in verdict.reason
