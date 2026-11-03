"""Synthetic isolated CPU contracts; never execute the monitor launch path."""

import ast
import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL = ROOT / "docs/evidence/raw/task15-cpu-go-monitor.py"
CANDIDATE = ROOT / "docs/evidence/raw/task15-cpu-round1-go-monitor.py"


# The unchanged group() may skip an individual /proc entry that has already
# exited. This contract covers errors propagated from the whole operation into
# scan(), including independently anchored leader/group identity operations.
def actual_scan():
    source = Path(os.environ.get("TASK15_ROUND1_MONITOR", str(CANDIDATE))).resolve()
    assert source in (HISTORICAL, CANDIDATE)
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scan"
    )
    namespace = {"record": {"scan_errors": []}}
    # Compile the actual source node, never copy scan logic or execute imports/launch.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - root-approved extraction of the actual local scan node only
    return namespace["scan"], namespace["record"]["scan_errors"]


@pytest.mark.parametrize(
    "label",
    ["owned", "canonical", "initial_leader", "leader", "group", "signal_anchor", "terminal_group"],
)
@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("synthetic denied"),
        OSError(errno.EIO, "synthetic EIO"),
        ValueError("synthetic malformed"),
        RuntimeError("unexpected owned symlink"),
    ],
)
def test_monitor_nontransient_scan_error_aborts(label, failure):
    scan, errors = actual_scan()
    calls = []

    def operation():
        calls.append("whole operation")
        if len(calls) == 1:
            raise failure
        return "would incorrectly succeed on retry"

    with pytest.raises(type(failure)) as caught:
        scan(label, operation)
    assert caught.value is failure
    assert calls == ["whole operation"]
    assert errors == [{"scan": label, "attempt": 0, "error": str(failure)}]


@pytest.mark.parametrize("label", ["owned", "canonical"])
def test_monitor_transient_disappearance_rescans_whole_operation(label):
    scan, errors = actual_scan()
    visits = []
    calls = 0
    failure = FileNotFoundError("synthetic temporary file disappeared")

    def operation():
        nonlocal calls
        calls += 1
        # Each call starts a whole byte scan from the first entry. No partial
        # successful prefix is returned after the first scan loses its last entry.
        visits.append((calls, "first-entry"))
        if calls == 1:
            raise failure
        visits.append((calls, "last-entry"))
        return 41 + 1

    assert scan(label, operation) == 42
    assert visits == [(1, "first-entry"), (2, "first-entry"), (2, "last-entry")]
    assert calls == 2
    assert errors == [{"scan": label, "attempt": 0, "error": str(failure)}]


@pytest.mark.parametrize("label", ["owned", "canonical"])
def test_monitor_repeated_disappearance_is_bounded_and_retained(label):
    scan, errors = actual_scan()
    failures = []

    def operation():
        failure = FileNotFoundError(f"synthetic disappearing file {len(failures)}")
        failures.append(failure)
        raise failure

    with pytest.raises(FileNotFoundError) as caught:
        scan(label, operation)
    assert len(failures) == 3 and caught.value is failures[-1]
    assert errors == [
        {"scan": label, "attempt": i, "error": str(f)} for i, f in enumerate(failures)
    ]


@pytest.mark.parametrize(
    "label", ["initial_leader", "leader", "group", "signal_anchor", "terminal_group", "unknown"]
)
def test_monitor_process_identity_disappearance_aborts_immediately(label):
    scan, errors = actual_scan()
    calls = []
    failure = FileNotFoundError("synthetic process identity disappeared")

    def operation():
        calls.append("whole operation")
        if len(calls) == 1:
            raise failure
        return "would incorrectly succeed on retry"

    with pytest.raises(FileNotFoundError) as caught:
        scan(label, operation)
    assert caught.value is failure and len(calls) == 1
    assert errors == [{"scan": label, "attempt": 0, "error": str(failure)}]


def test_counter_wording_and_eos_stop_limit_in_three_files():
    paths = [
        ROOT / "spec/eval_acceptance.py",
        ROOT / "spec/README.md",
        ROOT / "docs/evidence/spec-decoding.md",
    ]
    tree = ast.parse(paths[0].read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "acceptance_length"
    )
    texts = [ast.get_docstring(function), *(path.read_text() for path in paths[1:])]
    for text in texts:
        text = " ".join(text.split())
        assert "counter-based acceptance length" in text.lower()
        assert "guaranteed verification token" in text
        assert "EOS/stop" in text and "completion_tokens" in text
        assert "attestation" in text
        assert "emitted tokens per verification" not in text


def test_candidate_changes_only_scan_and_module_description():
    old, new = [ast.parse(path.read_text()) for path in (HISTORICAL, CANDIDATE)]
    for tree in (old, new):
        tree.body = [
            node
            for i, node in enumerate(tree.body)
            if i != 0 and not (isinstance(node, ast.FunctionDef) and node.name == "scan")
        ]
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_historical_execution_identity_and_clean_scan_receipts():
    expected = "f29319324bcb5f753a2b6828276da41b651114beb37b1ced3a701d0d907871cd"
    assert hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() == expected
    for phase in ("smoke", "red", "green", "vet"):
        receipt = json.loads((ROOT / f"docs/evidence/raw/task15-cpu-go-{phase}.json").read_text())
        assert receipt["monitor_sha256"] == expected and receipt["scan_errors"] == []
