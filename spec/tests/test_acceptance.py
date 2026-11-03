"""Synthetic CPU fixtures; these are not measured model observations."""

import json
from copy import deepcopy

import pytest

from rl.eval.heldout import digest, select_heldout
from spec.eval_acceptance import acceptance_length, build_plan, evaluate, main, validate_plan


def test_acceptance_length_on_fixture():
    traces = [
        {"request_id": "synthetic-a", "spec_accepted_tokens": 6, "spec_verify_ct": 2},
        {"request_id": "synthetic-b", "spec_accepted_tokens": 0, "spec_verify_ct": 6},
    ]
    assert acceptance_length(traces) == pytest.approx(1.75)


def test_acceptance_guaranteed_token_and_zero_verification():
    assert (
        acceptance_length(
            [{"request_id": "synthetic-a", "spec_accepted_tokens": 0, "spec_verify_ct": 4}]
        )
        == 1.0
    )
    with pytest.raises(ValueError):
        acceptance_length(
            [{"request_id": "synthetic-a", "spec_accepted_tokens": 0, "spec_verify_ct": 0}]
        )
    assert (
        acceptance_length(
            [
                {"request_id": "synthetic-a", "spec_accepted_tokens": 0, "spec_verify_ct": 0},
                {"request_id": "synthetic-b", "spec_accepted_tokens": 2, "spec_verify_ct": 1},
            ]
        )
        == 3.0
    )


@pytest.mark.parametrize(
    "traces",
    [
        [],
        {},
        [{"request_id": "", "spec_accepted_tokens": 1, "spec_verify_ct": 1}],
        [{"request_id": " ", "spec_accepted_tokens": 1, "spec_verify_ct": 1}],
        [{"request_id": 1, "spec_accepted_tokens": 1, "spec_verify_ct": 1}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": True, "spec_verify_ct": 1}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 1, "spec_verify_ct": False}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": -1, "spec_verify_ct": 1}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 1, "spec_verify_ct": -1}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 1.0, "spec_verify_ct": 1}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 1, "spec_verify_ct": "1"}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 1, "spec_verify_ct": 0}],
        [{"request_id": "synthetic-a", "spec_verify_ct": 1}],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 1, "spec_verify_ct": 1, "x": 0}],
        [
            {"request_id": "synthetic-a", "spec_accepted_tokens": 1, "spec_verify_ct": 1},
            {"request_id": "synthetic-a", "spec_accepted_tokens": 2, "spec_verify_ct": 2},
        ],
        [{"request_id": "synthetic-a", "spec_accepted_tokens": 10**1000, "spec_verify_ct": 1}],
    ],
)
def test_acceptance_rejects_invalid_terminal_counters(traces):
    with pytest.raises(ValueError):
        acceptance_length(traces)


def fixture_plan():
    train = [
        {"id": "synthetic-train", "question": "synthetic training question", "source": "fixture"}
    ]
    candidates = [
        {
            "id": "synthetic-p1",
            "question": "synthetic question one",
            "answer": "gold-one",
            "source": "fixture",
        },
        {
            "id": "synthetic-p2",
            "question": "synthetic question two",
            "answer": "gold-two",
            "source": "fixture",
        },
    ]
    manifest = select_heldout(train, candidates, count=2, seed=15, synthetic_fixture=True)
    identity = {"name": "synthetic-identity", "sha256": "a" * 64}
    requests = [
        {
            "request_id": "synthetic-r1",
            "prompt_id": "synthetic-p1",
            "batch_id": "synthetic-b2",
            "batch_size": 2,
            "slot": 0,
        },
        {
            "request_id": "synthetic-r2",
            "prompt_id": "synthetic-p2",
            "batch_id": "synthetic-b2",
            "batch_size": 2,
            "slot": 1,
        },
    ]
    return build_plan(
        manifest,
        candidates,
        protocol=identity,
        target=identity,
        draft=identity,
        requests=requests,
        synthetic_fixture=True,
    )


def fixture_outcomes(plan):
    return {
        "schema_version": 1,
        "plan_sha256": plan["sha256"],
        "synthetic_fixture": True,
        "rows": [
            {
                "request_id": "synthetic-r1",
                "status": "completed",
                "spec_accepted_tokens": 6,
                "spec_verify_ct": 2,
            },
            {
                "request_id": "synthetic-r2",
                "status": "completed",
                "spec_accepted_tokens": 0,
                "spec_verify_ct": 6,
            },
        ],
    }


def test_complete_plan_report_order_and_input_immutability():
    plan = fixture_plan()
    outcomes = fixture_outcomes(plan)
    before = deepcopy((plan, outcomes))
    report = evaluate(plan, outcomes)
    outcomes["rows"].reverse()
    assert evaluate(plan, outcomes) == report
    outcomes["rows"].reverse()
    assert (plan, outcomes) == before
    assert report["counts"] == {
        "planned": 2,
        "observed": 2,
        "completed": 2,
        "failed": 0,
        "not_run": 0,
        "missing": 0,
    }
    assert report["counter_statistic"]["acceptance_length"] == 1.75
    assert report["per_batch"][0]["counter_statistic"] == report["counter_statistic"]
    assert report["coverage_complete"] and report["all_requests_completed"]
    assert report["complete_counter_report"] and report["synthetic_fixture"] is True
    assert "unverified" in report["authenticity"]
    assert not any("eligible" in key or "benchmark" in key for key in report)
    report["requests"][0]["slot"] = 100
    assert plan == before[0]


def rebuild(plan, **changes):
    values = {
        "manifest": plan["heldout_manifest"],
        "selected_records": plan["selected_records"],
        **{k: plan[k] for k in ("protocol", "target", "draft", "requests", "synthetic_fixture")},
    }
    values.update(changes)
    return build_plan(**values)


def test_canonical_plan_order_and_all_content_bindings():
    plan = fixture_plan()
    assert (
        rebuild(
            plan,
            requests=list(reversed(plan["requests"])),
            selected_records=list(reversed(plan["selected_records"])),
        )
        == plan
    )
    for field in ("protocol", "target", "draft"):
        changed = deepcopy(plan[field])
        changed["sha256"] = "b" * 64
        assert rebuild(plan, **{field: changed})["sha256"] != plan["sha256"]
    requests = deepcopy(plan["requests"])
    for request in requests:
        request["batch_id"] = "synthetic-other-batch"
    assert rebuild(plan, requests=requests)["sha256"] != plan["sha256"]
    records = deepcopy(plan["selected_records"])
    for field in ("question", "answer", "source"):
        changed = deepcopy(records)
        changed[0][field] = "synthetic altered content"
        with pytest.raises(ValueError):
            rebuild(plan, selected_records=changed)
    manifest = deepcopy(plan["heldout_manifest"])
    manifest["train"][0]["record_sha256"] = "c" * 64
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    changed = rebuild(plan, manifest=manifest)
    assert changed["sha256"] != plan["sha256"]
    with pytest.raises(ValueError):
        evaluate(changed, fixture_outcomes(plan))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(sha256="0" * 64),
        lambda p: p.update(schema_version=True),
        lambda p: p.update(synthetic_fixture=1),
        lambda p: p.update(extra=1),
        lambda p: p["selected_records"][0].update(answer="tampered gold"),
        lambda p: p["requests"][0].update(slot=True),
        lambda p: p["requests"][0].update(batch_size=65),
        lambda p: p["requests"][0].update(batch_id=""),
        lambda p: p["requests"][0].update(prompt_id="unknown"),
        lambda p: p["requests"][1].update(request_id="synthetic-r1"),
        lambda p: p["requests"][1].update(slot=0),
        lambda p: p["requests"].pop(),
        lambda p: p["requests"].reverse(),
        lambda p: p["protocol"].update(sha256="ABC"),
    ],
)
def test_plan_tampering_rejected(mutate):
    plan = fixture_plan()
    mutate(plan)
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_missing_failed_unknown_and_not_run_preserved():
    plan = fixture_plan()
    outcomes = fixture_outcomes(plan)
    outcomes["rows"] = [
        {"request_id": "synthetic-r1", "status": "failed", "reason": "synthetic fault"}
    ]
    report = evaluate(plan, outcomes)
    assert report["counts"] == {
        "planned": 2,
        "observed": 1,
        "completed": 0,
        "failed": 1,
        "not_run": 0,
        "missing": 1,
    }
    assert report["counter_statistic"]["acceptance_length"] is None
    assert "spec_verify_ct" not in report["failed_prefixes"][0]
    assert report["requests"][1]["status"] == "missing"
    assert not report["coverage_complete"] and not report["complete_counter_report"]
    outcomes["rows"].append(
        {"request_id": "synthetic-r2", "status": "not_run", "reason": "synthetic skipped"}
    )
    assert evaluate(plan, outcomes)["counts"]["not_run"] == 1
    outcomes["rows"][1] = fixture_outcomes(plan)["rows"][1]
    report = evaluate(plan, outcomes)
    assert report["coverage_complete"] and not report["all_requests_completed"]
    assert report["counter_statistic"]["acceptance_length"] == 1.0
    outcomes["rows"][0].update(spec_accepted_tokens=100, spec_verify_ct=1)
    report = evaluate(plan, outcomes)
    assert report["counter_statistic"]["acceptance_length"] == 1.0
    assert report["failed_prefixes"][0]["spec_accepted_tokens"] == 100
    outcomes["rows"][0].update(spec_accepted_tokens=None, spec_verify_ct=None)
    assert evaluate(plan, outcomes)["failed_prefixes"][0]["spec_verify_ct"] is None


def test_zero_verification_report_is_unavailable():
    plan = fixture_plan()
    outcomes = fixture_outcomes(plan)
    for row in outcomes["rows"]:
        row.update(spec_accepted_tokens=0, spec_verify_ct=0)
    report = evaluate(plan, outcomes)
    assert report["all_requests_completed"] and report["coverage_complete"]
    assert report["counter_statistic"]["acceptance_length"] is None
    assert not report["complete_counter_report"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(plan_sha256="a" * 64),
        lambda o: o.update(synthetic_fixture=False),
        lambda o: o.update(synthetic_fixture=1),
        lambda o: o.update(schema_version=True),
        lambda o: o.update(extra=0),
        lambda o: o.update(rows={}),
        lambda o: o["rows"].append(deepcopy(o["rows"][0])),
        lambda o: o["rows"][0].update(request_id="unknown"),
        lambda o: o["rows"][0].update(status="streamed"),
        lambda o: o["rows"][0].update(spec_verify_ct=True),
        lambda o: o["rows"][0].update(spec_verify_ct=0),
        lambda o: o["rows"][0].update(extra=0),
        lambda o: o["rows"][0].update(status="failed"),
        lambda o: o["rows"].__setitem__(
            0,
            {
                "request_id": "synthetic-r1",
                "status": "failed",
                "reason": "synthetic",
                "spec_verify_ct": 1,
            },
        ),
        lambda o: o["rows"].__setitem__(
            0,
            {
                "request_id": "synthetic-r1",
                "status": "failed",
                "reason": "synthetic",
                "spec_verify_ct": 1,
                "spec_accepted_tokens": None,
            },
        ),
        lambda o: o["rows"][0].update(status="not_run", reason="synthetic"),
    ],
)
def test_outcome_invalid_rebindings_and_snapshots_rejected(mutate):
    plan = fixture_plan()
    outcomes = fixture_outcomes(plan)
    mutate(outcomes)
    with pytest.raises(ValueError):
        evaluate(plan, outcomes)


def test_cli_json_errors_are_strict_and_untrusted(tmp_path, capsys):
    path = tmp_path / "synthetic-invalid.json"
    for body in ('{"synthetic_fixture":true,"synthetic_fixture":false}', '{"x":NaN}', "{"):
        path.write_text(body)
        assert main(["--plan", str(path), "--outcomes", str(path)]) == 2
        output = json.loads(capsys.readouterr().out)
        assert output["synthetic_fixture"] is None and "error" in output
