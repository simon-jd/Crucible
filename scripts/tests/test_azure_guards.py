import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUB = "11111111-1111-1111-1111-111111111111"
GROUP = f"/subscriptions/{SUB}/resourceGroups/rg-crucible-lab"
TAGS = {
    "project": "crucible",
    "owner": "codex-crucible",
    "ttl": "2000-01-01T00:00:00Z",
    "est_usd_per_hour": "0",
}


def resource(name="vm", **changes):
    return {
        "id": f"{GROUP}/providers/Microsoft.Compute/virtualMachines/{name}",
        "resourceGroup": "rg-crucible-lab",
        "tags": dict(TAGS),
        **changes,
    }


@pytest.fixture
def azure(tmp_path):
    data = {
        "account": {"id": SUB, "name": "Azure for Students", "state": "Enabled"},
        "group": {"id": GROUP, "name": "rg-crucible-lab", "tags": dict(TAGS)},
        "resources": [resource()],
    }
    fixture = tmp_path / "fixture.json"
    log = tmp_path / "calls.jsonl"
    shim = tmp_path / "az"
    shim.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
d = json.loads(Path(os.environ["AZ_FIXTURE"]).read_text())
a = sys.argv[1:]
with open(os.environ["AZ_CALLS"], "a") as f: f.write(json.dumps(a) + "\\n")
if d.get("failure") == a[:2]:
    print("fixture CLI failure", file=sys.stderr); sys.exit(17)
if a[:2] == ["account", "show"]: result = d["account"]
elif a[:2] == ["group", "exists"]: result = d.get("exists", True)
elif a[:2] == ["group", "show"]:
    calls = [json.loads(x)[:2] for x in Path(os.environ["AZ_CALLS"]).read_text().splitlines()]
    result = d.get("fresh_group", d["group"]) if calls.count(a[:2]) > 1 else d["group"]
elif a[:2] == ["resource", "list"]:
    calls = [json.loads(x)[:2] for x in Path(os.environ["AZ_CALLS"]).read_text().splitlines()]
    result = d.get("fresh_resources", d["resources"]) if calls.count(a[:2]) > 1 else d["resources"]
elif a[:2] == ["resource", "show"]:
    result = d.get("fresh", next(r for r in d["resources"] if r["id"] == a[a.index("--ids")+1]))
elif a[1] == "delete": result = None
else: sys.exit(19)
print(json.dumps(result))
""")
    shim.chmod(0o755)

    def run(*args, entry="azure_reaper.sh"):
        fixture.write_text(json.dumps(data))
        log.write_text("")
        env = dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            AZ_FIXTURE=str(fixture),
            AZ_CALLS=str(log),
        )
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / entry), *args],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return data, run


def deleted(calls):
    return [a for a in calls if a[1] == "delete"]


def test_reaper_plan_never_deletes_and_apply_pins_subscription(azure):
    _, run = azure
    result, calls = run()
    assert result.returncode == 0, result.stderr
    assert '"event": "plan"' in result.stdout
    assert not deleted(calls)
    result, calls = run("--apply")
    assert result.returncode == 0, result.stderr
    assert len(deleted(calls)) == 1
    assert any(a[:2] == ["resource", "show"] for a in calls)
    assert all(a[a.index("--subscription") + 1] == SUB for a in calls[1:])


@pytest.mark.parametrize(
    "change",
    [
        {"tags": {}},
        {"tags": {**TAGS, "owner": "other"}},
        {"tags": {**TAGS, "project": "keel"}},
        {"tags": {**TAGS, "ttl": "tomorrow"}},
        {"tags": {**TAGS, "ttl": "2000-02-30T00:00:00Z"}},
        {"tags": {**TAGS, "ttl": "2000-01-01T00:00:00+01:00"}},
        {"tags": {**TAGS, "ttl": "2999-01-01T00:00:00Z"}},
        {"tags": {**TAGS, "est_usd_per_hour": "NaN"}},
        {"id": resource()["id"].replace(SUB, "22222222-2222-2222-2222-222222222222")},
        {"id": resource()["id"].replace("rg-crucible-lab", "rg-mlinfra-westus")},
        {"resourceGroup": "rg-mlinfra-westus"},
    ],
)
def test_reaper_skips_unsafe_resources(azure, change):
    data, run = azure
    data["resources"] = [resource(**change)]
    result, calls = run("--apply")
    assert result.returncode == 0, result.stderr
    assert not deleted(calls)
    assert '"event": "skip"' in result.stdout


@pytest.mark.parametrize(
    "change",
    [
        {"tags": {**TAGS, "owner": "other"}},
        {"tags": {**TAGS, "ttl": "2999-01-01T00:00:00Z"}},
        {"tags": {**TAGS, "ttl": "2001-01-01T00:00:00Z"}},
        {"id": resource()["id"].replace("rg-crucible-lab", "rg-mlinfra-westus")},
    ],
)
def test_reaper_refuses_changed_fresh_resource(azure, change):
    data, run = azure
    data["fresh"] = resource(**change)
    result, calls = run("--apply")
    assert result.returncode != 0
    assert not deleted(calls)


@pytest.mark.parametrize("entry", ["azure_reaper.sh", "azure_down.sh"])
def test_unknown_arguments_fail_before_cli(azure, entry):
    _, run = azure
    result, calls = run("--resource-group", "rg-mlinfra-westus", entry=entry)
    assert result.returncode != 0
    assert not calls


@pytest.mark.parametrize(
    "failure",
    [["account", "show"], ["group", "exists"], ["resource", "list"], ["resource", "delete"]],
)
def test_cli_failures_are_visible_and_nonzero(azure, failure):
    data, run = azure
    data["failure"] = failure
    result, _ = run("--apply")
    assert result.returncode != 0
    assert "fixture CLI failure" in result.stderr


def test_wrong_account_fails_without_resource_reads(azure):
    data, run = azure
    data["account"]["name"] = "Production"
    result, calls = run("--apply")
    assert result.returncode != 0
    assert len(calls) == 1


def test_down_plan_and_explicit_apply(azure):
    _, run = azure
    result, calls = run(entry="azure_down.sh")
    assert result.returncode == 0, result.stderr
    assert not deleted(calls)
    result, calls = run("--apply", entry="azure_down.sh")
    assert result.returncode == 0, result.stderr
    assert deleted(calls)[0][:2] == ["group", "delete"]
    assert deleted(calls)[0][deleted(calls)[0].index("--name") + 1] == "rg-crucible-lab"
    assert sum(a[:2] == ["group", "show"] for a in calls) == 2
    assert sum(a[:2] == ["resource", "list"] for a in calls) == 2


@pytest.mark.parametrize("target", ["group", "resources"])
def test_down_rejects_mixed_or_unknown_ownership(azure, target):
    data, run = azure
    if target == "group":
        data["group"]["tags"] = {}
    else:
        data["resources"].append(resource("foreign", tags={**TAGS, "project": "other"}))
    result, calls = run("--apply", entry="azure_down.sh")
    assert result.returncode != 0
    assert not deleted(calls)


def test_down_rejects_foreign_resource_id(azure):
    data, run = azure
    data["resources"] = [resource(id=resource()["id"].replace(SUB, "other-sub"))]
    result, calls = run("--apply", entry="azure_down.sh")
    assert result.returncode != 0
    assert not deleted(calls)


@pytest.mark.parametrize("entry", ["azure_reaper.sh", "azure_down.sh"])
def test_absent_group_is_explicit_noop(azure, entry):
    data, run = azure
    data["exists"] = False
    result, calls = run("--apply", entry=entry)
    assert result.returncode == 0, result.stderr
    assert '"event": "absent"' in result.stdout
    assert [a[:2] for a in calls] == [["account", "show"], ["group", "exists"]]


def test_resource_id_is_authority_when_optional_resource_group_field_absent(azure):
    data, run = azure
    del data["resources"][0]["resourceGroup"]
    result, calls = run("--apply")
    assert result.returncode == 0, result.stderr
    assert deleted(calls)


@pytest.mark.parametrize("fresh", ["fresh_group", "fresh_resources"])
def test_down_rejects_inventory_or_group_changes_before_delete(azure, fresh):
    data, run = azure
    if fresh == "fresh_group":
        data[fresh] = {**data["group"], "tags": {**TAGS, "owner": "other"}}
    else:
        data[fresh] = [resource(), resource("new")]
    result, calls = run("--apply", entry="azure_down.sh")
    assert result.returncode != 0
    assert not deleted(calls)


@pytest.mark.parametrize(
    "tags",
    [None, [], {**TAGS, "est_usd_per_hour": True}, {**TAGS, "ttl": 100}, {**TAGS, "ttl": ""}],
)
def test_malformed_tags_cannot_delete(azure, tags):
    data, run = azure
    data["resources"] = [resource(tags=tags)]
    result, calls = run("--apply")
    assert result.returncode == 0, result.stderr
    assert not deleted(calls)


def test_missing_cli_is_reported_without_attempted_cleanup(monkeypatch):
    from scripts.azure_guard import main

    def missing(*args, **kwargs):
        raise FileNotFoundError("shim: Azure CLI absent")

    monkeypatch.setattr("sys.argv", ["azure_guard.py", "reaper"])
    monkeypatch.setattr("scripts.azure_guard.subprocess.run", missing)
    assert main() == 1
