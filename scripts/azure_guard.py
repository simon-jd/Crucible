"""Fixed-scope Azure cleanup. No provision operations; plan is the default."""

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID

GROUP = "rg-crucible-lab"
UTC_TTL = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)")


def event(kind, **details):
    print(json.dumps({"event": kind, **details}), flush=True)


def az(*args, subscription=None):
    command = ["az", *args]
    if subscription:
        command += ["--subscription", subscription]
    command += ["--output", "json", "--only-show-errors"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ValueError("Azure CLI unavailable; no cleanup performed") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Azure CLI failed ({exc.returncode}): {exc.stderr.strip()}") from exc
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    try:
        return json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError as exc:
        raise ValueError("Azure CLI returned invalid JSON") from exc


def account_subscription():
    account = az("account", "show")
    if not isinstance(account, dict) or account.get("name") != "Azure for Students":
        raise ValueError("Active account must be Azure for Students")
    if account.get("state") != "Enabled":
        raise ValueError("Active subscription is not Enabled")
    subscription = account.get("id")
    if not isinstance(subscription, str) or str(UUID(subscription)) != subscription.lower():
        raise ValueError("Invalid subscription ID")
    return subscription


def ttl(tags):
    if not isinstance(tags, dict):
        raise ValueError("missing tags")  # noqa: TRY004 - external JSON validation
    if tags.get("project") != "crucible" or tags.get("owner") != "codex-crucible":
        raise ValueError("unknown project/owner")
    value = tags.get("ttl")
    if not isinstance(value, str) or not UTC_TTL.fullmatch(value):
        raise ValueError("ttl must be an ISO UTC timestamp with Z or +00:00")
    try:
        expires = datetime.fromisoformat(value)
        if not isinstance(tags.get("est_usd_per_hour"), str):
            raise TypeError("cost estimate tag must be a string")
        estimate = Decimal(tags["est_usd_per_hour"])
        if not estimate.is_finite() or estimate < 0:
            raise ValueError("invalid cost estimate")
    except (KeyError, TypeError, InvalidOperation, ValueError) as exc:
        raise ValueError("invalid ttl or est_usd_per_hour tag") from exc
    return expires


def scoped_id(value, subscription, *, group=False):
    prefix = f"/subscriptions/{subscription}/resourceGroups/{GROUP}"
    if not isinstance(value, str):
        raise ValueError("missing resource ID")  # noqa: TRY004 - external JSON validation
    if group:
        valid = value.casefold() == prefix.casefold()
    else:
        suffix = value[len(prefix) :]
        valid = value[: len(prefix)].casefold() == prefix.casefold() and re.fullmatch(
            r"/providers/[^/\s]+(?:/[^/\s]+/[^/\s]+)+", suffix, re.IGNORECASE
        )
    if not valid:
        raise ValueError("ID outside fixed resource group/subscription or malformed")
    return value


def owned_resource(resource, subscription):
    if not isinstance(resource, dict):
        raise ValueError("malformed resource")  # noqa: TRY004 - external JSON validation
    identity = scoped_id(resource.get("id"), subscription)
    if "resourceGroup" in resource and str(resource["resourceGroup"]).casefold() != GROUP:
        raise ValueError("resourceGroup outside fixed scope")
    expires = ttl(resource.get("tags"))
    return identity, expires


def resources(subscription):
    result = az("resource", "list", "--resource-group", GROUP, subscription=subscription)
    if not isinstance(result, list):
        raise ValueError("resource list is not an array")  # noqa: TRY004
    return result


def reaper(subscription, apply):
    for resource in resources(subscription):
        try:
            identity, expires = owned_resource(resource, subscription)
            if expires >= datetime.now(UTC):
                raise ValueError("ttl has not expired")
        except ValueError as exc:
            event(
                "skip",
                id=resource.get("id") if isinstance(resource, dict) else None,
                reason=str(exc),
            )
            continue
        event("plan", action="delete-resource", id=identity)
        if apply:
            fresh = az("resource", "show", "--ids", identity, subscription=subscription)
            fresh_id, fresh_expiry = owned_resource(fresh, subscription)
            if (
                fresh_id != identity
                or fresh.get("tags") != resource.get("tags")
                or fresh_expiry >= datetime.now(UTC)
            ):
                raise ValueError(f"Resource changed before deletion: {identity}")
            az("resource", "delete", "--ids", identity, subscription=subscription)
            event("deleted", id=identity)


def group_snapshot(subscription):
    group = az("group", "show", "--name", GROUP, subscription=subscription)
    if not isinstance(group, dict) or str(group.get("name", "")).casefold() != GROUP:
        raise ValueError("group name does not match fixed scope")
    scoped_id(group.get("id"), subscription, group=True)
    ttl(group.get("tags"))
    inventory = resources(subscription)
    snapshot = {}
    for resource in inventory:
        identity, _ = owned_resource(resource, subscription)
        if identity.casefold() in snapshot:
            raise ValueError("duplicate resource ID")
        snapshot[identity.casefold()] = resource["tags"]
    return group["id"], group["tags"], snapshot


def down(subscription, apply):
    before = group_snapshot(subscription)
    event("plan", action="delete-group", id=before[0], resource_count=len(before[2]))
    if apply:
        if group_snapshot(subscription) != before:
            raise ValueError("group ownership/inventory changed before deletion")
        az("group", "delete", "--name", GROUP, "--yes", subscription=subscription)
        event("deleted", id=before[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reaper", "down"])
    parser.add_argument("--apply", action="store_true", help="perform validated deletions")
    parser.add_argument("--plan", action="store_true", help="read-only (default)")
    args = parser.parse_args()
    if args.apply and args.plan:
        parser.error("--apply and --plan are mutually exclusive")
    try:
        subscription = account_subscription()
        event(
            "scope",
            subscription=subscription,
            resource_group=GROUP,
            mode="apply" if args.apply else "plan",
        )
        exists = az("group", "exists", "--name", GROUP, subscription=subscription)
        if exists is False:
            event("absent", resource_group=GROUP, action="none")
            return 0
        if exists is not True:
            raise ValueError("group exists returned a non-boolean result")
        (reaper if args.action == "reaper" else down)(subscription, args.apply)
    except (ValueError, OSError) as exc:
        print(json.dumps({"event": "error", "message": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
