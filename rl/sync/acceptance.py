"""Trusted local run provenance; not authentication against a malicious host.

Callers verify exact owned manifests through VersionStore before passing their
version→digest map. Only the parent that joined the source may write a binding.
No model imports, process supervision, or historical-evidence rewriting here.
"""

import hashlib
import json
from pathlib import Path

PROTOCOL_SHA256 = "a9d94ba329dddb322156549bd1492a26d8387062d7585fb292b3ff0bd7455b4c"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def load_protocol(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("frozen confirmation protocol changed")
    return json.loads(raw)


def _expected(run, protocol_path, snapshots, source_pid, source_exitcode):
    protocol = load_protocol(protocol_path)
    versions = {str(version) for version in protocol["versions"]}
    if set(snapshots) != versions or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in snapshots.values()
    ):
        raise ValueError("complete verified snapshot digest map required")
    if (
        type(source_pid) is not int
        or source_pid <= 0
        or type(source_exitcode) is not int
        or source_exitcode != 0
    ):
        raise ValueError("binding requires an actually joined source exit0")
    source_raw = (Path(run) / "source.json").read_bytes()
    source = json.loads(source_raw)
    if (
        source.get("base_revision") != BASE_REVISION
        or source.get("source_pid") != source_pid
        or source.get("source_step", {}).get("policy_version") != max(protocol["versions"])
        or source.get("failed_learner_publication_rejected") is not True
    ):
        raise ValueError("source report does not match confirmed learner process")
    return {
        "format": 1,
        "run_directory": str(Path(run).resolve()),
        "base_revision": BASE_REVISION,
        "protocol_sha256": PROTOCOL_SHA256,
        "snapshots": dict(snapshots),
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "source_process": {"pid": source_pid, "exitcode": source_exitcode},
        "reference": "exact published BF16 weights; HF SDPA batch1",
        "inference": "vLLM0.10.2 V1 BF16 raw_logits batch2; full vocabulary",
    }


def write_binding(run, protocol_path, snapshots, *, source_pid, source_exitcode):
    """Bind THIS fresh run's content, never the historical protocol's weight IDs."""
    binding = _expected(run, protocol_path, snapshots, source_pid, source_exitcode)
    with (Path(run) / "binding.json").open("x") as stream:
        json.dump(binding, stream, indent=2)
        stream.write("\n")
    return binding


def validate_reuse(run, protocol_path, snapshots):
    """Refuse missing/changed source, protocol, exit proof or owned snapshot content."""
    binding = json.loads((Path(run) / "binding.json").read_text())
    try:
        process = binding["source_process"]
        expected = _expected(run, protocol_path, snapshots, process["pid"], process["exitcode"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid source-process binding") from exc
    if binding != expected:
        raise ValueError("run binding/source/snapshot identity changed")
    return binding
