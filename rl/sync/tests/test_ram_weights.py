"""CPU unit boundaries for RAM policy-weight export and readback."""

import os
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from rl.learner import handoff_stream as stream
from rl.learner import state_handoff as state
from rl.sync.ram_export import export as export_actual
from rl.sync.ram_weights import (
    BridgeBudget,
    RamPolicyCheckpoint,
    digest,
    make_receipt,
    provenance,
    read_weights,
    validate_source,
)
from rl.sync.tests.ram_worker import CAP, META, contract, source_artifact


def budget(**changes):
    return replace(
        BridgeBudget(CAP, CAP, 1_000_000_000, metadata_cap=META, ram_limit=2_000_000_000), **changes
    )


def export(*args):
    return export_actual(*args, provenance("/fixture", "fixture", ()))


def checkpoint(fd, source, plan):
    info = os.fstat(fd)
    receipt = make_receipt(fd, (info.st_dev, info.st_ino), source.version, source.binding, plan)
    return RamPolicyCheckpoint(
        "/fixture",
        receipt.sha256,
        receipt.version,
        receipt,
        digest(contract()),
        source.sha256,
        "fixture",
        (),
    )


def test_export_actual_bf16_source_immutable_and_no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU exporter queried CUDA")

    for name in (
        "is_initialized",
        "is_available",
        "device_count",
        "current_device",
        "init",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    # Source fixture calls existing learner validation; create before forbidding
    # is_initialized, which belongs to the existing capture contract.
    monkeypatch.undo()
    with source_artifact() as (source, receipt):
        expected_contract = contract()
        for name in (
            "is_initialized",
            "is_available",
            "device_count",
            "current_device",
            "init",
            "synchronize",
        ):
            monkeypatch.setattr(torch.cuda, name, forbidden)
        fd = os.memfd_create("weight-unit", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            export(source, receipt, fd, budget(), digest(expected_contract))
            cp = checkpoint_without_contract(fd, receipt, expected_contract)
            records, start = read_weights(fd, cp)
            saved, _, source_start = stream.read_metadata(source, receipt, budget().source_limits)
            for record in records:
                original = saved["policy"]["tensors"][record["name"]].metadata
                value = stream._allocate_record(torch, original)
                stream._read_record(source, value, source_start + original["offset"])
                converted = value.to(getattr(torch, record["dtype"][6:]))
                assert os.pread(fd, record["bytes"], start + record["offset"]) == bytes(
                    stream._raw(converted)
                )
            receipt.validate(source)
            with pytest.raises(OSError):
                os.write(fd, b"x")
        finally:
            os.close(fd)


def checkpoint_without_contract(fd, receipt, expected):
    info = os.fstat(fd)
    output = make_receipt(
        fd, (info.st_dev, info.st_ino), receipt.version, receipt.binding, budget()
    )
    return RamPolicyCheckpoint(
        "/fixture",
        output.sha256,
        output.version,
        output,
        digest(expected),
        receipt.sha256,
        "fixture",
        (),
    )


@pytest.mark.parametrize("fault", ["contract", "version", "alias", "optimizer", "reference", "rng"])
def test_source_contract_rejections(fault):
    with source_artifact() as (fd, receipt):
        saved, _, _ = stream.read_metadata(fd, receipt, budget().source_limits)
        expected = contract()
        if fault == "contract":
            expected["config"]["group_size"] = 8
        elif fault == "version":
            saved["version"] += 1
        elif fault == "alias":
            saved["policy"]["aliases"]["lm_head.weight"] = "missing"
        elif fault == "optimizer":
            saved["optimizer"]["state"].pop(next(iter(saved["optimizer"]["state"])))
        elif fault == "reference":
            expected["reference"]["schema"]["offset"]["tensor"]["dtype"] = "torch.float32"
        else:
            expected["rng"]["cpu"]["shape"] = [1]
        with pytest.raises(ValueError):
            validate_source(saved, receipt, expected)


@pytest.mark.parametrize("fault", ["unsealed", "truncated", "identity", "digest", "cap"])
def test_source_receipt_rejections_before_output(fault):
    with source_artifact() as (source, receipt):
        fd = os.memfd_create("empty-output", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            if fault == "unsealed":
                receipt = replace(receipt, seals=0)
            elif fault == "truncated":
                receipt = replace(receipt, size=receipt.size - 1)
            elif fault == "identity":
                receipt = replace(receipt, inode=receipt.inode + 1)
            elif fault == "digest":
                receipt = replace(receipt, sha256="0" * 64)
            else:
                receipt = replace(receipt, cap=1)
            with pytest.raises(ValueError):
                export(source, receipt, fd, budget(), digest(contract()))
            assert os.fstat(fd).st_size == 0
        finally:
            os.close(fd)


@pytest.mark.parametrize("fault", ["host", "source", "future", "gpu"])
def test_budget_includes_future_output_and_baselines(fault):
    changes = {
        "host": {"host_baseline": 0},
        "source": {"source_cap": 0},
        "future": {"max_bytes": 2_000_000_000},
        "gpu": {"gpu_total": 24_000_000_001},
    }
    with pytest.raises(ValueError):
        budget(**changes[fault]).check()


def test_stream_aggregate_gate_still_precedes_decode(monkeypatch):
    with source_artifact() as (fd, receipt):
        monkeypatch.setattr(
            stream, "_decode", lambda *a: pytest.fail("decoded before aggregate admission")
        )
        original = stream.SerialBudget(CAP, 1_000_000_000, 0, CAP, metadata_cap=META)
        with pytest.raises(ValueError, match="aggregate"):
            stream.read_manifest(fd, receipt, original)


@pytest.mark.parametrize("fault", ["nan", "step", "overflow"])
def test_actual_numeric_content_rejected(fault):
    with source_artifact() as (source, receipt):
        saved, _, start = stream.read_metadata(source, receipt, budget().source_limits)
        record = (
            next(iter(saved["optimizer"]["state"].values()))["step"]["tensor"]
            if fault == "step"
            else next(iter(saved["policy"]["tensors"].values()))
        )
        data = bytearray(receipt.size)
        assert os.preadv(source, [data], 0) == receipt.size
        value = {"nan": float("nan"), "step": -1.0, "overflow": torch.finfo(torch.float32).max}[
            fault
        ]
        import struct

        data[start + record.metadata["offset"] : start + record.metadata["offset"] + 4] = (
            struct.pack("<f", value)
        )
    # Retire original before creating replacement; release byte copy before
    # output acquisition. No three checkpoint payloads, including RAM buffers.
    changed = os.memfd_create("numeric-source", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    output = None
    try:
        stream._write(changed, data)
        del data
        state.fcntl.fcntl(changed, state.F_ADD_SEALS, state.SEALS)
        info = os.fstat(changed)
        altered = replace(
            receipt,
            device=info.st_dev,
            inode=info.st_ino,
            sha256=state._digest(changed, info.st_size),
        )
        output = os.memfd_create("numeric-output", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        with pytest.raises(ValueError):
            export(changed, altered, output, budget(), digest(contract()))
    finally:
        os.close(changed)
        if output is not None:
            os.close(output)
