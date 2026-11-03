"""CPU-only owned exporter. Entry is authorized by PhaseBridge, never a GPU task."""

import json
import os

from rl.learner import handoff_stream as stream
from rl.learner import state_handoff as state
from rl.sync.ram_weights import (
    HEADER,
    MAGIC,
    BridgeBudget,
    describe_source,
    digest,
    encode,
    validate_source,
)


def export(source_fd, receipt, output_fd, budget, expected_contract, expected_provenance):
    budget.check(receipt.size)
    if (
        type(expected_provenance) is not str
        or len(expected_provenance) != 64
        or any(c not in "0123456789abcdef" for c in expected_provenance)
    ):
        raise ValueError("explicit pinned provenance digest required")
    saved, records, start = stream.read_metadata(source_fd, receipt, budget.source_limits)
    contract = describe_source(saved)
    if digest(contract) != expected_contract:
        raise ValueError("export trusted contract digest")
    steps = validate_source(saved, receipt, contract)
    import torch

    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError("pinned Torch required")
    # No CUDA API calls, even when source describes initialized CUDA RNG.
    for index, record in enumerate(records):
        tensor = stream._allocate_record(torch, record)
        stream._read_record(source_fd, tensor, start + record["offset"])
        for chunk in tensor.reshape(-1).split(16384):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("nonfinite training content")
        if index in steps:
            step = tensor.item()
            if step < 0 or not float(step).is_integer():
                raise ValueError("Adam step value")
        if index == saved["rng"]["cpu"].index:
            torch.Generator(device="cpu").set_state(tensor)
        # CUDA RNG is only bounded metadata/raw bytes from trusted producer.
        del tensor
    output_records, offset = [], 0
    for name, spec in sorted(contract["inference"].items()):
        source = saved["policy"]["tensors"][name].metadata
        nbytes = source["bytes"] // (2 if source["dtype"] == "torch.float32" else 1)
        output_records.append({"name": name, **spec, "offset": offset, "bytes": nbytes})
        offset += nbytes
    manifest = {
        "format": "crucible-weights-v1",
        "version": receipt.version,
        "source": receipt.sha256,
        "contract": expected_contract,
        "provenance": expected_provenance,
        "records": output_records,
    }
    encoded = encode(manifest, budget.metadata_cap)
    total = HEADER.size + len(encoded) + offset
    if (
        total > budget.max_bytes
        or os.fstat(output_fd).st_size
        or state.fcntl.fcntl(output_fd, state.F_GET_SEALS)
    ):
        raise ValueError("weight output bounds/empty ownership")
    stream._write(output_fd, HEADER.pack(MAGIC, len(encoded)))
    stream._write(output_fd, encoded)
    for target in output_records:
        source = saved["policy"]["tensors"][target["name"]].metadata
        tensor = stream._allocate_record(torch, source)
        stream._read_record(source_fd, tensor, start + source["offset"])
        converted = tensor.to(dtype=getattr(torch, target["dtype"][6:]))
        for chunk in converted.reshape(-1).split(16384):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("nonfinite BF16 conversion")
        stream._write(output_fd, stream._raw(converted))
        del converted, tensor
    if os.fstat(output_fd).st_size != total:
        raise ValueError("weight output length")
    state.fcntl.fcntl(output_fd, state.F_ADD_SEALS, state.SEALS)
    receipt.validate(source_fd)
    return {"bytes": total, "version": receipt.version, "contract": expected_contract}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU exporter requires explicit empty CUDA visibility")
    source = int(os.environ["CRUCIBLE_HANDOFF_SOURCE_FD"])
    receipt = state.Receipt(**json.loads(os.environ["CRUCIBLE_HANDOFF_SOURCE_RECEIPT"]))
    result = export(
        source,
        receipt,
        int(os.environ["CRUCIBLE_HANDOFF_FD"]),
        BridgeBudget(**json.loads(os.environ["CRUCIBLE_RAM_BUDGET"])),
        os.environ["CRUCIBLE_RAM_CONTRACT"],
        os.environ["CRUCIBLE_RAM_PROVENANCE"],
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
