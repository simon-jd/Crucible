"""MFU estimate and read-only Nsight Systems SQLite kernel interval summary."""

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelStructure:
    """N active non-embedding parameters/token; L layers, H query heads, D head dim."""

    active_parameters: int
    layers: int
    heads: int
    head_dim: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("N, L, H, D must be explicit positive integers")


def mfu(tokens_per_s, params, seq_len, n_gpus, peak_flops) -> float:
    """Return global tokens/s * (6N + 12LHDS) / (n_gpus * peak_flops).

    S=seq_len tokens, N=active non-embedding parameters per token (MoE: only
    activated experts plus dense/shared weights), L=layers, H=query heads,
    D=head dimension. Attention follows TorchTitan's full, non-triangular
    estimate. tokens_per_s counts distinct global effective tokens, never TP/EP
    rank duplicates. peak_flops is per-GPU theoretical FLOPs/s at the measured
    precision (BF16 vs FP8 must be specified separately). Result is a fraction,
    not percent. This estimate omits memory/communication/recomputation overhead;
    it is not clamped. Parameter count alone cannot determine attention cost.
    """
    if not isinstance(params, ModelStructure):
        raise TypeError("params must specify active N, L, H and D")
    for name, value in [("seq_len", seq_len), ("n_gpus", n_gpus)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in [("tokens_per_s", tokens_per_s), ("peak_flops", peak_flops)]:
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if peak_flops == 0:
        raise ValueError("peak_flops must be positive")
    flops = (
        6 * params.active_parameters + 12 * params.layers * params.heads * params.head_dim * seq_len
    )
    result = (tokens_per_s / n_gpus) * (flops / peak_flops)
    if not math.isfinite(result):
        raise ValueError("MFU overflow")
    return result


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _length(intervals):
    return sum(end - start for start, end in intervals)


def summarize_kernels(rows):
    """Summarize same-device kernel unions; overlap does not prove hidden cost.

    rows: (device identifier, start ns, end ns, demangled name). Multiple
    processes in one exported node trace share device timelines. Never add
    separate nodes' clocks or claim a dependency critical path from durations.
    """
    devices = {}
    for device, start, end, name in rows:
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("kernel timestamps must be increasing integer nanoseconds")
        if not isinstance(name, str) or not name:
            raise ValueError("resolved kernel name required")
        d = devices.setdefault(str(device), {"compute": [], "nccl": [], "all_to_all": []})
        lower = name.lower()
        kind = "nccl" if "nccl" in lower else "compute"
        d[kind].append((start, end))
        if kind == "nccl" and ("alltoall" in lower or "all_to_all" in lower):
            d["all_to_all"].append((start, end))
    if not devices:
        raise ValueError("no CUDA kernels found; a CUDA-enabled trace is required")
    result = {}
    for device, groups in devices.items():
        comp, comm = _union(groups["compute"]), _union(groups["nccl"])
        combined = _union(comp + comm)
        overlap = _length(comp) + _length(comm) - _length(combined)
        span = combined[-1][1] - combined[0][0]
        result[device] = {
            "kernel_span_ns": span,
            "compute_union_ns": _length(comp),
            "nccl_duration_sum_ns": _length(groups["nccl"]),
            "nccl_union_ns": _length(comm),
            "nccl_compute_overlap_ns": overlap,
            "nccl_without_compute_ns": _length(comm) - overlap,
            "no_kernel_in_span_ns": span - _length(combined),
            "named_all_to_all_union_ns": _length(_union(groups["all_to_all"])),
        }
    return {
        "devices": result,
        "interpretation": (
            "Kernel intervals only: NCCL without compute is a candidate exposure signal, "
            "not a critical path or causal bottleneck. NCCL send/recv kernels may implement "
            "EP all-to-all without an all-to-all name; inspect NVTX/API/timeline manually. "
            "CPU affinity, bucket sizes and NCCL algorithm/protocol require controlled reruns."
        ),
    }


def analyze_sqlite(path):
    """Read nsys export --type sqlite output without modifying or creating a DB."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        required = {"deviceId", "start", "end", "demangledName"}
        if not required <= columns:
            raise ValueError(
                "unsupported nsys SQLite schema: require CUPTI_ACTIVITY_KIND_KERNEL "
                "deviceId/start/end/demangledName and StringIds(id,value)"
            )
        rows = db.execute(
            "SELECT k.deviceId,k.start,k.end,s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "LEFT JOIN StringIds s ON s.id=k.demangledName"
        ).fetchall()
    return summarize_kernels(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    try:
        result = analyze_sqlite(args.sqlite)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"invalid trace: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
