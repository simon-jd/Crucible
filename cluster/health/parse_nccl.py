"""Strict nccl-tests traditional (13-column) output and fraction-of-peak checks."""

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    size_bytes: int
    algbw_gbs: float
    busbw_gbs: float
    time_us: float
    mode: str = "out-of-place"
    wrong: int | None = 0
    count: int = 0
    datatype: str = "float"
    redop: str = "sum"
    root: int = -1


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    size_bytes: int | None = None
    mode: str | None = None


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")  # noqa: TRY004
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def fraction(value, name):
    positive(value, name)
    if value > 1:
        raise ValueError(f"{name} must be a fraction in (0, 1], not percentage points")
    return value


def parse_nccl(text: str) -> list[Row]:
    rows = []
    columns = units = modes = False
    summary = False
    for raw in text.splitlines():
        # Accept srun --label output; strip only its explicit numeric rank prefix.
        line = re.sub(r"^\s*\d+:\s*", "", raw).strip()
        if line.startswith("#"):
            if "out-of-place" in line and "in-place" in line:
                modes = True
            if re.match(r"#\s+size\s+count", line):
                expected = (
                    "size count type redop root time algbw busbw #wrong time algbw busbw #wrong"
                )
                if " ".join(line[1:].split()) != expected:
                    raise ValueError("unsupported nccl-tests columns")
                columns = True
            if "(elements)" in line:
                if line[1:].split() != [
                    "(B)",
                    "(elements)",
                    "(us)",
                    "(GB/s)",
                    "(GB/s)",
                    "(us)",
                    "(GB/s)",
                    "(GB/s)",
                ]:
                    raise ValueError("expected bytes, microseconds and decimal GB/s")
                units = True
            if "Out of bounds values" in line:
                if summary:
                    raise ValueError("duplicate correctness summary")
                if not re.fullmatch(r"#\s*Out of bounds values\s*:\s*0 OK", line):
                    raise ValueError("nccl-tests correctness summary failed")
                summary = True
            if "Avg bus bandwidth" in line:
                match = re.fullmatch(r"#\s*Avg bus bandwidth\s*:\s*([\d.eE+-]+)(?:\s+OK)?", line)
                if not match:
                    raise ValueError("failed or invalid bandwidth summary")
                positive(float(match[1]), "average bandwidth")
            continue
        if not line or not re.match(r"\d", line):
            continue
        if not (columns and units and modes) or summary:
            raise ValueError("data outside nccl-tests table")
        fields = line.split()
        if len(fields) != 13:
            raise ValueError("expected full 13-column nccl-tests row")
        try:
            size, count, root = int(fields[0]), int(fields[1]), int(fields[4])
            if size <= 0 or count <= 0 or root < -1:
                raise ValueError("invalid NCCL size/count/root")
            for offset, mode in [(5, "out-of-place"), (9, "in-place")]:
                time, alg, bus = map(float, fields[offset : offset + 3])
                positive(time, "time_us")
                for value in [alg, bus]:
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid bandwidth")
                wrong = None if fields[offset + 3] == "N/A" else int(fields[offset + 3])
                if wrong is not None and wrong < 0:
                    raise ValueError("negative correctness count")
                rows.append(
                    Row(size, alg, bus, time, mode, wrong, count, fields[2], fields[3], root)
                )
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"invalid NCCL row: {line}") from exc
    if not rows or not summary or not (columns and units and modes):
        raise ValueError("missing measurements, units, table or correctness summary")
    return rows


def evaluate(rows, thresholds) -> list[Violation]:
    if thresholds.get("bandwidth_unit") != "GB/s" or thresholds.get("time_unit") != "us":
        raise ValueError("threshold units must be decimal GB/s and us")
    peak = positive(thresholds.get("theoretical_busbw_gbs"), "theoretical_busbw_gbs")
    minimum = fraction(thresholds.get("min_busbw_fraction"), "min_busbw_fraction")
    size = thresholds.get("min_size_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError("min_size_bytes must be a positive integer")
    if not rows or not any(r.size_bytes >= size for r in rows):
        raise ValueError("no measurements in configured size range")
    violations = []
    for r in rows:
        positive(r.time_us, "time_us")
        if not math.isfinite(r.busbw_gbs) or r.busbw_gbs < 0:
            raise ValueError("invalid measured bandwidth")
        if r.wrong is None:
            violations.append(
                Violation("unchecked", "correctness validation disabled", r.size_bytes, r.mode)
            )
        elif r.wrong != 0:
            violations.append(
                Violation("correctness", f"{r.wrong} incorrect results", r.size_bytes, r.mode)
            )
        if r.size_bytes >= size and r.busbw_gbs < peak * minimum:
            violations.append(
                Violation(
                    "slow_bandwidth",
                    f"{r.busbw_gbs} GB/s < {minimum} * {peak} GB/s",
                    r.size_bytes,
                    r.mode,
                )
            )
    return violations
