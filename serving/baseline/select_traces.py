"""Arrival-only selection of two disjoint windows from bounded real trace prefixes."""

import argparse
import bisect
import csv
import datetime
import hashlib
import json
from decimal import Decimal
from pathlib import Path

MAX_PREFIX_BYTES = 4 << 20


def _timestamp(value, kind):
    if kind == "burstgpt":
        result = Decimal(value)
    else:
        whole, _, fraction = value.partition(".")
        stamp = datetime.datetime.strptime(whole, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.UTC)
        result = Decimal(int(stamp.timestamp())) + Decimal("0." + (fraction or "0"))
    if not result.is_finite() or result < 0:
        raise ValueError("invalid source timestamp")
    return result


def select_windows(raw, kind, duration, min_rows=1):
    """Keep every complete CSV row inside each [start,end) window, including zeros.

    Azure uses its first two adjacent windows. BurstGPT chooses the earliest
    arrival-anchored qualifying window, then the earliest disjoint qualifying one.
    Lengths/outcomes never influence selection. Original selected bytes survive.
    """
    if kind not in {"azure", "burstgpt"} or not 0 < duration <= 600 or min_rows < 1:
        raise ValueError("invalid selection configuration")
    if len(raw) > MAX_PREFIX_BYTES:
        raise ValueError("prefix exceeds bounded bytes")
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    if not lines:
        raise ValueError("empty prefix")
    header = next(csv.reader([lines[0].decode()]))
    keys = (
        ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
        if kind == "azure"
        else ("Timestamp", "Request tokens", "Response tokens")
    )
    if len(set(header)) != len(header) or any(k not in header for k in keys):
        raise ValueError("invalid source header")
    indexes = [header.index(k) for k in keys]
    rows = []
    for number, line in enumerate(lines[1:], 1):
        values = next(csv.reader([line.decode()]))
        if len(values) != len(header):
            raise ValueError("malformed source row")
        stamp = _timestamp(values[indexes[0]], kind)
        inputs, outputs = (int(values[i]) for i in indexes[1:])
        if inputs < 0 or outputs < 0:
            raise ValueError("negative source token length")
        if rows and stamp < rows[-1][0]:
            raise ValueError("source is not sorted")
        rows.append((stamp, inputs, outputs, number, line))
    if not rows:
        raise ValueError("need two disjoint windows")
    stamps = [r[0] for r in rows]
    duration = Decimal(duration)
    starts = []
    if kind == "azure":
        starts = [stamps[0], stamps[0] + duration]
    else:
        index = 0
        while index < len(rows) and len(starts) < 2:
            end = bisect.bisect_left(stamps, stamps[index] + duration)
            if end - index >= min_rows:
                starts.append(stamps[index])
                index = end
            else:
                index += 1
    windows = []
    for start in starts:
        selected = rows[
            bisect.bisect_left(stamps, start) : bisect.bisect_left(stamps, start + duration)
        ]
        if not selected:
            continue
        payload = lines[0] + b"".join(r[4] for r in selected)
        windows.append(
            {
                "start": str(start),
                "end_exclusive": str(start + duration),
                "source_rows": [r[3] for r in selected],
                "rows": len(selected),
                "first_source_timestamp": str(selected[0][0]),
                "last_source_timestamp": str(selected[-1][0]),
                "leading_idle_seconds": str(selected[0][0] - start),
                "trailing_idle_seconds": str(start + duration - selected[-1][0]),
                "replay_schedule_span_seconds": str(selected[-1][0] - selected[0][0]),
                "unique_input_lengths": len({r[1] for r in selected}),
                "zero_input": sum(r[1] == 0 for r in selected),
                "zero_output": sum(r[2] == 0 for r in selected),
                "over_context_4096": sum(r[1] + r[2] > 4096 for r in selected),
                "max_input": max(r[1] for r in selected),
                "max_output": max(r[2] for r in selected),
                "selection_sha256": hashlib.sha256(payload).hexdigest(),
                "csv_bytes": payload,
            }
        )
    if len(windows) != 2:
        raise ValueError("need two disjoint windows within bounded prefix")
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--format", choices=("azure", "burstgpt"), required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.prefix.open("rb") as source:
        raw = source.read(MAX_PREFIX_BYTES + 1)
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("source prefix hash mismatch")
    windows = select_windows(raw, args.format, args.duration, args.min_rows)
    args.output.mkdir(parents=True, exist_ok=False)
    for role, window in zip(("calibration", "evaluation"), windows, strict=True):
        (args.output / f"{role}.csv").write_bytes(window.pop("csv_bytes"))
    (args.output / "selection.json").write_text(json.dumps(windows, indent=2) + "\n")


if __name__ == "__main__":
    main()
