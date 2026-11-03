"""Step-burst schedule: real Azure request lengths, seeded Poisson arrivals per phase."""

import csv
import datetime
import random

EPOCH = datetime.datetime(2023, 11, 16, 18, 0, 0)


def step_burst(lengths, phases, base_rps, seed):
    """Rows (arrival_s, in_tokens, out_tokens). phases = [(seconds, rate multiple)].

    Lengths are reused in source order, cycling, so every condition sees the same
    sequence. Arrivals are a seeded Poisson process at base_rps * multiple.
    """
    rng = random.Random(seed)
    # The replayer rebases arrivals to the first row, so the first request is at 0
    # and phase edges in source time equal phase edges in replay time.
    rows, start, i = [(0.0, *lengths[0])], 0.0, 1
    for seconds, multiple in phases:
        t = start
        while True:
            t += rng.expovariate(base_rps * multiple)
            if t >= start + seconds:
                break
            rows.append((t, *lengths[i % len(lengths)]))
            i += 1
        start += seconds
    return rows


def phase_of(scheduled_ms, phases):
    edge = 0.0
    for index, (seconds, _) in enumerate(phases):
        edge += seconds * 1000
        if scheduled_ms < edge:
            return index
    return len(phases) - 1


def write_azure_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        for t, n_in, n_out in rows:
            stamp = EPOCH + datetime.timedelta(seconds=t)
            w.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S.%f") + "0", n_in, n_out])


def read_lengths(path):
    with open(path, newline="") as f:
        return [(int(r["ContextTokens"]), int(r["GeneratedTokens"])) for r in csv.DictReader(f)]
