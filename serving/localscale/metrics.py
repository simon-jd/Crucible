"""Parse vLLM 0.10.2 Prometheus text; histogram quantiles over a window delta."""

import math
import re
from dataclasses import dataclass, field

LINE = re.compile(r"^(vllm:[a-z_]+)\{([^}]*)\}\s+([0-9.eE+-]+|[+-]Inf|NaN)$")


@dataclass
class Sample:
    waiting: int = 0
    running: int = 0
    success: int = 0
    ttft_buckets: dict = field(default_factory=dict)  # le (float, inf last) -> cumulative count


def parse(text):
    s = Sample()
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        name, labels, value = m[1], m[2], float(m[3])
        if name == "vllm:num_requests_waiting":
            s.waiting += int(value)
        elif name == "vllm:num_requests_running":
            s.running += int(value)
        elif name == "vllm:request_success_total":
            s.success += int(value)
        elif name == "vllm:time_to_first_token_seconds_bucket":
            le = re.search(r'le="([^"]+)"', labels)[1]
            bound = math.inf if le == "+Inf" else float(le)
            s.ttft_buckets[bound] = s.ttft_buckets.get(bound, 0.0) + value
    return s


def quantile(q, before, after):
    """histogram_quantile over (after - before) cumulative buckets; None without samples.

    Linear interpolation inside the bucket, as Prometheus does; a rank in the +Inf
    bucket returns the largest finite bound.
    """
    bounds = sorted(after)
    delta = [after[b] - before.get(b, 0.0) for b in bounds]
    total = delta[-1] if delta else 0.0
    if total <= 0:
        return None
    rank = q * total
    lower, below = 0.0, 0.0
    for bound, cumulative in zip(bounds, delta):
        if cumulative >= rank:
            if math.isinf(bound):
                return lower
            inside = cumulative - below
            return lower + (bound - lower) * ((rank - below) / inside if inside else 1.0)
        lower, below = bound, cumulative
    return lower
