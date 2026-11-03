"""Dependency-free training textfile hook and inventory-aware Prometheus input."""

import json
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

from cluster.drain.controller import NodeMetrics, atomic_write, finite


def prometheus_label(value):
    """UTF-8 text labels: only backslash, quote and LF use protocol escapes.

    Reject other control characters and surrogate code points rather than emit
    ambiguous/unencodable exposition. Validation happens before any file write.
    """
    if (
        not isinstance(value, str)
        or not value
        or any(c != "\n" and unicodedata.category(c) in ("Cc", "Cs") for c in value)
    ):
        raise ValueError("labels must be nonempty UTF-8 strings without controls except LF")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


class StepTimeHook:
    """Publish each 50 completed steps to a dedicated textfile-collector directory.

    step_time_seconds is the mean wall time of the last reporting window. Call on
    one reporting rank per node, with identical run_id only for comparable loads.
    """

    def __init__(self, node, run_id, path, *, every=None, clock=time.time):
        if every is None:
            every = yaml.safe_load(Path(__file__).with_name("rules.yaml").read_text())[
                "report_every_steps"
            ]
        if not node or not run_id or type(every) is not int or every <= 0:
            raise ValueError("node, comparable run_id and positive interval required")
        self.labels = f"node={prometheus_label(node)},run_id={prometheus_label(run_id)}"
        self.node, self.run_id, self.path = node, run_id, Path(path)
        self.every, self.clock, self.last_step = every, clock, 0

    def observe(self, step, step_time_seconds):
        if (
            type(step) is not int
            or step <= 0
            or not finite(step_time_seconds)
            or step_time_seconds <= 0
        ):
            raise ValueError("positive completed step and finite positive step time required")
        if step <= self.last_step or step % self.every:
            return False
        timestamp = self.clock()
        if not finite(timestamp) or timestamp < 0:
            raise ValueError("invalid sample clock")
        values = {
            "crucible_step_time_seconds": step_time_seconds,
            "crucible_step_sample_timestamp_seconds": timestamp,
            "crucible_step_index": step,
        }
        body = "".join(
            f"# TYPE {name} gauge\n{name}{{{self.labels}}} {value}\n"
            for name, value in values.items()
        )
        atomic_write(self.path, body)
        self.last_step = step
        return True


class PrometheusSource:
    """Map expected GPU UUIDs and NIC names; missing series never imply health.

    query injection accepts Prometheus instant API JSON. HTTP is only used when
    collect() is explicitly called without an injected query function.
    """

    NAMES = (
        "DCGM_FI_DEV_XID_ERRORS",
        "node_network_carrier",
        "crucible_step_time_seconds",
        "crucible_step_sample_timestamp_seconds",
    )

    def __init__(self, inventory, *, url="http://127.0.0.1:9090", query=None):
        if not inventory or any(
            not v.get("gpus") or not v.get("links") for v in inventory.values()
        ):
            raise ValueError("explicit expected node GPU UUIDs and NICs are required")
        self.inventory, self.url = inventory, url.rstrip("/")
        self.query = query or self._http_query

    def _http_query(self, expression):
        with urlopen(
            self.url
            + "/api/v1/query?"
            + urlencode({"query": expression, "time": self._evaluation_at}),
            timeout=10,
        ) as r:
            return json.load(r)

    def _vector(self, expression):
        data = self.query(expression)
        if data.get("status") != "success" or data["data"].get("resultType") != "vector":
            raise ValueError("Prometheus did not return a successful instant vector")
        rows = []
        for row in data["data"]["result"]:
            value = float(row["value"][1])
            labels = {k: v for k, v in row["metric"].items() if k != "__name__"}
            rows.append((labels, value))
        return rows

    def collect(self):
        self._evaluation_at = time.time()  # All HTTP queries observe one evaluation instant.
        series = {}
        for name in self.NAMES:
            # Instant API result time is evaluation time, not source sample freshness.
            stamps = {
                tuple(sorted(labels.items())): value
                for labels, value in self._vector(f"timestamp({name})")
            }
            series[name] = [
                (labels, value, stamps.get(tuple(sorted(labels.items())), -1))
                for labels, value in self._vector(name)
            ]
        metrics = {}
        for node, expected in self.inventory.items():

            def matching(name, node=node):
                return [r for r in series[name] if r[0].get("node") == node]

            def devices(name, key, wanted):
                selected = [r for r in matching(name) if r[0].get(key) in wanted]
                if (
                    {r[0].get(key) for r in selected} != set(wanted)
                    or len(selected) != len(wanted)
                    or not all(finite(r[1]) and finite(r[2]) for r in selected)
                ):
                    return []
                return selected

            gpu = devices(self.NAMES[0], "UUID", expected["gpus"])
            link = devices(self.NAMES[1], "device", expected["links"])
            xid = (
                int(max(r[1] for r in gpu))
                if gpu and all(r[1].is_integer() and 0 <= r[1] < 1_000_000 for r in gpu)
                else None
            )
            up = (
                all(r[1] == 1 for r in link) if link and all(r[1] in (0, 1) for r in link) else None
            )
            observed = min((r[2] for r in gpu + link), default=-1)
            step, stamp = matching(self.NAMES[2]), matching(self.NAMES[3])
            # Multiple reporting ranks/runs on one node are ambiguous, never collapse them.
            if len(step) == len(stamp) == 1 and step[0][0] == stamp[0][0]:
                run_id = step[0][0].get("run_id")
                step_time, step_at = step[0][1], stamp[0][1]
                if not 0 <= step_at <= min(step[0][2], stamp[0][2]):
                    step_at = -1  # Future/malformed training clock must not become fresh via min().
            else:
                run_id, step_time, step_at = "", None, None
            metrics[node] = NodeMetrics(observed, xid, up, step_time, step_at, run_id)
        return metrics
