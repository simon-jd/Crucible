"""Every cross-node directed HCA pair, with GPUDirect and token-scoped cleanup."""

import math
import re
import time
import uuid
from itertools import permutations, product

from cluster.health.nccl_matrix import ssh, validate_inventory
from cluster.health.parse_nccl import fraction, positive
from cluster.health.remote_control import remaining, stop_remote


def build_pairs(inv, write_size=8388608, read_size=8):
    validate_inventory(inv)
    pairs = []
    for server, client in permutations(inv["nodes"], 2):
        for a, b in product(server["hcas"], client["hcas"]):
            for tool, size in [("ib_write_bw", write_size), ("ib_read_lat", read_size)]:

                def command(hca, tool=tool, size=size):
                    argv = [
                        "numactl",
                        f"--cpunodebind={hca['numa_node']}",
                        f"--membind={hca['numa_node']}",
                        tool,
                        "-d",
                        hca["name"],
                        "-i",
                        str(hca["port"]),
                        "--use_cuda",
                        str(hca["gpu_index"]),
                        "-p",
                        "18515",
                        "-s",
                        str(size),
                        "-n",
                        "1000",
                    ]
                    if tool == "ib_write_bw":
                        argv.append("--report_gbits")
                    return argv

                pairs.append(
                    {
                        "id": f"ib:{server['name']}:{a['name']}->{client['name']}:{b['name']}:{tool}",
                        "server": server["name"],
                        "client": client["name"],
                        "server_hca": a["name"],
                        "client_hca": b["name"],
                        "tool": tool,
                        "size_bytes": size,
                        "port": 18515,
                        "server_command": command(a),
                        "client_command": command(b) + [server["address"]],
                        "link_gbps": min(a.get("link_gbps") or 0, b.get("link_gbps") or 0),
                        "ideal_read_latency_us": max(
                            a.get("ideal_read_latency_us") or 0, b.get("ideal_read_latency_us") or 0
                        ),
                    }
                )
    return pairs


def parse_perftest(text, tool, size):
    """Return unidirectional decimal GB/s or median latency in microseconds."""
    header = None
    values = []
    for line in text.splitlines():
        if "#bytes" in line and "#iterations" in line:
            if header is not None:
                raise ValueError("ambiguous perftest tables")
            if tool == "ib_write_bw":
                match = re.search(r"BW\s+average\[(GB/sec|Gb/sec)\]", line)
                if not match:
                    raise ValueError("unknown perftest bandwidth units")
                header = (5, 3, 1 if match[1] == "GB/sec" else 1 / 8)
                if not re.search(r"BW\s+peak\[" + re.escape(match[1]) + r"\]", line):
                    raise ValueError("inconsistent bandwidth units")
            elif tool == "ib_read_lat":
                expected = (
                    "#bytes #iterations t_min[usec] t_max[usec] t_typical[usec] "
                    "t_avg[usec] t_stdev[usec] 99%[usec] 99.9%[usec]"
                )
                if " ".join(line.split()) != expected:
                    raise ValueError("unknown latency columns/units")
                header = (9, 4, 1)
            else:
                raise ValueError("unsupported perftest tool")
        elif re.match(r"^\s*\d+\s+\d+", line):
            fields = line.split()
            if header is None or len(fields) != header[0]:
                raise ValueError("incomplete perftest table")
            if int(fields[0]) != size or int(fields[1]) <= 0:
                raise ValueError("unexpected perftest size/iterations")
            numbers = [float(v) for v in fields[2:]]
            if any(not math.isfinite(x) or x < 0 for x in numbers):
                raise ValueError("invalid perftest metric")
            value = float(fields[header[1]]) * header[2]
            positive(value, "perftest measurement")
            values.append(value)
    if len(values) != 1:
        raise ValueError("exactly one perftest measurement required")
    return values[0]


def evaluate_ib(value, tool, link_gbps, ideal_latency_us, bw_fraction, latency_fraction):
    positive(value, "measurement")
    fraction(bw_fraction, "min_write_bw_fraction")
    fraction(latency_fraction, "min_read_latency_efficiency_fraction")
    if tool == "ib_write_bw":
        limit = positive(link_gbps, "link_gbps") / 8 * bw_fraction
        return [] if value >= limit else [f"slow write bandwidth: {value} GB/s < {limit} GB/s"]
    if tool != "ib_read_lat":
        raise ValueError("unknown tool")
    limit = positive(ideal_latency_us, "ideal_read_latency_us") / latency_fraction
    return [] if value <= limit else [f"slow read latency: {value} us > {limit} us"]


def run_pair(pair, inv, runner, timeout=180, deadline_epoch=None):
    nodes = {n["name"]: n for n in inv["nodes"]}
    helper = inv["shared_root"] + "/cluster/health/remote_worker.py"
    tokens = {"server": uuid.uuid4().hex, "client": uuid.uuid4().hex}
    end = min(time.time() + timeout, deadline_epoch if deadline_epoch is not None else float("inf"))
    base = ["python3", helper, "run"]

    def command(side):
        argv = base + ["--token", tokens[side], "--deadline-epoch", str(end)]
        if side == "server":
            argv += ["--ready-port", str(pair["port"])]
        return ssh(nodes[pair[side]], argv + ["--"] + pair[side + "_command"])

    server_argv = command("server")
    commands, cleanup, error, value = [], [], None, None
    process = None
    try:
        remaining(end)
        process = runner.start(server_argv)
        if not process.wait_ready(min(20, remaining(end))):
            raise ValueError("server did not confirm owned listening socket")
        # Readiness/setup consumes this same deadline; never reset a client budget.
        client = runner.run(command("client"), remaining(end))
        commands.append(client.as_dict())
        if client.returncode:
            raise ValueError("perftest client failed")
        server = process.finish(remaining(end))
        commands.append(server.as_dict())
        process = None
        if server.returncode:
            raise ValueError("perftest server failed")
        value = parse_perftest(client.stdout, pair["tool"], pair["size_bytes"])
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record transport failure and clean up
        error = str(exc) or type(exc).__name__
        if process is None and not commands:
            commands.append(
                {
                    "argv": server_argv,
                    "returncode": 127,
                    "stdout": "",
                    "stderr": error,
                    "timed_out": False,
                }
            )
    finally:
        for side in ["client", "server"]:
            stopped = stop_remote(inv, nodes[pair[side]], tokens[side], runner)
            cleanup.append(stopped)
            commands.append(stopped["command"])
            if not stopped["verified"]:
                error = (error or "") + f"; remote {side} cleanup unverified"
        if process is not None:
            try:
                process.terminate()
                commands.append(process.finish(2).as_dict())
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record transport failure and clean up
                error = (error or "") + "; local server transport cleanup: " + str(exc)
    return {
        "id": pair["id"],
        "passed": error is None,
        "value": value,
        "error": error,
        "commands": commands,
        "cleanup": cleanup,
    }
