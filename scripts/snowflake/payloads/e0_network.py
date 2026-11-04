"""E0 payload: network and communication baseline for the Crucible Snowflake experiments.

  --mode two-node   2 Snowflake GPU_NV_SM nodes (1x A10 24 GB each) over TCP, one Ray actor per node
                    (STRICT_SPREAD): raw TCP throughput (1, 8 and 32 streams, each direction, plus 8 and 32
                    streams both ways at once), then NCCL all-reduce and broadcast over NET/Socket for several
                    NCCL_SOCKET_NTHREADS x NCCL_NSOCKS_PERTHREAD settings, in interleaved passes.
  --mode one-node   Snowflake GPU_NV_2M (2x A10 24 GB vGPU in one node, no NVLink/P2P): plain GPU0<->GPU1 and
                    pinned host<->GPU copies, then NCCL all-reduce and broadcast (SHM transport) for the default,
                    NCCL_SHM_USE_CUDA_MEMCPY=1 and that plus NCCL_SHM_MEMCPY_MODE=3, in interleaved passes.
  --mode worker     one rank of the one-node NCCL run (started by torchrun from --mode one-node).
  --quick           small sizes, two variants, one pass: the smoke test.

Every NCCL variant runs in fresh processes, because NCCL reads its environment once per process. Bandwidth is
bytes / seconds / 1e9 (GB/s, decimal); sizes are binary (1 MiB .. 1 GiB). For 2 ranks the all-reduce bus
bandwidth equals the algorithm bandwidth (factor 2(n-1)/n = 1), and a ring all-reduce of S bytes sends S bytes
in each direction at the same time, so its ceiling is the bidirectional link rate. Prints one "RESULT <json>".
Cross-node ports stay in E0's band 12900-12919.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root inside the job

SIZES = (1 << 20, 16 << 20, 256 << 20, 1 << 30)
QUICK_SIZES = (1 << 20, 16 << 20)
# (warm-up, timed) iterations per size. Cross-node 1 GiB at ~0.2 GB/s is ~5 s per call, so few timed calls.
ITERS_NET = {1 << 20: (5, 50), 16 << 20: (3, 20), 256 << 20: (1, 5), 1 << 30: (1, 3)}
ITERS_SHM = {1 << 20: (5, 50), 16 << 20: (3, 20), 256 << 20: (2, 10), 1 << 30: (2, 5)}
TCP_PORT = 12900
# NCCL 4x4 opens 16 sockets per channel, so raw TCP is also measured with more than 8 streams.
TCP_STREAMS = (1, 8, 32)
TCP_BIDIR_STREAMS = (8, 32)
NCCL_PORTS = tuple(range(12901, 12920))
CHUNK = 4 << 20


def socket_env(threads, socks, **extra):
    return {"NCCL_SOCKET_NTHREADS": str(threads), "NCCL_NSOCKS_PERTHREAD": str(socks), **extra}


SOCKET_VARIANTS = [
    ("1x1", socket_env(1, 1)),
    ("2x2", socket_env(2, 2)),
    ("4x4", socket_env(4, 4)),
    ("8x8", socket_env(8, 8)),
    ("4x4_buff16m", socket_env(4, 4, NCCL_BUFFSIZE="16777216")),
]
# Follow-up (--variants): the brief's grid peaked at 8x8, so look past it. NCCL caps threads x sockets at 64 per
# connection and NCCL_SOCKET_NTHREADS at 16; more channels give more connections, hence more TCP flows.
EXTRA_VARIANTS = [
    ("16x4", socket_env(16, 4)),
    ("8x8_buff16m", socket_env(8, 8, NCCL_BUFFSIZE="16777216")),
    ("8x8_ch4", socket_env(8, 8, NCCL_MIN_NCHANNELS="4")),
    ("8x8_ch4_buff16m", socket_env(8, 8, NCCL_MIN_NCHANNELS="4", NCCL_BUFFSIZE="16777216")),
]
ALL_VARIANTS = dict([("default", {})] + SOCKET_VARIANTS + EXTRA_VARIANTS)
SHM_VARIANTS = [
    ("shm_default", {}),
    ("shm_cudamemcpy", {"NCCL_SHM_USE_CUDA_MEMCPY": "1"}),
    ("shm_cudamemcpy_mode3", {"NCCL_SHM_USE_CUDA_MEMCPY": "1", "NCCL_SHM_MEMCPY_MODE": "3"}),
]
NCCL_LOG_KEYS = ("NCCL version", " via ", "NET/Socket", "Using network", "threads and", "comm 0x")


# ---------------------------------------------------------------- raw TCP (pure sockets, testable on loopback)

def split_bytes(total, streams):
    """Split total into streams parts that differ by at most one CHUNK-aligned remainder."""
    base = total // streams
    return [base + (total - base * streams if i == streams - 1 else 0) for i in range(streams)]


def listen(port, backlog=256):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(backlog)
    return srv


def recv_streams(srv, streams, total, timeout=300):
    """Accept `streams` connections on srv and read until `total` bytes arrived in all; time first to last byte."""
    srv.settimeout(timeout)
    conns = [srv.accept()[0] for _ in range(streams)]
    got = [0] * streams
    first, last = [None] * streams, [None] * streams

    def drain(i, conn):
        buf = bytearray(CHUNK)
        conn.settimeout(timeout)
        while True:
            n = conn.recv_into(buf)
            if not n:
                break
            now = time.perf_counter()
            if first[i] is None:
                first[i] = now
            got[i] += n
            last[i] = now
        conn.close()

    threads = [threading.Thread(target=drain, args=(i, c)) for i, c in enumerate(conns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    srv.close()
    t0 = min(f for f in first if f is not None)
    dt = max(x for x in last if x is not None) - t0
    nbytes = sum(got)
    return {"streams": streams, "bytes": nbytes, "seconds": round(dt, 3), "GBps": round(nbytes / dt / 1e9, 3),
            "complete": nbytes == total,
            "per_stream_GBps": [round(g / (b - a) / 1e9, 3) for g, a, b in zip(got, first, last) if a and b > a]}


def send_streams(host, port, streams, total, connect_seconds=60):
    """Open `streams` connections to host:port and send total bytes split across them, one thread each."""
    parts = split_bytes(total, streams)
    payload = memoryview(bytes(CHUNK))
    errors = []

    def push(nbytes):
        deadline = time.time() + connect_seconds
        while True:
            try:
                s = socket.create_connection((host, port), timeout=10)
                break
            except OSError:
                if time.time() > deadline:
                    errors.append("connect timed out")
                    return
                time.sleep(0.1)
        try:
            sent = 0
            while sent < nbytes:
                n = min(CHUNK, nbytes - sent)
                s.sendall(payload[:n])
                sent += n
        except OSError as e:
            errors.append(repr(e))
        finally:
            s.close()

    threads = [threading.Thread(target=push, args=(p,)) for p in parts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"sent": total if not errors else None, "errors": errors}


def loopback(streams, total, port=0):
    srv = listen(port)
    port = srv.getsockname()[1]
    res = {}
    t = threading.Thread(target=lambda: res.update(recv_streams(srv, streams, total)))
    t.start()
    send_streams("127.0.0.1", port, streams, total)
    t.join()
    return res


# ---------------------------------------------------------------- stats

def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    med = statistics.median(vals)
    return {"n": len(vals), "median": round(med, 3), "min": round(min(vals), 3), "max": round(max(vals), 3),
            "spread_pct": round((max(vals) - min(vals)) / med * 100, 1) if med else None}


def summarize_runs(runs):
    """runs: [{"variant", "pass", "ranks": [{"bw": [{"op","bytes","GBps"}]}]}] -> {variant: {op: {bytes: stats}}}.
    Per run the slower rank counts (both ranks must finish an op for it to be done)."""
    acc = {}
    for run in runs:
        ranks = run.get("ranks") or []
        if len(ranks) < 2:
            continue
        for rows in zip(*[r["bw"] for r in ranks]):
            gbps = min(x["GBps"] for x in rows)
            acc.setdefault(run["variant"], {}).setdefault(rows[0]["op"], {}).setdefault(str(rows[0]["bytes"]), []).append(gbps)
    return {v: {op: {b: summarize(xs) for b, xs in by.items()} for op, by in ops.items()} for v, ops in acc.items()}


# ---------------------------------------------------------------- collectives (shared by both modes)

def run_collectives(rank, sizes, iters, device="cuda"):
    import torch
    import torch.distributed as dist

    def sync():
        if device != "cpu":
            torch.cuda.synchronize()

    out = []
    for op in ("all_reduce", "broadcast"):
        for nbytes in sizes:
            warm, n = iters[nbytes]
            x = torch.zeros(nbytes // 4, device=device)

            def call(x=x, op=op):
                if op == "all_reduce":
                    dist.all_reduce(x)
                else:
                    dist.broadcast(x, src=0)

            for _ in range(warm):
                call()
            sync()
            dist.barrier()
            t = time.perf_counter()
            for _ in range(n):
                call()
            sync()
            dt = (time.perf_counter() - t) / n
            out.append({"op": op, "bytes": nbytes, "iters": n, "ms": round(dt * 1e3, 3),
                        "GBps": round(nbytes / dt / 1e9, 3)})
            del x
    x = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(x)
    y = torch.full((4,), float(rank + 7), device=device)
    dist.broadcast(y, src=0)
    world = dist.get_world_size()
    ok = x.tolist() == [float(world * (world + 1) // 2)] * 4 and y.tolist() == [7.0] * 4
    return out, ok


def nccl_log_lines(path_pattern, pid=None):
    path = path_pattern.replace("%h", socket.gethostname()).replace("%p", str(pid or os.getpid()))
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as e:
        return [repr(e)]
    keep = [line.split("NCCL INFO ", 1)[-1] for line in lines if any(k in line for k in NCCL_LOG_KEYS)]
    return list(dict.fromkeys(keep))[:16]  # drop repeats (one line per channel), keep order


def versions():
    import importlib.metadata as md
    out = {}
    for p in ("torch", "numpy", "ray", "nvidia-nccl-cu12"):
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


def sh(*cmd, timeout=30):
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
        return (r.stdout.strip() or r.stderr.strip())[:3000]
    except Exception as e:  # noqa: BLE001 - record any failure and go on
        return repr(e)


def read(path):
    try:
        return open(path).read().strip()
    except OSError as e:
        return repr(e)


def node_facts():
    return {"host": socket.gethostname(), "cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
            "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"), "mtu_eth0": read("/sys/class/net/eth0/mtu"),
            "speed_eth0": read("/sys/class/net/eth0/speed"),
            "port_range": read("/proc/sys/net/ipv4/ip_local_port_range"),
            "tcp_cc": read("/proc/sys/net/ipv4/tcp_congestion_control"),
            "tcp_rmem": read("/proc/sys/net/ipv4/tcp_rmem"), "tcp_wmem": read("/proc/sys/net/ipv4/tcp_wmem"),
            "rmem_max": read("/proc/sys/net/core/rmem_max"), "wmem_max": read("/proc/sys/net/core/wmem_max"),
            "gpus": sh("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader")}


# ---------------------------------------------------------------- two nodes

def two_node(a):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    @ray.remote(num_gpus=1, num_cpus=1, max_concurrency=4)
    class Worker:
        def info(self):
            import torch
            return {"ip": ray.util.get_node_ip_address(), "gpu": torch.cuda.get_device_name(0),
                    "nccl": ".".join(map(str, torch.cuda.nccl.version())), "cuda": torch.version.cuda,
                    "versions": versions(), **node_facts()}

        def listen(self, port):
            self.srv = listen(port)
            return True

        def recv(self, streams, total):
            return recv_streams(self.srv, streams, total)

        def send(self, host, port, streams, total):
            return send_streams(host, port, streams, total)

        def loopback(self, streams, total):
            return loopback(streams, total)

        def nccl(self, rank, master, port, sizes):
            import datetime

            import torch
            import torch.distributed as dist
            torch.cuda.set_device(0)
            t0 = time.perf_counter()
            dist.init_process_group("nccl", init_method=f"tcp://{master}:{port}", rank=rank, world_size=2,
                                    timeout=datetime.timedelta(minutes=8), device_id=torch.device("cuda", 0))
            bw, ok = run_collectives(rank, sizes, ITERS_NET)
            dist.destroy_process_group()
            return {"rank": rank, "bw": bw, "check_ok": ok, "seconds": round(time.perf_counter() - t0, 1),
                    "nccl_log": nccl_log_lines(os.environ.get("NCCL_DEBUG_FILE", ""))}

    def spread(env):
        pg = placement_group([{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}], strategy="STRICT_SPREAD")
        ray.get(pg.ready(), timeout=300)
        ws = [Worker.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i), runtime_env={"env_vars": env}).remote()
            for i in range(2)]
        return pg, ws

    def teardown(pg, ws):
        for w in ws:
            ray.kill(w)
        remove_placement_group(pg)

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    out = {"mode": "two-node", "quick": a.quick,
           "ray_nodes": [{"ip": n["NodeManagerAddress"], "alive": n["Alive"], "gpus": n["Resources"].get("GPU", 0)}
                         for n in ray.nodes()]}
    t_start = time.time()
    pg, ws = spread({})
    info = ray.get([w.info.remote() for w in ws], timeout=300)
    out["workers"] = info
    out["distinct_nodes"] = len({i["ip"] for i in info}) == 2
    ips = [i["ip"] for i in info]
    progress(f"workers {ips} nccl {[i['nccl'] for i in info]}")

    total = (256 << 20) if a.quick else (2 << 30)
    out["tcp_loopback_node_a"] = [ray.get(ws[0].loopback.remote(s, total), timeout=300) for s in sorted({1, 8, max(a.tcp_streams)})]
    tcp = []
    for rep in range(1 if a.quick else a.tcp_repeats):
        n0 = len(tcp)
        for streams in a.tcp_streams:
            for name, src, dst in (("a_to_b", 0, 1), ("b_to_a", 1, 0)):
                try:
                    ray.get(ws[dst].listen.remote(TCP_PORT), timeout=60)
                    recv = ws[dst].recv.remote(streams, total)
                    send = ws[src].send.remote(ips[dst], TCP_PORT, streams, total)
                    r, s = ray.get([recv, send], timeout=600)
                    tcp.append({"test": name, "rep": rep, **r, "send_errors": s["errors"]})
                except Exception as e:  # noqa: BLE001 - record any failure and go on
                    tcp.append({"test": name, "rep": rep, "streams": streams, "error": repr(e)[:500]})
        for streams in a.bidir_streams:  # both ways at once: the ceiling a 2-rank ring all-reduce sees
            try:
                ray.get([w.listen.remote(TCP_PORT) for w in ws], timeout=60)
                recvs = [w.recv.remote(streams, total) for w in ws]
                sends = [ws[0].send.remote(ips[1], TCP_PORT, streams, total),
                         ws[1].send.remote(ips[0], TCP_PORT, streams, total)]
                rb = ray.get(recvs + sends, timeout=900)
                tcp.append({"test": "bidir_into_a", "rep": rep, **rb[0]})
                tcp.append({"test": "bidir_into_b", "rep": rep, **rb[1]})
            except Exception as e:  # noqa: BLE001 - record any failure and go on
                tcp.append({"test": "bidir", "rep": rep, "streams": streams, "error": repr(e)[:500]})
        progress(f"tcp rep {rep}: " + ", ".join(f"{t['test']}/{t.get('streams')}={t.get('GBps')}" for t in tcp[n0:]))
    out["tcp"] = tcp
    teardown(pg, ws)

    sizes = QUICK_SIZES if a.quick else SIZES
    if a.variants is not None:
        variants, first = [(v, ALL_VARIANTS[v]) for v in a.variants], []
    else:
        variants = [SOCKET_VARIANTS[0], SOCKET_VARIANTS[2]] if a.quick else SOCKET_VARIANTS
        first = [] if a.quick else [("default", {})]
    runs, k = [], 0
    for p in range(1 if a.quick else a.passes):
        plan = (first if p == 0 else []) + variants
        for name, extra in plan:
            env = {"NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "INFO", "NCCL_DEBUG_FILE": "/tmp/nccl.%h.%p.log", **extra}
            port = NCCL_PORTS[k % len(NCCL_PORTS)]
            k += 1
            t0 = time.time()
            pg, ws = spread(env)
            try:
                ip0 = ray.get(ws[0].info.remote(), timeout=300)["ip"]
                ranks = ray.get([w.nccl.remote(r, ip0, port, sizes) for r, w in enumerate(ws)], timeout=1200)
                runs.append({"variant": name, "pass": p, "env": extra, "port": port, "ranks": ranks})
            except Exception as e:  # noqa: BLE001 - record any failure and go on
                runs.append({"variant": name, "pass": p, "env": extra, "port": port, "error": repr(e)[:1500]})
            teardown(pg, ws)
            last = runs[-1]
            brief = {f"{b['op']}@{b['bytes'] >> 20}M": b["GBps"] for b in last["ranks"][0]["bw"]} if "ranks" in last \
                else last["error"][:200]
            progress(f"pass {p} {name}: {round(time.time() - t0)} s {brief}")
    out["nccl_runs"] = runs
    out["nccl_summary"] = summarize_runs(runs)
    out["tcp_summary"] = {f"{t['test']}/{t['streams']}": summarize([x.get("GBps") for x in tcp
                                                                     if x["test"] == t["test"]
                                                                     and x.get("streams") == t["streams"]])
                          for t in tcp if "GBps" in t}
    out["seconds"] = round(time.time() - t_start, 1)
    print("RESULT " + json.dumps(out), flush=True)


# ---------------------------------------------------------------- one node, two GPUs

def copy_bench(a):
    import torch
    out = []
    reps = 1 if a.quick else 3

    def timed(dst, src, n=5):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        t = time.perf_counter()
        for _ in range(n):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        return (time.perf_counter() - t) / n

    for nbytes in ((16 << 20, 256 << 20) if a.quick else (16 << 20, 256 << 20, 1 << 30)):
        bufs = {"gpu0": torch.empty(nbytes, dtype=torch.uint8, device="cuda:0"),
                "gpu1": torch.empty(nbytes, dtype=torch.uint8, device="cuda:1"),
                "host_pinned": torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)}
        for src, dst in (("gpu0", "gpu1"), ("gpu1", "gpu0"), ("host_pinned", "gpu0"), ("gpu0", "host_pinned")):
            for rep in range(reps):
                dt = timed(bufs[dst], bufs[src])
                out.append({"copy": f"{src}->{dst}", "bytes": nbytes, "rep": rep, "ms": round(dt * 1e3, 3),
                            "GBps": round(nbytes / dt / 1e9, 2)})
        del bufs
        torch.cuda.empty_cache()
    return out


def one_node(a):
    import torch
    t_start = time.time()
    n = torch.cuda.device_count()
    out = {"mode": "one-node", "quick": a.quick, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
           "cuda": torch.version.cuda, "versions": versions(), "node": node_facts(), "device_count": n,
           "topo": sh("nvidia-smi", "topo", "-m"),
           "peer_access": {f"{i}->{j}": torch.cuda.can_device_access_peer(i, j)
                           for i in range(n) for j in range(n) if i != j}}
    if n < 2:
        out["error"] = "fewer than 2 GPUs"
        print("RESULT " + json.dumps(out), flush=True)
        return
    copies = copy_bench(a)
    out["copies"] = copies
    out["copy_summary"] = {f"{c['copy']}@{c['bytes'] >> 20}M": summarize([x["GBps"] for x in copies
                                                                         if x["copy"] == c["copy"]
                                                                         and x["bytes"] == c["bytes"]])
                           for c in copies}
    progress("copies " + json.dumps(out["copy_summary"]))
    runs = []
    variants = SHM_VARIANTS[:2] if a.quick else SHM_VARIANTS
    for p in range(1 if a.quick else a.passes):
        for name, extra in variants:
            t0 = time.time()
            res_dir = tempfile.mkdtemp(prefix="e0_shm_")
            env = {**os.environ, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_FILE": os.path.join(res_dir, "nccl.%h.%p.log"),
                   **extra}
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(Path(__file__).resolve()), "--mode", "worker", "--result-dir", res_dir] + \
                (["--quick"] if a.quick else [])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, check=False)
            ranks = []
            for i in range(2):
                try:
                    ranks.append(json.loads(Path(res_dir, f"rank{i}.json").read_text()))
                except OSError:
                    pass
            run = {"variant": name, "pass": p, "env": extra, "ranks": ranks}
            if r.returncode != 0 or len(ranks) != 2:
                run["error"] = f"torchrun exit {r.returncode}: " + "\n".join((r.stdout + r.stderr).splitlines()[-30:])
            runs.append(run)
            brief = {f"{b['op']}@{b['bytes'] >> 20}M": b["GBps"] for b in ranks[0]["bw"]} if ranks else run.get("error")
            progress(f"pass {p} {name}: {round(time.time() - t0)} s {brief}")
    out["nccl_runs"] = runs
    out["nccl_summary"] = summarize_runs(runs)
    out["seconds"] = round(time.time() - t_start, 1)
    print("RESULT " + json.dumps(out), flush=True)


def worker(a):
    import datetime

    import torch
    import torch.distributed as dist
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5), device_id=torch.device("cuda", local))
    rank = dist.get_rank()
    bw, ok = run_collectives(rank, QUICK_SIZES if a.quick else SIZES, ITERS_SHM)
    dist.destroy_process_group()
    res = {"rank": rank, "bw": bw, "check_ok": ok, "nccl_log": nccl_log_lines(os.environ["NCCL_DEBUG_FILE"])}
    with open(os.path.join(a.result_dir, f"rank{rank}.json"), "w") as f:
        json.dump(res, f)


def progress(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("two-node", "one-node", "worker"), required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tcp-repeats", type=int, default=3)
    ap.add_argument("--tcp-streams", type=int, nargs="*", default=list(TCP_STREAMS))
    ap.add_argument("--bidir-streams", type=int, nargs="*", default=list(TCP_BIDIR_STREAMS))
    ap.add_argument("--variants", nargs="*", choices=sorted(ALL_VARIANTS), default=None,
                    help="two-node NCCL variants to run instead of the default grid (no extra 'default' run)")
    ap.add_argument("--result-dir")
    a = ap.parse_args(argv)
    {"two-node": two_node, "one-node": one_node, "worker": worker}[a.mode](a)


if __name__ == "__main__":
    main()
