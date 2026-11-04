"""E4 payload: serve Qwen3-8B-AWQ in one or more layouts and replay the Azure conv trace against each.

  mljob.py --pool <POOL> [--instances 2] --entry scripts/snowflake/payloads/e4_pd.py \\
      --include scripts/snowflake serving/baseline serving/pd .cache/e4 \\
      --pip vllm==0.10.2 torch==2.8.0 transformers==4.55.4 numpy==2.2.6 aiohttp==3.12.15 \\
      --args --layouts colo --tag e4-colo-1 [--calibrate] [--rounds 3] [--smoke 40] [--wall-minutes 60]

Per layout (serving.pd.layouts): start the vLLM servers (the decode server of pd-tcp on the second
instance, through a Ray actor pinned to that node), start serving.pd.proxy, send the Task 12
warmup, then (with --calibrate) three 1x runs of the calibration window and three rounds of the
evaluation window at 1x/2x/4x in the baseline's rotated order, waiting for an idle system after
each trial. "1x" is 0.25x the trace's source rate, as in docs/evidence/serving-baseline.md.

No SLO is applied here: goodput is computed afterwards by serving.pd.analyze from the raw replay
records, with the SLO calibrated on the colocated layout. Everything (replay JSON, proxy log, KV
probe summaries, server logs, environment) is tarred and put on
@CRUCIBLE_LAB.PROBE.ASSETS/e4-results/<tag>.tar; compact per-trial summaries are printed as
"RESULT <json>" lines as well.
"""

import argparse
import base64
import csv
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # repository root inside the job

from serving.baseline import vllm_baseline as vb
from serving.pd import analyze, kvprobe, layouts

MODEL_ASSET = "models/Qwen3-8B-AWQ"
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"
TRACE = ROOT / "serving/baseline/traces/azure-60s"
BASE_MULT = 0.25
REPLAY = ROOT / ".cache/e4/bin/replay"
IDS = ROOT / ".cache/e4/qwen3-8b-awq-ids.json"
RESULT_STAGE = "@CRUCIBLE_LAB.PROBE.ASSETS/e4-results"
WORK = Path("/tmp/e4")
SERVER_MODULE = "serving.pd.vllm_entry"  # tests swap in a fake server
# The job image exports OpenTelemetry settings whose collector refuses connections; keep our processes quiet.
QUIET = {"OTEL_SDK_DISABLED": "true", "OTEL_TRACES_EXPORTER": "none", "OTEL_METRICS_EXPORTER": "none",
         "OTEL_LOGS_EXPORTER": "none"}
# What the second instance needs to start a probed vLLM server and fetch the model (shipped as text).
SOURCES = ("serving/pd/__init__.py", "serving/pd/kvprobe.py", "serving/pd/vllm_entry.py", "scripts/snowflake/assets.py")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def result(obj):
    print("RESULT " + json.dumps(obj, separators=(",", ":")), flush=True)


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60,
                              check=False).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)


def node_info():
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except Exception:  # noqa: BLE001
            return None

    info = {"host": socket.gethostname(), "ip": _node_ip(),
            "gpus": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            "topo": _sh("nvidia-smi topo -m"),
            "mem_total": _sh("grep MemTotal /proc/meminfo"),
            "cgroup_mem_max": _sh("cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cpus": os.cpu_count(), "shm": _sh("df -h /dev/shm | tail -1"),
            "otel_env": sorted(k for k in os.environ if k.startswith("OTEL_")),
            "packages": {p: ver(p) for p in ("vllm", "torch", "transformers", "aiohttp", "pyzmq", "nvidia-nccl-cu12", "ray")}}
    return info


def _node_ip():
    try:
        import ray

        if ray.is_initialized():
            return ray.util.get_node_ip_address()
    except Exception:  # noqa: BLE001, S110 - no Ray here: fall back to the routing table
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read(1 << 20).decode()


# ---------------------------------------------------------------- server processes


class LocalServer:
    def __init__(self, server, argv, env, out):
        self.server, self.url = server, f"http://127.0.0.1:{server.port}"
        self.log_path = out / f"server-{server.name}.log"
        self._log = open(self.log_path, "wb")  # noqa: SIM115 - closed in stop()
        full_env = {**os.environ, **QUIET, **env, "PYTHONPATH": str(ROOT)}
        self.proc = subprocess.Popen([sys.executable, "-m", SERVER_MODULE, *argv], cwd=ROOT, env=full_env,
                                     stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)

    def poll(self):
        return self.proc.poll()

    def stop(self):
        _killpg(self.proc)
        self._log.close()
        return self.proc.returncode

    def collect(self, out):
        return None


def _killpg(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def remote_actor_class():
    import ray

    @ray.remote(num_gpus=1, num_cpus=4)
    class RemoteNode:
        """Runs one vLLM server on the second instance. The repo files it needs are shipped as text."""

        def __init__(self, sources):
            self.root = Path("/tmp/e4src")
            for rel, text in sources.items():
                p = self.root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            sys.path.insert(0, str(self.root))
            self.proc = None

        def info(self):
            return node_info()

        def fetch(self, asset, dest):
            from scripts.snowflake.assets import fetch

            return fetch(asset, dest)

        def start(self, argv, env, log_path):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "wb")  # noqa: SIM115
            full_env = {**os.environ, **env, "PYTHONPATH": str(self.root)}  # env carries QUIET
            self.proc = subprocess.Popen([sys.executable, "-m", "serving.pd.vllm_entry", *argv], cwd=self.root,
                                         env=full_env, stdout=self._log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            return self.proc.pid

        def poll(self):
            return None if self.proc is None else self.proc.poll()

        def stop(self):
            if self.proc is not None:
                _killpg(self.proc)
                self._log.close()
                return self.proc.returncode
            return None

        def files(self, pattern):
            return {os.path.basename(p): Path(p).read_bytes() for p in sorted(glob.glob(pattern))}

    return RemoteNode


class RemoteServer:
    def __init__(self, actor, server, argv, env, out, remote_ip):
        import ray

        self.ray, self.actor, self.server = ray, actor, server
        self.url = f"http://{remote_ip}:{server.port}"
        self.remote_dir = f"/tmp/e4/remote-{out.name}"
        self.log_path = out / f"server-{server.name}.log"
        ray.get(actor.start.remote(argv, env, f"{self.remote_dir}/server-{server.name}.log"), timeout=60)

    def poll(self):
        return self.ray.get(self.actor.poll.remote(), timeout=60)

    def stop(self):
        return self.ray.get(self.actor.stop.remote(), timeout=180)

    def collect(self, out):
        files = self.ray.get(self.actor.files.remote(f"{self.remote_dir}/*.log"), timeout=300)
        for name, data in files.items():
            (out / name).write_bytes(data)
        kv = self.ray.get(self.actor.files.remote(f"{self.remote_dir}/kv/*.jsonl"), timeout=600)
        (out / "kv").mkdir(exist_ok=True)
        for name, data in kv.items():
            (out / "kv" / f"remote-{name}").write_bytes(data)
        return sorted(files) + sorted(kv)


class Cluster:
    """The second instance of a two-instance job, reached through Ray."""

    def __init__(self):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
        head = ray.get_runtime_context().get_node_id()
        deadline = time.time() + 600
        while True:
            others = [n for n in ray.nodes() if n["Alive"] and n["NodeID"] != head and n["Resources"].get("GPU", 0) >= 1]
            if others or time.time() > deadline:
                break
            time.sleep(10)
        if not others:
            raise RuntimeError(f"no second GPU node in the Ray cluster: {ray.nodes()}")
        node = others[0]
        sources = {rel: (ROOT / rel).read_text() for rel in SOURCES}
        cls = remote_actor_class()
        self.actor = cls.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(
            sources)
        self.info = ray.get(self.actor.info.remote(), timeout=300)
        self.ip = self.info["ip"]

    def fetch(self, asset, dest):
        import ray

        return ray.get(self.actor.fetch.remote(asset, dest), timeout=1800)


# ---------------------------------------------------------------- one layout


def wait_ready(servers, limit_s=900):
    t0 = time.time()
    pending = list(servers)
    while pending:
        for s in list(pending):
            code = s.poll()
            if code is not None:
                raise RuntimeError(f"server {s.server.name} exited {code} before ready")
            try:
                if _get(s.url + "/health")[0] == 200:
                    pending.remove(s)
            except OSError:
                pass
        if time.time() - t0 > limit_s:
            raise RuntimeError(f"not ready in {limit_s} s: {[s.server.name for s in pending]}")
        time.sleep(3)
    return time.time() - t0


def wait_idle(servers, proxy_url, limit_s=180):
    deadline = time.time() + limit_s
    quiet = 0
    while time.time() < deadline:
        try:
            depth = sum(vb.queue_depth(_get(s.url + "/metrics")[1]) for s in servers)
            depth += json.loads(_get(proxy_url + "/stats")[1])["inflight"]
        except OSError:
            depth = -1
        quiet = quiet + 1 if depth == 0 else 0
        if quiet >= 2:
            return True
        time.sleep(1)
    return False


def _replay_bin():
    """The stage mount drops the executable bit, so run a copy."""
    dest = WORK / "bin" / "replay"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPLAY, dest)
        dest.chmod(0o755)
    return dest


def short_trace(n, dest):
    """The first n requests of the calibration and evaluation windows (smoke runs)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("calibration.csv", "evaluation.csv"):
        with open(TRACE / name, newline="") as f:
            rows = list(csv.reader(f))
        with open(dest / name, "w", newline="") as f:
            csv.writer(f, lineterminator="\r\n").writerows(rows[: n + 1])
    return dest


def run_layout(name, a, snapshot, cluster, deadline):
    out = WORK / "out" / name
    out.mkdir(parents=True, exist_ok=True)
    probe_dir = str(WORK / f"kv-{name}")
    p = layouts.plan(name, decode_pool_gb=a.decode_pool_gb, decode_buffer_bytes=a.decode_buffer_bytes,
                     shm_cuda_memcpy=not a.shm_default)
    layouts.check_ports(p)
    summary = {"layout": name, "plan": {"servers": [vars(s) for s in p.servers], "proxy_mode": p.proxy_mode,
                                        "notes": p.notes, "max_pd_inflight": a.max_pd_inflight},
               "base_mult": BASE_MULT, "levels": a.levels, "calibration": [], "evaluation": [], "stop_reason": None}
    servers, proxy = [], None
    head_ip = _node_ip()
    started = time.time()
    try:
        for s in p.servers:
            argv = layouts.server_args(s, snapshot, MODEL_NAME, p.notes)
            if s.node == 0:
                env = layouts.server_env(s, "127.0.0.1" if p.nodes == 1 else head_ip, str(WORK / "vllm-cache"),
                                         probe_dir)
                servers.append(LocalServer(s, argv, env, out))
            else:
                remote_dir = f"/tmp/e4/remote-{name}"
                env = {**QUIET, **layouts.server_env(s, cluster.ip, "/tmp/e4/vllm-cache", f"{remote_dir}/kv")}
                servers.append(RemoteServer(cluster.actor, s, argv, env, out, cluster.ip))
            summary.setdefault("server_args", {})[s.name] = argv
        summary["startup_s"] = wait_ready(servers)
        log(f"{name}: servers ready after {summary['startup_s']:.0f} s")

        proxy_cmd = [sys.executable, "-m", "serving.pd.proxy", "--listen", f"127.0.0.1:{layouts.PROXY_PORT}",
                     "--mode", p.proxy_mode, "--log", str(out / "proxy.jsonl")]
        if p.proxy_mode == "dp":
            for s in servers:
                proxy_cmd += ["--backend", s.url]
        else:
            pre, dec = servers
            pre_ip = "127.0.0.1" if p.nodes == 1 else head_ip
            dec_ip = "127.0.0.1" if p.nodes == 1 else cluster.ip
            proxy_cmd += ["--prefill", pre.url, "--prefill-kv", f"{pre_ip}:{layouts.KV_PORTS['kv_producer']}",
                          "--decode", dec.url, "--decode-kv", f"{dec_ip}:{layouts.KV_PORTS['kv_consumer']}",
                          "--max-pd-inflight", str(a.max_pd_inflight)]
        summary["proxy_command"] = proxy_cmd[1:]
        proxy_log = open(out / "proxy.log", "wb")  # noqa: SIM115
        proxy = subprocess.Popen(proxy_cmd, cwd=ROOT, env={**os.environ, **QUIET, "PYTHONPATH": str(ROOT)},
                                 stdout=proxy_log, stderr=subprocess.STDOUT, start_new_session=True)
        proxy_url = f"http://127.0.0.1:{layouts.PROXY_PORT}"
        for _ in range(60):
            try:
                _get(proxy_url + "/stats")
                break
            except OSError:
                time.sleep(1)

        rargs = SimpleNamespace(replay_bin=str(_replay_bin()), token_ids=str(IDS), model_name=MODEL_NAME)
        warm = WORK / f"warmup-{name}.csv"
        warm.write_text("Timestamp,Request tokens,Response tokens\n"
                        + "".join(f"{i},{x},{y}\n" for i, (x, y) in enumerate(vb.WARMUP)))
        code, data = vb.run_replay(rargs, warm, "burstgpt", 1.0, out / "warmup.json", layouts.PROXY_PORT)
        summary["warmup"] = {"exit": code, **vb.summarize(data, None)}
        log(f"{name}: warmup ok={summary['warmup']['ok']}/{summary['warmup']['requests']}")
        wait_idle(servers, proxy_url)
        trace_dir = short_trace(a.smoke, WORK / "smoke-trace") if a.smoke else TRACE

        def trial(role, csv_name, level, rnd):
            dead = [s.server.name for s in servers if s.poll() is not None]
            if dead or proxy.poll() is not None:
                summary["stop_reason"] = summary["stop_reason"] or f"exited: {dead or 'proxy'}"
                return None
            if time.time() > deadline:
                summary["stop_reason"] = summary["stop_reason"] or "wall deadline"
                return None
            fname = f"{role}-r{rnd}-x{level}.json"
            mult = BASE_MULT * level
            t0 = time.time()
            code, data = vb.run_replay(rargs, trace_dir / csv_name, "azure", mult, out / fname, layouts.PROXY_PORT)
            t1 = time.time()
            idle = wait_idle(servers, proxy_url)
            entry = {"file": fname, "round": rnd, "level": level, "mult": mult, "exit": code, "idle_after": idle,
                     "t_start": t0, "t_end": t1, **vb.summarize(data, None)}
            result({"layout": name, **{k: entry[k] for k in (
                "file", "sender_valid", "requests", "ok", "timeouts", "other_failures", "completed_rps",
                "output_tok_per_s", "ttft_ms", "tpot_ms")}})
            return entry

        if a.calibrate:
            for rnd in range(3):
                e = trial("calibration", "calibration.csv", 1, rnd)
                if e is None:
                    break
                summary["calibration"].append(e)
        for rnd, order in enumerate(vb.eval_order(a.rounds, tuple(a.levels))):
            for level in order:
                e = trial("evaluation", "evaluation.csv", level, rnd)
                if e is None:
                    break
                summary["evaluation"].append(e)
            if summary["stop_reason"]:
                break
    except Exception as exc:  # noqa: BLE001 - recorded as the terminal cause
        summary["stop_reason"] = summary["stop_reason"] or f"{type(exc).__name__}: {exc}"[:2000]
        log(f"{name}: {summary['stop_reason']}")
    finally:
        if proxy is not None:
            _killpg(proxy)
        summary["server_exit"] = {}
        for s in servers:
            try:
                summary["server_exit"][s.server.name] = s.stop()
            except Exception as e:  # noqa: BLE001
                summary["server_exit"][s.server.name] = repr(e)[:300]
            try:
                s.collect(out)
            except Exception as e:  # noqa: BLE001
                summary.setdefault("collect_errors", []).append(repr(e)[:300])
        summary["wall_s"] = time.time() - started
        _finish_kv(out, probe_dir, summary)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _finish_kv(out, probe_dir, summary):
    """Per-tensor probe events -> per-request rows (kv-requests.jsonl) and per-trial KV stats."""
    kvdir = out / "kv"
    kvdir.mkdir(exist_ok=True)
    for f in glob.glob(f"{probe_dir}/*.jsonl"):
        shutil.copy(f, kvdir / f"local-{os.path.basename(f)}")
    files = sorted(glob.glob(str(kvdir / "*.jsonl")))
    if not files:
        return
    rows = kvprobe.summarize(kvprobe.read_events(files))
    with open(out / "kv-requests.jsonl", "w") as f:
        f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in rows.values())
    proxy_recs = analyze.read_jsonl(out / "proxy.jsonl") if (out / "proxy.jsonl").exists() else []
    for t in summary["calibration"] + summary["evaluation"]:
        t["kv"] = analyze.kv_trial_stats(proxy_recs, rows, t["t_start"], t["t_end"])
        result({"layout": summary["layout"], "file": t["file"], "kv": t["kv"]})


def upload(tag):
    tar_path = WORK / f"{tag}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(WORK / "out", arcname=tag)
    from snowflake.snowpark import Session

    s = Session.builder.getOrCreate()
    t = time.time()
    r = s.file.put(str(tar_path), RESULT_STAGE, auto_compress=False, overwrite=True)
    return {"tar": f"{RESULT_STAGE}/{tag}.tar", "bytes": tar_path.stat().st_size, "put_s": round(time.time() - t, 1),
            "status": [str(x.status) for x in r]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", required=True, help="comma list of " + ",".join(layouts.LAYOUTS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--smoke", type=int, default=0, help="replay only the first N evaluation requests")
    ap.add_argument("--wall-minutes", type=float, default=60, help="no new trial after this")
    ap.add_argument("--decode-pool-gb", type=float, default=24)
    ap.add_argument("--decode-buffer-bytes", type=float, default=2e9)
    ap.add_argument("--max-pd-inflight", type=int, default=96)
    ap.add_argument("--shm-default", action="store_true", help="leave NCCL_SHM_USE_CUDA_MEMCPY unset")
    ap.add_argument("--snapshot", help="use this local model directory instead of fetching the asset")
    a = ap.parse_args()
    a.levels = [int(x) for x in a.levels.split(",")]
    names = a.layouts.split(",")
    deadline = time.time() + a.wall_minutes * 60
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "out").mkdir(exist_ok=True)
    env = {"head": node_info()}
    log(f"head: {json.dumps(env['head'])[:1500]}")
    cluster = None
    if any(layouts.plan(n).nodes == 2 for n in names):
        cluster = Cluster()
        env["remote"] = cluster.info
        log(f"remote: {json.dumps(cluster.info)[:1500]}")
    snapshot = a.snapshot or "/tmp/models/Qwen3-8B-AWQ"
    if not a.snapshot:
        from scripts.snowflake.assets import fetch

        env["fetch_head"] = fetch(MODEL_ASSET, snapshot)
        if cluster is not None:
            env["fetch_remote"] = cluster.fetch(MODEL_ASSET, snapshot)
        log(f"model fetched: {env['fetch_head']} {env.get('fetch_remote')}")
    (WORK / "out" / "environment.json").write_text(json.dumps(env, indent=1) + "\n")
    result({"environment": env})
    for name in names:
        s = run_layout(name, a, snapshot, cluster, deadline)
        result({"layout": name, "done": True, "stop_reason": s["stop_reason"], "startup_s": s.get("startup_s"),
                "trials": len(s["calibration"]) + len(s["evaluation"]), "wall_s": round(s["wall_s"], 1)})
    try:
        result({"upload": upload(a.tag)})
    except Exception as e:  # noqa: BLE001
        result({"upload_error": repr(e)[:1000]})
        # fall back to the log: summaries only, base64 of a small gzip
        import gzip

        for f in sorted((WORK / "out").glob("*/summary.json")):
            blob = base64.b64encode(gzip.compress(f.read_bytes())).decode()
            print(f"SUMMARY {f.parent.name} {blob}", flush=True)


if __name__ == "__main__":
    main()
