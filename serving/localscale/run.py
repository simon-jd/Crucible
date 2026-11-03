"""Task 14 local experiment: real vLLM replicas behind the Go router and autoscaler.

Must run under scripts/with_gpu.sh (see scripts/run_localscale_gpu.sh).

`--mode scale`: matched static-vs-autoscaled trials of one step-burst schedule
(Azure request lengths, seeded Poisson arrivals, phases such as 1x/4x/1x). Replica
`a` serves every trial. The Go autoscaler runs dry-run in both conditions and only
proposes; in `auto` trials this driver acts on its proposals by spawning replica `b`
(scale up) or draining and stopping it (scale down), and reports the new desired
count back through the adapter so the policy observes the change. In `static`
trials the proposals are only logged. Two replicas share the one GPU, so a replica
adds a batch of `--max-num-seqs` slots, not GPU compute.

`--mode cold`: alternating warm/cold start-to-ready of one replica. "cold" evicts the
snapshot's files from the page cache with posix_fadvise(DONTNEED) first (no root).
"""

import argparse
import datetime
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from serving.baseline.vllm_baseline import Monitor
from serving.localscale import adapter, analyze, trace

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORTS = {"a": 31101, "b": 31102}
ROUTER_PORT, ADAPTER_PORT = 31180, 31190
TENANT = "bench"
REQUEST_TIMEOUT_S = 60


def alternate(conditions, pairs):
    """ABAB... with the order reversed on every other pair (AB, BA, AB, ...)."""
    order = []
    for pair in range(pairs):
        order += list(conditions) if pair % 2 == 0 else list(conditions)[::-1]
    return order


def kills(condition):
    return condition in ("kill", "kill-retry", "kill-client-retry")


def router_flags(condition):
    """Fault conditions: kill (eject on transport error), kill-retry (also resend dial errors),
    kill-client-retry (kill-retry plus a client resend of cut streams, see replay_flags)."""
    return ["-retry-dial"] if condition in ("kill-retry", "kill-client-retry") else []


def replay_flags(condition):
    """The replayer resends a cut stream once; the resend is accounted as retried."""
    return ["-client-retries", "1"] if condition == "kill-client-retry" else []


def router_log_stats(text):
    """Ejections and dial retries from the router's stderr (see serving/router/cmd/router)."""
    out = {"ejections": None, "dial_retries": None, "eject_lines": 0, "first_eject_t": None}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[1] == "eject":
            out["eject_lines"] += 1
            if out["first_eject_t"] is None:
                stamp = datetime.datetime.fromisoformat(parts[0])
                out["first_eject_t"] = stamp.timestamp()
        elif line.startswith("router stats "):
            fields = dict(f.split("=") for f in parts[2:])
            out["ejections"], out["dial_retries"] = int(fields["ejections"]), int(fields["dial_retries"])
    return out


def now():
    return time.time()


class Replica:
    def __init__(self, name, args, log_dir):
        self.name, self.args, self.port = name, args, PORTS[name]
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = Path(log_dir) / f"server-{name}.log"
        self.proc = None

    def command(self):
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.args.snapshot, "--served-model-name", MODEL,
            "--host", "127.0.0.1", "--port", str(self.port),
            "--gpu-memory-utilization", str(self.args.gpu_util), "--seed", "17",
            "--max-model-len", "8192", "--max-num-seqs", str(self.args.max_num_seqs),
            "--no-enable-prefix-caching", "--generation-config", "vllm", "--enforce-eager",
        ]  # fmt: skip

    def spawn(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "ab") as out:  # the child keeps its own descriptor
            self.proc = subprocess.Popen(self.command(), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, limit_s=300):
        """Ready = /health 200 and one fresh completion succeeds."""
        deadline = now() + limit_s
        while now() < deadline:
            if not self.alive():
                raise RuntimeError(f"replica {self.name} exited with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    healthy = r.status == 200
            except OSError:
                healthy = False
            if healthy:
                body = json.dumps({"model": MODEL, "prompt": " the model", "max_tokens": 2, "temperature": 0})
                req = urllib.request.Request(self.url + "/v1/completions", body.encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    if r.status == 200:
                        return now()
            time.sleep(0.2)
        raise TimeoutError(f"replica {self.name} not ready in {limit_s}s")

    def stop(self):
        if not self.alive():
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(30)


def start_process(cmd, log):
    with open(log, "ab") as out:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=out, start_new_session=True, text=True)


def stop_process(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(10)


def parse_phases(text):
    return [(float(s), float(m)) for s, m in (p.split(":") for p in text.split(","))]


class Trial:
    def __init__(self, args, condition, index, replica_a, out, replica_b=None):
        self.args, self.condition, self.index, self.a = args, condition, index, replica_a
        self.out = out
        self.events, self.timeline, self.decisions = [], [], []
        self.b = replica_b or Replica("b", args, out)
        self.state = adapter.State(MODEL, {"a": replica_a.url, "b": self.b.url}, desired=2 if replica_b else 1)
        self.state.serving["a"] = True
        self.state.serving["b"] = replica_b is not None
        self.lock = threading.Lock()

    def kill_b(self):
        """Fault: SIGKILL replica b's process group via serving/faults/kill_decode.sh."""
        r = subprocess.run(["bash", "serving/faults/kill_decode.sh", str(self.b.proc.pid)],
                           capture_output=True, text=True, timeout=30, check=False)  # fmt: skip
        self.event("decode_killed", replica="b", returncode=r.returncode, stderr=r.stderr[-500:])

    def event(self, kind, **fields):
        with self.lock:
            self.events.append({"t": now(), "event": kind, **fields})

    def live(self, name):
        return self.a.alive() if name == "a" else self.b.alive()

    def sample_timeline(self, halt):
        while not halt.wait(1.0):
            agg = self.state.aggregate(now())
            with self.state.lock:
                queues = {n: (h[-1][1].waiting, h[-1][1].running, h[-1][1].success) for n, h in self.state.history.items() if h}
            self.timeline.append({"t": now(), "desired": agg["replicas"], "p95_ttft_s": agg["p95_ttft_seconds"],
                                  "replicas": queues, "serving": dict(self.state.serving)})  # fmt: skip

    def scale_up(self):
        self.state.desired = 2
        self.b.spawn()
        self.event("replica_spawn", replica="b")
        self.b.wait_ready()
        self.event("replica_ready", replica="b")
        with self.state.lock:
            base = self.state.history["b"][-1][1].success if self.state.history["b"] else 0
        base = max(base, 1)  # the readiness completion
        self.state.serving["b"] = True
        self.event("replica_routable", replica="b")
        while self.b.alive() and self.state.serving["b"]:
            with self.state.lock:
                h = self.state.history["b"]
                done = h[-1][1].success if h else 0
            if done > base:
                self.event("replica_first_success", replica="b")
                return
            time.sleep(0.2)

    def scale_down(self):
        self.state.desired = 1
        self.state.serving["b"] = False
        self.event("replica_drain", replica="b")
        time.sleep(21)  # router FreshFor 15 s + one 5 s poll
        deadline = now() + 120
        while now() < deadline:
            with self.state.lock:
                h = self.state.history["b"]
                busy = h[-1][1].waiting + h[-1][1].running if h else 0
            if busy == 0:
                break
            time.sleep(0.5)
        self.b.stop()
        self.state.forget("b")
        self.event("replica_stopped", replica="b")

    def read_decisions(self, proc):
        workers = []
        for line in proc.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            self.decisions.append({"t": now(), **d})
            if d["want"] == d["current"]:
                continue
            self.event("scale_decision", current=d["current"], want=d["want"])
            if self.condition != "auto":
                continue
            if d["want"] > d["current"] and self.state.desired == 1 and not self.b.alive():
                workers.append(threading.Thread(target=self.scale_up, daemon=True))
            elif d["want"] < d["current"] and self.state.desired == 2 and self.state.serving["b"]:
                workers.append(threading.Thread(target=self.scale_down, daemon=True))
            else:
                continue
            workers[-1].start()

    def run(self):
        a = self.args
        self.out.mkdir(parents=True, exist_ok=True)
        server = adapter.serve(self.state, ADAPTER_PORT)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        poller = adapter.Poller(self.state, self.live)
        poller.start()
        halt = threading.Event()
        sampler = threading.Thread(target=self.sample_timeline, args=(halt,), daemon=True)
        sampler.start()
        config = self.out / "router.json"
        config.write_text(json.dumps({
            "Scope": {"Model": MODEL, "Revision": a.revision, "Adapter": "base"},
            "Backends": [{"ID": n, "URL": f"http://127.0.0.1:{p}", "MetricsURL": f"http://127.0.0.1:{ADAPTER_PORT}/snapshot/{n}"}
                         for n, p in PORTS.items()],
        }))  # fmt: skip
        time.sleep(2)  # first adapter scrape before the router's first poll
        router = start_process([a.router_bin, "-config", str(config), "-listen", f"127.0.0.1:{ROUTER_PORT}", *router_flags(self.condition)],
                               self.out / "router.log")  # fmt: skip
        if "listening" not in router.stdout.readline():
            raise RuntimeError("router did not start")
        threading.Thread(target=lambda: [None for _ in router.stdout], daemon=True).start()
        autoscaler = None
        timer = threading.Timer(a.kill_at, self.kill_b) if kills(self.condition) else None
        try:
            if self.condition in ("static", "auto"):  # fault mode runs no autoscaler
                autoscaler = start_process([
                    a.autoscaler_bin, "-metrics-url", f"http://127.0.0.1:{ADAPTER_PORT}/aggregate",
                    "-interval", "2s", "-min", "1", "-max", "2", "-queue-high", str(a.queue_high),
                    "-queue-low", str(a.queue_low), "-ttft-high", str(a.ttft_high), "-ttft-low", str(a.ttft_low),
                ], self.out / "autoscaler.log")  # fmt: skip
                reader = threading.Thread(target=self.read_decisions, args=(autoscaler,), daemon=True)
                reader.start()
            span = sum(s for s, _ in a.phases)
            cmd = [
                a.replay_bin, "-trace", str(a.trace_csv), "-format", "azure",
                "-url", f"http://127.0.0.1:{ROUTER_PORT}/v1/completions", "-mult", "1",
                "-token-ids", a.token_ids, "-timeout", f"{REQUEST_TIMEOUT_S}s", "-inflight", "256",
                "-seed", "17", "-api", "openai-text", "-model", MODEL, "-tenant", TENANT, "-account",
                *replay_flags(self.condition),
            ]  # fmt: skip
            self.event("replay_start")
            if timer:
                timer.daemon = True
                timer.start()
            with open(self.out / "replay.json", "wb") as stdout:
                proc = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, timeout=span + 180, check=False)
            self.event("replay_end", returncode=proc.returncode)
            (self.out / "replay.stderr").write_bytes(proc.stderr[:65536])
            # Let an in-progress scale-down finish before tearing down.
            deadline = now() + 180
            while self.condition == "auto" and self.state.desired == 2 and self.b.alive() and not self.state.serving["b"] and now() < deadline:
                time.sleep(1)
        finally:
            if timer:
                timer.cancel()
            stop_process(autoscaler)
            stop_process(router)
            if self.b.alive():
                self.b.stop()
                self.event("replica_stopped_at_trial_end", replica="b")
            halt.set()
            poller.halt.set()
            server.shutdown()
            server.server_close()
        for name, data in (("events", self.events), ("timeline", self.timeline), ("decisions", self.decisions)):
            (self.out / f"{name}.json").write_text(json.dumps(data))
        return json.loads((self.out / "replay.json").read_text())


def wait_idle(url, limit_s=180):
    deadline, quiet = now() + limit_s, 0
    while now() < deadline:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as r:
            text = r.read(4 << 20).decode()
        s = adapter.metrics.parse(text)
        quiet = quiet + 1 if s.waiting + s.running == 0 else 0
        if quiet >= 2:
            return True
        time.sleep(1)
    return False


def summarize(trials, phases, burst_s, drop_s):
    slo = analyze.slo_from_base_phase([t["records"] for t in trials if t["condition"] == "static"], phases)
    rows = []
    for t in trials:
        rows.append({
            "condition": t["condition"], "index": t["index"], "sender_valid": t["sender_valid"],
            "phases": analyze.phase_summary(t["records"], phases, slo),
            "reactions": analyze.reactions(t["events"], burst_s, drop_s),
        })  # fmt: skip

    def spread(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}

    by = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        by[cond] = {
            "phase_goodput_rps": [spread([r["phases"][i]["goodput_rps"] for r in rs]) for i in range(len(phases))],
            "phase_p99_ttft_ms": [spread([(r["phases"][i]["ttft_ms"] or {}).get("p99") for r in rs]) for i in range(len(phases))],
            "phase_failed": [spread([r["phases"][i]["failed"] for r in rs]) for i in range(len(phases))],
            "reactions": {k: spread([r["reactions"][k] for r in rs]) for k in rs[0]["reactions"]},
        }  # fmt: skip
    return {"slo": slo, "trials": rows, "by_condition": by}


def run_scale(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lengths = trace.read_lengths("serving/baseline/traces/azure-60s/calibration.csv") + trace.read_lengths(
        "serving/baseline/traces/azure-60s/evaluation.csv"
    )
    rows = trace.step_burst(lengths, args.phases, args.base_rps, seed=17)
    args.trace_csv = out / "trace.csv"
    trace.write_azure_csv(rows, args.trace_csv)
    order = alternate(args.conditions, args.pairs)
    monitor = Monitor(args.cache_root, 2 * 10**9)
    monitor.start()
    a = Replica("a", args, out)
    started = now()
    a.spawn()
    ready = a.wait_ready()
    meta = {"replica_a_start_to_ready_s": ready - started, "order": order, "phases": args.phases,
            "base_rps": args.base_rps, "requests": len(rows), "args": {k: str(v) for k, v in vars(args).items()},
            "nproc": os.cpu_count(), "loadavg_start": os.getloadavg()}  # fmt: skip
    trials, stop_reason = [], None
    try:
        for index, condition in enumerate(order):
            if monitor.stop_reason:
                stop_reason = monitor.stop_reason
                break
            wait_idle(a.url)
            trial = Trial(args, condition, index, a, out / f"trial-{index}-{condition}")
            result = trial.run()
            trials.append({"condition": condition, "index": index, "records": result["Run"]["Records"],
                           "sender_valid": bool(result["Run"]["SenderValid"]) and not result["Run"]["Aborted"],
                           "events": trial.events})  # fmt: skip
            print(json.dumps({"trial": index, "condition": condition, "events": len(trial.events)}), flush=True)
    finally:
        a.stop()
        monitor.halt.set()
    burst_s = args.phases[0][0]
    drop_s = burst_s + args.phases[1][0]
    summary = summarize(trials, args.phases, burst_s, drop_s) if trials else {}
    summary.update(meta=meta, stop_reason=stop_reason, peak_owned_rss_bytes=monitor.peak_rss,
                   loadavg_end=os.getloadavg())  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def fault_summary(records, events):
    """Request accounting and when failures were scheduled relative to the kill."""
    start = next(e["t"] for e in events if e["event"] == "replay_start")
    kill = next((e["t"] - start for e in events if e["event"] == "decode_killed"), None)
    failed = [r for r in records if r["Status"] != "ok"]
    statuses = {}
    for r in failed:
        statuses[r["Status"]] = statuses.get(r["Status"], 0) + 1
    rel = [r["ScheduledMs"] / 1000 - kill for r in failed] if kill is not None else []
    retried = [r for r in records if r.get("Attempts")]
    resent = {}
    for r in retried:
        for attempt in r["Attempts"][:-1]:
            resent[attempt["Status"]] = resent.get(attempt["Status"], 0) + 1
    return {
        "planned": len(records), "ok": len(records) - len(failed), "failed": len(failed),
        "failure_statuses": statuses, "kill_s": kill,
        "retried": len(retried), "retried_ok": sum(r["Status"] == "ok" for r in retried),
        "retried_attempt_statuses": resent,
        "first_failure_after_kill_s": min(rel) if rel else None,
        "last_failure_after_kill_s": max(rel) if rel else None,
    }  # fmt: skip


def run_fault(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lengths = trace.read_lengths("serving/baseline/traces/azure-60s/calibration.csv") + trace.read_lengths(
        "serving/baseline/traces/azure-60s/evaluation.csv"
    )
    args.trace_csv = out / "trace.csv"
    trace.write_azure_csv(trace.step_burst(lengths, args.phases, args.base_rps, seed=17), args.trace_csv)
    order = alternate(args.fault_conditions, args.pairs)
    monitor = Monitor(args.cache_root, 2 * 10**9)
    monitor.start()
    a = Replica("a", args, out)
    a.spawn()
    a.wait_ready()
    rows = []
    try:
        for index, condition in enumerate(order):
            wait_idle(a.url)
            trial_out = out / f"trial-{index}-{condition}"
            b = Replica("b", args, trial_out)
            b.spawn()
            b.wait_ready()
            trial = Trial(args, condition, index, a, trial_out, replica_b=b)
            result = trial.run()
            row = {"index": index, "condition": condition, "accounting": result.get("Accounting"),
                   "sender_valid": bool(result["Run"]["SenderValid"]),
                   **fault_summary(result["Run"]["Records"], trial.events)}  # fmt: skip
            st = router_log_stats((trial_out / "router.log").read_text())
            kill_t = next((e["t"] for e in trial.events if e["event"] == "decode_killed"), None)
            first = st.pop("first_eject_t")
            row["router"] = {**st, "kill_to_first_eject_s": None if kill_t is None or first is None else first - kill_t}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        a.stop()
        monitor.halt.set()
    (out / "fault-summary.json").write_text(json.dumps({
        "trials": rows, "order": order, "phases": args.phases, "base_rps": args.base_rps,
        "kill_at_s": args.kill_at, "peak_owned_rss_bytes": monitor.peak_rss,
        "nproc": os.cpu_count(), "loadavg_end": os.getloadavg(),
    }, indent=1))  # fmt: skip


def evict(snapshot):
    for path in Path(snapshot).iterdir():
        fd = os.open(path.resolve(), os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def prewarm(snapshot):
    for path in Path(snapshot).iterdir():
        with open(path.resolve(), "rb") as f:
            while f.read(16 << 20):
                pass


def run_cold(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in range(args.pairs):
        for cond in (("warm", "cold") if pair % 2 == 0 else ("cold", "warm")):
            (evict if cond == "cold" else prewarm)(args.snapshot)
            r = Replica("a", args, out / f"cold-{pair}-{cond}")
            t0 = now()
            r.spawn()
            ready = r.wait_ready()
            r.stop()
            log = r.log.read_text(errors="replace")
            weights = [line for line in log.splitlines() if "Loading weights took" in line or "Model loading took" in line]
            results.append({"pair": pair, "condition": cond, "start_to_ready_s": ready - t0, "load_log": weights,
                            "loadavg": os.getloadavg()})  # fmt: skip
            print(json.dumps(results[-1]), flush=True)
    summary = {}
    for cond in ("warm", "cold"):
        v = [r["start_to_ready_s"] for r in results if r["condition"] == cond]
        summary[cond] = {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
    (out / "cold-summary.json").write_text(json.dumps({"trials": results, "summary": summary, "nproc": os.cpu_count()}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("scale", "cold", "fault"), required=True)
    p.add_argument("--kill-at", type=float, default=30.0, help="fault mode: seconds after replay start")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--router-bin", default=".cache/t14/bin/router")
    p.add_argument("--autoscaler-bin", default=".cache/t14/bin/autoscaler")
    p.add_argument("--replay-bin", default=".cache/t14/bin/replay")
    p.add_argument("--token-ids", default=".cache/t14/qwen25-05b-words.json")
    p.add_argument("--gpu-util", type=float, default=0.25)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--phases", type=parse_phases, default=parse_phases("60:1,120:4,360:1"))
    p.add_argument("--base-rps", type=float, default=1.0)
    p.add_argument("--conditions", type=lambda s: s.split(","), default=["static", "auto"])
    p.add_argument("--fault-conditions", type=lambda s: s.split(","), default=["kill", "none"],
                   help="fault mode: any of kill, kill-retry, kill-client-retry, none")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--queue-high", type=float, default=4)
    p.add_argument("--queue-low", type=float, default=1)
    p.add_argument("--ttft-high", type=float, default=0.8)
    p.add_argument("--ttft-low", type=float, default=0.2)
    args = p.parse_args(argv)
    {"scale": run_scale, "cold": run_cold, "fault": run_fault}[args.mode](args)


if __name__ == "__main__":
    main()
