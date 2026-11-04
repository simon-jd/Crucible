"""E1/E2 payload: run one learner/rollout pair (or time-shared GRPO runs) on Snowflake GPUs.

  e12.py --run NAME --task e1 --layout same-node|two-node [--role-env learner|rollout|all K=V ...]
  e12.py --run NAME --task pair --layout same-node|two-node --seed S [--steps N] [--eval-before]
  e12.py --run NAME --task timeshared --seeds S [S] [--steps N]      # rl/local/grpo_run.py unchanged
  common: [--eval-subset N] (smoke runs: first N problems of each held-out set) [--port P]

same-node: both roles are subprocesses of this process, CUDA_VISIBLE_DEVICES=0 (learner)
and 1 (rollout), i.e. GPU_NV_2M's two A10s. two-node: one Ray actor per node
(STRICT_SPREAD placement group over the job's 2 instances), each running its role as a
subprocess; the repository copy and the assets are shipped to the actors through Ray.
timeshared: one grpo_run per seed, each on its own node (Ray actors when several seeds).
Every process runs with an nvidia-smi sampler (0.5 s: utilization, memory). Outputs are
packed into one tar.gz, uploaded to @ASSETS/results/e12/<run>.tar.gz (scripts/snowflake/
results.py get), and summarized in one line "RESULT <json>".
"""

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.snowflake.assets import fetch

STAGE = "CRUCIBLE_LAB.PROBE.ASSETS"
MODEL_ASSET = "models/Qwen2.5-0.5B-Instruct"
DATA_ASSET = "e12/t11-data-full"
DATA_SHA = "dbdf1a1b5f70953da3e59d75c86c14555e6e0c71bb6d55186b870755595b271e"


def log(msg):
    print(f"[e12 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pack(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(path, arcname=".")
    return buf.getvalue()


def unpack(data, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def run_role(base, name, argv, env, timeout):
    """Run one role in <base>/repo with its own nvidia-smi sampler; returns status + paths."""
    out = Path(base) / "out" / name
    out.mkdir(parents=True, exist_ok=True)
    full_env = os.environ | {"OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"} | env
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=open(out / "nvidia-smi.csv", "w"), stderr=subprocess.DEVNULL,  # noqa: SIM115 - closed at exit
    )  # fmt: skip
    t = time.time()
    with open(out / "stdout.log", "w") as stream:
        proc = subprocess.Popen([sys.executable, *argv], cwd=Path(base) / "repo", env=full_env,
                                stdout=stream, stderr=subprocess.STDOUT)  # fmt: skip
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = "timeout"
    smi.terminate()
    smi.wait(10)
    text = (out / "stdout.log").read_text(errors="replace")
    results = [
        json.loads(x[len("ROLE_RESULT ") :])
        for x in text.splitlines()
        if x.startswith("ROLE_RESULT ")
    ]
    return {"role": name, "exit": code, "seconds": round(time.time() - t, 1), "host": socket.gethostname(),
            "role_results": results, "tail": text.splitlines()[-(5 if code == 0 else 40):]}  # fmt: skip


def stage_inputs(base, eval_subset):
    """Writable repository copy plus model and data under <base>."""
    base = Path(base)
    if not (base / "repo").exists():
        shutil.copytree(ROOT, base / "repo", ignore=shutil.ignore_patterns("__pycache__"))
    got = [fetch(MODEL_ASSET, str(base / "model")), fetch(DATA_ASSET, str(base / "data"))]
    data_file = base / "data" / "data-full.json"
    data = json.loads(data_file.read_text())
    if data["sha256"] != DATA_SHA:
        raise RuntimeError("held-out/train data digest differs from the 4090 runs")
    if eval_subset:
        data["gsm8k_eval"] = data["gsm8k_eval"][:eval_subset]
        data["math500_eval"] = data["math500_eval"][:eval_subset]
        data["sha256"] = f"smoke-subset-{eval_subset}-of-{DATA_SHA}"
        data_file = base / "data" / f"data-smoke{eval_subset}.json"
        data_file.write_text(json.dumps(data))
    return got, data_file


def make_actor_class():
    import ray

    @ray.remote(num_gpus=1, num_cpus=1)
    class Node:
        def info(self):
            return {"ip": ray.util.get_node_ip_address(), "host": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}  # fmt: skip

        def setup(self, bundle):
            self.base = tempfile.mkdtemp(prefix="e12-node-")
            unpack(bundle, self.base)
            return self.base

        def run(self, name, argv, env, timeout):
            argv = [a.replace("{BASE}", self.base) for a in argv]
            status = run_role(self.base, name, argv, env, timeout)
            status["output_tar"] = pack(Path(self.base) / "out" / name)
            return status

    return Node


def launch(roles, layout, base, timeout):
    """roles: [(name, argv, env)]; '{PEER}' in argv is replaced by the first role's host."""
    if layout == "local":
        procs = []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(len(roles)) as pool:
            for name, argv, env in roles:
                argv = [a.replace("{PEER}", "127.0.0.1").replace("{BASE}", str(base)) for a in argv]
                procs.append(pool.submit(run_role, base, name, argv, env, timeout))
                time.sleep(2)
            return [p.result() for p in procs]
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    pg = placement_group([{"GPU": 1, "CPU": 1}] * len(roles), strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=600)
    node_cls = make_actor_class()
    actors = [node_cls.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(len(roles))]  # fmt: skip
    infos = ray.get([a.info.remote() for a in actors], timeout=600)
    log(f"actors: {infos}")
    staging = Path(tempfile.mkdtemp(prefix="e12-bundle-"))
    for part in ("repo", "model", "data"):
        os.symlink(Path(base) / part, staging / part)
    t = time.time()
    with tarfile.open(staging / "bundle.tar", "w", dereference=True) as tar:
        for part in ("repo", "model", "data"):
            tar.add(staging / part, arcname=part)
    bundle = ray.put((staging / "bundle.tar").read_bytes())
    ray.get([a.setup.remote(bundle) for a in actors], timeout=900)
    log(f"bundle shipped to {len(actors)} actors in {time.time() - t:.0f} s")
    peer = infos[0]["ip"]
    calls = []
    for actor, (name, argv, env) in zip(actors, roles, strict=True):
        argv = [a.replace("{PEER}", peer) for a in argv]
        calls.append(actor.run.remote(name, argv, env, timeout))
        time.sleep(2)
    statuses = ray.get(calls)
    for status, info in zip(statuses, infos, strict=True):
        unpack(status.pop("output_tar"), Path(base) / "out" / status["role"])
        status["node"] = info
    return statuses


def gpu_profile(csv_path):
    """Peak memory and mean utilization per GPU index from an nvidia-smi sample file."""
    per = {}
    for line in Path(csv_path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            idx, util, used, total = (
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        except ValueError:
            continue
        d = per.setdefault(
            idx, {"samples": 0, "util_sum": 0.0, "peak_used_mib": 0.0, "total_mib": total}
        )
        d["samples"] += 1
        d["util_sum"] += util
        d["peak_used_mib"] = max(d["peak_used_mib"], used)
    return {str(k): {"samples": v["samples"], "mean_util_pct": round(v["util_sum"] / max(1, v["samples"]), 1),
                     "peak_used_mib": v["peak_used_mib"], "total_mib": v["total_mib"]} for k, v in per.items()}  # fmt: skip


def upload(base, run):
    archive = Path(base) / f"{run}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(Path(base) / "out", arcname=run)
    try:
        from snowflake.snowpark import Session

        s = Session.builder.getOrCreate()
        s.file.put(str(archive), f"@{STAGE}/results/e12", auto_compress=False, overwrite=True)
        return {"uploaded": f"@{STAGE}/results/e12/{archive.name}", "bytes": archive.stat().st_size}
    except Exception as e:  # noqa: BLE001 - report, do not lose the RESULT line
        return {"upload_error": repr(e)[:500], "bytes": archive.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", choices=("e1", "pair", "timeshared"), required=True)
    ap.add_argument("--layout", choices=("same-node", "two-node", "single"), default="same-node")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--seeds", type=int, nargs="*", default=[17])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eval-subset", type=int, default=0)
    ap.add_argument("--eval-before", action="store_true")
    ap.add_argument("--port", type=int, default=12930)
    ap.add_argument("--timeout", type=float, default=7200, help="per role (s)")
    ap.add_argument(
        "--role-env", nargs="*", default=[], help="ROLE:K=V with ROLE learner|rollout|all"
    )
    a = ap.parse_args()
    t0 = time.time()
    base = Path(tempfile.mkdtemp(prefix="e12-"))
    report = {"run": a.run, "task": a.task, "layout": a.layout, "nproc": os.cpu_count()}
    got, data_file = stage_inputs(base, a.eval_subset)
    report["fetch"] = got
    model, data = "{BASE}/model", "{BASE}/data/" + data_file.name
    env = {"learner": {}, "rollout": {}}
    for item in a.role_env:
        role, kv = item.split(":", 1)
        k, v = kv.split("=", 1)
        for r in env if role == "all" else {role: None}:
            env[r][k] = v
    local = a.layout != "two-node"
    gpu = (
        {"learner": "0", "rollout": "1"}
        if a.layout == "same-node"
        else {"learner": None, "rollout": None}
    )

    def role_env(role):
        e = dict(env[role])
        if gpu[role] is not None:
            e["CUDA_VISIBLE_DEVICES"] = gpu[role]
        return e

    if a.task == "e1":
        roles = [("learner", ["-m", "rl.sync.nccl_check", "--role", "learner", "--model", model, "--port", str(a.port),
                              "--output", "{BASE}/out/learner"], role_env("learner")),
                 ("rollout", ["-m", "rl.sync.nccl_check", "--role", "rollout", "--model", model, "--port", str(a.port),
                              "--peer", "{PEER}", "--output", "{BASE}/out/rollout"], role_env("rollout"))]  # fmt: skip
    elif a.task == "pair":
        common = ["--data", data, "--model", model, "--port", str(a.port), "--seed", str(a.seed),
                  "--steps", str(a.steps), "--reward", "eos", "--layout", a.layout]  # fmt: skip
        roles = [("learner", ["-m", "rl.local.async_grpo", "--role", "learner", *common, "--output", "{BASE}/out/learner"],
                  role_env("learner")),
                 ("rollout", ["-m", "rl.local.async_grpo", "--role", "rollout", *common, "--peer", "{PEER}",
                              "--output", "{BASE}/out/rollout", *(["--eval-before"] if a.eval_before else [])],
                  role_env("rollout"))]  # fmt: skip
    else:
        roles = [(f"s{seed}", ["-m", "rl.local.grpo_run", "--data", data, "--model", model, "--output",
                               f"{{BASE}}/out/s{seed}/run", "--staleness", "1", "--reward", "eos", "--seed", str(seed),
                               "--steps", str(a.steps)], {"CUDA_VISIBLE_DEVICES": "0"} if len(a.seeds) == 1 else {})
                 for seed in a.seeds]  # fmt: skip
        local = len(a.seeds) == 1
    statuses = launch(roles, "local" if local else "ray", base, a.timeout)
    report["roles"] = statuses
    for s in statuses:
        csv = base / "out" / s["role"] / "nvidia-smi.csv"
        if csv.exists():
            s["gpu_profile"] = gpu_profile(csv)
    if a.task == "pair" and all(s["exit"] == 0 for s in statuses):
        sys.path.insert(0, str(base / "repo"))
        from rl.local.async_grpo import merge

        merged = merge(base / "out/learner", base / "out/rollout", base / "out/merged")
        report["pipeline"] = merged["pipeline"]
        report["after"] = {k: v["pass_at_1"] for k, v in merged["after"].items()}
        if "before" in merged:
            report["before"] = {k: v["pass_at_1"] for k, v in merged["before"].items()}
    if a.task == "timeshared":
        for seed in a.seeds:
            summary = base / "out" / f"s{seed}" / "run" / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                report[f"s{seed}"] = {"before": {k: v["pass_at_1"] for k, v in s["before"].items()},
                                      "after": {k: v["pass_at_1"] for k, v in s["after"].items()},
                                      "wall_s": s["wall_s"], "peak": s["peak_cuda_allocated_bytes"]}  # fmt: skip
    report["upload"] = upload(base, a.run)
    report["seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
