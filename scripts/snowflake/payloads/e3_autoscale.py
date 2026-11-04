"""E3 payload: router + autoscaler + localscale actuator on one GPU_NV_2M node (2x A10), one replica per GPU.

  --args <tag> [serving.localscale.run --mode scale args, e.g. --phases 60:1,120:4,360:1 --pairs 3 ...]

Fetches models/Qwen2.5-0.5B-Instruct from the asset stage, records the node (GPUs, topology, CPUs, packages),
samples per-GPU memory and utilization once a second with nvidia-smi, and runs serving.localscale.run with
`--gpus a=0,b=1` (replica a on GPU0, the scaled-out or second fixed replica b on GPU1). The router, autoscaler
and replayer are the static binaries in .cache/snowflake/bin (make snowflake-bins), the replayer's word list is
.cache/snowflake/qwen25-05b-words.json. The whole output directory goes back to the asset stage as
e3/runs/<tag>.tar.gz (fetch with `assets.py get e3/runs/<tag>.tar.gz <dir>`); a RESULT line carries the summary.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root inside the job
sys.path.insert(0, str(ROOT))
from scripts.snowflake.assets import STAGE, fetch

REV = "7ae557604adf67be50417f59c2c2f167def9a775"
BIN = ROOT / ".cache/snowflake/bin"
WORDS = ROOT / ".cache/snowflake/qwen25-05b-words.json"


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        return f"{cmd[0]}: {e!r}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def node_info():
    import importlib.metadata as md

    def ver(name):
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            return None

    return {
        "gpus": sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
                    "--format=csv,noheader"]).splitlines(),
        "topo": sh(["nvidia-smi", "topo", "-m"]), "nproc": os.cpu_count(), "loadavg": os.getloadavg(),
        "meminfo": sh(["head", "-3", "/proc/meminfo"]).splitlines(), "python": sys.version.split()[0],
        "packages": {p: ver(p) for p in ("vllm", "torch", "transformers", "numpy")},
        "binaries": {p.name: sha256(p) for p in sorted(BIN.iterdir())}, "words_sha256": sha256(WORDS),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }  # fmt: skip


def condensed(summary):
    """The numbers the write-up needs, per condition (medians and ranges stay in summary.json)."""
    out = {"slo": summary.get("slo"), "stop_reason": summary.get("stop_reason"), "meta": summary.get("meta"),
           "peak_owned_rss_bytes": summary.get("peak_owned_rss_bytes"), "trials": []}  # fmt: skip
    for t in summary.get("trials", []):
        out["trials"].append({
            "index": t["index"], "condition": t["condition"], "sender_valid": t["sender_valid"],
            "accounting": t["accounting"], "reactions": t["reactions"],
            "goodput_rps": [p["goodput_rps"] for p in t["phases"]],
            "p50_tpot_ms": [(p["tpot_ms"] or {}).get("p50") for p in t["phases"]],
            "p50_ttft_ms": [(p["ttft_ms"] or {}).get("p50") for p in t["phases"]],
            "p99_ttft_ms": [(p["ttft_ms"] or {}).get("p99") for p in t["phases"]],
        })  # fmt: skip
    return out


def upload(out, tag):
    from snowflake.snowpark import Session

    tar_path = Path("/tmp") / f"{tag}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out, arcname=tag)
    s = Session.builder.getOrCreate()
    s.file.put(str(tar_path), f"@{STAGE}/e3/runs", auto_compress=False, overwrite=True)
    return {"stage_path": f"e3/runs/{tar_path.name}", "bytes": tar_path.stat().st_size, "sha256": sha256(tar_path)}


def main():
    tag, extra = sys.argv[1], sys.argv[2:]
    os.chdir(ROOT)
    work = Path("/tmp/e3")
    rt = work / "runtime"
    for d in ("vllm", "triton", "torch", "torch-home", "xdg", "cuda", "tmp"):
        (rt / d).mkdir(parents=True, exist_ok=True)
    out = work / tag
    out.mkdir(parents=True, exist_ok=True)
    info = node_info()
    print("NODE " + json.dumps(info), flush=True)
    model = work / "model"
    got = fetch("models/Qwen2.5-0.5B-Instruct", model)
    info["model_fetch"] = got
    info["model_safetensors_sha256"] = sha256(model / "model.safetensors")
    (out / "node.json").write_text(json.dumps(info, indent=1))
    print("FETCH " + json.dumps(got), flush=True)
    # The job's stage mount drops the executable bit, so run copies of the binaries from local disk.
    bins = work / "bin"
    bins.mkdir(exist_ok=True)
    for src in BIN.iterdir():
        shutil.copyfile(src, bins / src.name)
        (bins / src.name).chmod(0o755)

    with open(out / "gpu-samples.csv", "w") as samples:  # the child keeps its own descriptor
        sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits", "-lms", "1000"], stdout=samples,
                                   stderr=subprocess.STDOUT)  # fmt: skip
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               CUDA_CACHE_PATH=str(rt / "cuda"), XDG_CACHE_HOME=str(rt / "xdg"), TORCH_HOME=str(rt / "torch-home"),
               VLLM_CACHE_ROOT=str(rt / "vllm"), TRITON_CACHE_DIR=str(rt / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(rt / "torch"), TMPDIR=str(rt / "tmp"), OMP_NUM_THREADS="4",
               PYTHONPATH=str(ROOT))  # fmt: skip
    cmd = [sys.executable, "-m", "serving.localscale.run", "--mode", "scale", "--snapshot", str(model),
           "--revision", REV, "--cache-root", str(rt), "--output", str(out),
           "--router-bin", str(bins / "router"), "--autoscaler-bin", str(bins / "autoscaler"),
           "--replay-bin", str(bins / "replay"), "--token-ids", str(WORDS), "--gpus", "a=0,b=1", *extra]  # fmt: skip
    print("RUN " + " ".join(cmd), flush=True)
    t0 = time.time()
    with open(out / "driver.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            log.write(line)
            print("DRIVER " + line.rstrip()[:2000], flush=True)
        rc = proc.wait()
    sampler.terminate()
    sampler.wait(10)
    result = {"tag": tag, "returncode": rc, "seconds": round(time.time() - t0, 1), "node": info}
    summary_path = out / "summary.json"
    if summary_path.exists():
        result["summary"] = condensed(json.loads(summary_path.read_text()))
    shutil.rmtree(rt / "tmp", ignore_errors=True)
    try:
        result["upload"] = upload(out, tag)
    except Exception as e:  # noqa: BLE001 - the RESULT line still carries the summary
        result["upload"] = {"error": repr(e)[:500]}
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
