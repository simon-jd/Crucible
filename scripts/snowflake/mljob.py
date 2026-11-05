"""Run a payload as a Snowflake ML Job on one or more GPU nodes, and always stop the compute pool afterwards.

  python scripts/snowflake/mljob.py --pool CRUCIBLE_2XA10_POOL --entry scripts/snowflake/payloads/x.py \\
      --include rl kernels scripts/snowflake --pip vllm==0.10.2 torch==2.8.0 --out .cache/snowflake/x \\
      [--instances 2] [--max-minutes 60] [--env KEY=VALUE ...] [--args ...]

What it does, in order:
- takes an exclusive lock for the pool (/var/tmp/snowflake-pool-<POOL>.lock), so runs from several shells queue up
  instead of meeting on one pool;
- refuses to start if the pool still runs services or jobs it did not start;
- copies the --include paths (relative to the repository root) into a temporary payload directory and submits it
  with submit_directory, target_instances = min_instances = --instances;
- installs --pip packages from Snowflake's PyPI proxy (the trial account has no internet egress);
- waits up to --max-minutes, then cancels the job if it is still running;
- saves every instance's log to <out>/instance<i>.log, collects lines that start with "RESULT " into
  <out>/report.json, runs STOP ALL + SUSPEND on the pool and records the pool state;
- appends one line per run to the ledger ($SNOWFLAKE_LEDGER, default ~/.cache/crucible-snowflake-ledger.jsonl)
  with the wall time and an upper-bound credit estimate, so several shells can track one budget.

Environment facts this relies on (measured 2026-11-03/04 on a Snowflake trial account):
- GPU_NV_SM = 1x A10 24 GB (1.70 credits/h), GPU_NV_2M = 2x A10 on one node (3.50 credits/h); A100 families
  need a quota request.
- Between job instances only ports 10001, 12031-13000 and 32768-59999 are open: put rendezvous ports
  (MASTER_PORT, vLLM KV ports) in 12900-12999.
- Jobs read models and data from a stage with a Snowpark session (Session.builder.getOrCreate()); see assets.py.
Connects as CRUCIBLE_PROBE_SVC with the key at ~/.snowflake-keys/crucible/CRUCIBLE_PROBE_SVC.p8 (path only).
Set SNOWFLAKE_ACCOUNT to the account identifier (<orgname>-<accountname>).
"""
import argparse
import fcntl
import json
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from snowflake.ml.jobs import submit_directory
from snowflake.snowpark import Session

ROOT = Path(__file__).resolve().parents[2]
KEY = os.path.expanduser("~/.snowflake-keys/crucible/CRUCIBLE_PROBE_SVC.p8")
STAGE = "CRUCIBLE_LAB.PROBE.MLJOB_STAGE"
PYPI = "snowflake.snowpark.pypi_shared_repository"
LEDGER = os.path.expanduser(os.environ.get("SNOWFLAKE_LEDGER", "~/.cache/crucible-snowflake-ledger.jsonl"))
RATE = {"GPU_NV_SM": 1.70, "GPU_NV_2M": 3.50, "GPU_NV_XS": 0.25, "GPU_NV_3M": 3.55, "GPU_NV_SL": 13.50}


def log(msg):
    print(f"{datetime.now(UTC):%H:%M:%SZ} {msg}", flush=True)


def session():
    return Session.builder.configs({
        "account": os.environ["SNOWFLAKE_ACCOUNT"], "user": "CRUCIBLE_PROBE_SVC", "role": "CRUCIBLE_PROBE_ROLE",
        "warehouse": "CRUCIBLE_XS", "database": "CRUCIBLE_LAB", "schema": "PROBE",
        "authenticator": "SNOWFLAKE_JWT", "private_key_file": KEY}).create()


def pool_state(s, pool):
    row = s.sql(f"SHOW COMPUTE POOLS LIKE '{pool}'").collect()[0].as_dict()
    return {k: str(row.get(k)) for k in ("state", "instance_family", "active_nodes", "idle_nodes", "num_services",
                                         "num_jobs")}


def stop_pool(s, pool):
    out = []
    for q in (f"ALTER COMPUTE POOL {pool} STOP ALL", f"ALTER COMPUTE POOL {pool} SUSPEND"):
        try:
            s.sql(q).collect()
            out.append(f"{q}: ok")
        except Exception as e:  # noqa: BLE001 - record the error and still try the next statement
            out.append(f"{q}: {str(e).splitlines()[0][:200]}")
    return out


def build_payload(includes, entry):
    tmp = Path(tempfile.mkdtemp(prefix="mljob-payload-"))
    for rel in sorted(set(includes) | {entry}):
        src = ROOT / rel
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".pytest_cache"))
        else:
            shutil.copy2(src, dst)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--entry", required=True, help="payload script, relative to the repository root")
    ap.add_argument("--include", nargs="*", default=[], help="extra repository paths to ship with the payload")
    ap.add_argument("--pip", nargs="*", default=[])
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE environment variables for the job")
    ap.add_argument("--args", nargs=argparse.REMAINDER, default=[], help="arguments for the payload (put last)")
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--lock-wait-minutes", type=float, default=240)
    ap.add_argument("--name", default=None, help="job name prefix (default: the entry file's stem)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name = "crucible_" + (a.name or Path(a.entry).stem)
    report = {"started_at": datetime.now(UTC).isoformat(timespec="seconds"), "pool": a.pool,
              "instances": a.instances, "entry": a.entry, "include": a.include, "pip": a.pip, "env": a.env,
              "args": a.args}

    lock = open(f"/var/tmp/snowflake-pool-{a.pool}.lock", "w")  # noqa: SIM115 - held for the whole run
    t_lock = time.time()
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t_lock > a.lock_wait_minutes * 60:
                raise SystemExit(f"pool lock for {a.pool} not free after {a.lock_wait_minutes} min")
            time.sleep(20)
    report["lock_wait_s"] = round(time.time() - t_lock, 1)

    s = session()
    report["pool_before"] = before = pool_state(s, a.pool)
    if int(before["num_services"] or 0) or int(before["num_jobs"] or 0):
        raise SystemExit(f"{a.pool} already runs services or jobs: {before}; not starting")
    job, payload = None, build_payload(a.include, a.entry)
    t0 = time.time()
    try:
        s.sql(f"CREATE STAGE IF NOT EXISTS {STAGE}").collect()
        # submit_directory uploads to <stage>/<name>/ without overwriting, so clear the copy of an earlier run.
        s.sql(f"REMOVE @{STAGE}/{name}/").collect()
        kw = {"pip_requirements": a.pip, "artifact_repositories": [PYPI]} if a.pip else {}
        if a.env:
            kw["env_vars"] = dict(e.split("=", 1) for e in a.env)
        job = submit_directory(str(payload), a.pool, entrypoint=a.entry, stage_name=STAGE, session=s,
                               args=a.args or None, target_instances=a.instances, min_instances=a.instances,
                               database="CRUCIBLE_LAB", schema="PROBE", query_warehouse="CRUCIBLE_XS", name=name,
                               **kw)
        report["job_id"] = job.id
        log(f"submitted {job.id} on {a.pool} x{a.instances}")
        try:
            report["status"] = job.wait(timeout=a.max_minutes * 60)
        except Exception as e:  # noqa: BLE001 - a failed wait still collects logs and stops the pool
            report["status"] = f"wait: {e!r}"[:500]
        log(f"status {report['status']} after {time.time() - t0:.0f} s")
        report["results"] = []
        for i in range(a.instances):
            try:
                text = job.get_logs(instance_id=i)
            except Exception as e:  # noqa: BLE001 - keep the other instances' logs
                text = f"get_logs failed: {e!r}"
            (a.out / f"instance{i}.log").write_text(text)
            for line in text.splitlines():
                if line.startswith("RESULT "):
                    try:
                        report["results"].append({"instance": i, **json.loads(line[len("RESULT "):])})
                    except json.JSONDecodeError:
                        report["results"].append({"instance": i, "unparsed": line[:2000]})
    except Exception as e:  # noqa: BLE001 - recorded in the report; the pool is stopped in finally
        report["error"] = repr(e)[:2000]
        log(f"error: {report['error'][:300]}")
    finally:
        if job is not None:
            try:
                if job.status in ("PENDING", "RUNNING"):
                    job.cancel()
                    report["cancelled"] = True
            except Exception as e:  # noqa: BLE001 - cancellation is best effort before the pool stop
                report["cancelled"] = repr(e)[:300]
        report["seconds"] = round(time.time() - t0, 1)
        report["stop"] = stop_pool(s, a.pool)
        report["pool_after"] = pool_state(s, a.pool)
        rate = RATE.get(before["instance_family"])
        # Upper-bound estimate: wall time from submit to stop for every requested node, including node start.
        report["credits_estimate"] = round(report["seconds"] / 3600 * a.instances * rate, 3) if rate else None
        s.close()
        shutil.rmtree(payload, ignore_errors=True)
        (a.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        with open(LEDGER, "a") as f:
            f.write(json.dumps({k: report.get(k) for k in ("started_at", "pool", "instances", "entry", "job_id",
                                                           "status", "seconds", "credits_estimate")}
                               | {"out": str(a.out.resolve())}) + "\n")
        log(f"wrote {a.out / 'report.json'} (credits estimate {report['credits_estimate']})")
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
