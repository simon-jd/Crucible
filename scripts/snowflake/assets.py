"""Move models and data to the Snowflake asset stage and back into a job (the trial account has no egress).

  python scripts/snowflake/assets.py put <local dir or file> <name>   # e.g. models/Qwen2.5-0.5B-Instruct
  python scripts/snowflake/assets.py ls
  python scripts/snowflake/assets.py get <stage file> <local dir>       # e.g. a job's output tar

Each asset is one uncompressed tar, @CRUCIBLE_LAB.PROBE.ASSETS/<name>.tar, with <name>.sha256 next to it, so a
job downloads a single file and checks it. Symlinks are followed, so a Hugging Face snapshot directory becomes
plain files. Inside a job, fetch(name, dest) downloads, verifies and unpacks it into dest and returns timings.
On the host, set SNOWFLAKE_ACCOUNT to the account identifier (<orgname>-<accountname>).
"""
import hashlib
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

STAGE = "CRUCIBLE_LAB.PROBE.ASSETS"
KEY = os.path.expanduser("~/.snowflake-keys/crucible/CRUCIBLE_PROBE_SVC.p8")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _host_session():
    from snowflake.snowpark import Session
    return Session.builder.configs({
        "account": os.environ["SNOWFLAKE_ACCOUNT"], "user": "CRUCIBLE_PROBE_SVC", "role": "CRUCIBLE_PROBE_ROLE",
        "warehouse": "CRUCIBLE_XS", "database": "CRUCIBLE_LAB", "schema": "PROBE",
        "authenticator": "SNOWFLAKE_JWT", "private_key_file": KEY}).create()


def put(local, name):
    local = Path(local)
    s = _host_session()
    s.sql(f"CREATE STAGE IF NOT EXISTS {STAGE}").collect()
    with tempfile.TemporaryDirectory(dir=os.environ.get("ASSET_TMP")) as tmp:
        tar_path = Path(tmp) / f"{Path(name).name}.tar"
        with tarfile.open(tar_path, "w", dereference=True) as tar:
            tar.add(local, arcname=".")
        sha = Path(tmp) / f"{Path(name).name}.sha256"
        sha.write_text(_sha256(tar_path) + "\n")
        sub = os.path.dirname(name)
        for f in (tar_path, sha):
            s.file.put(str(f), f"@{STAGE}/{sub}", auto_compress=False, overwrite=True, parallel=8)
        size = tar_path.stat().st_size
    s.close()
    print(f"put {name}.tar ({size} bytes)")


def fetch(name, dest):
    """Inside a job: download <name>.tar, check its sha256, unpack into dest. Returns sizes and timings."""
    from snowflake.snowpark import Session
    s = Session.builder.getOrCreate()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = time.time()
        s.file.get(f"@{STAGE}/{name}.tar", tmp)
        s.file.get(f"@{STAGE}/{name}.sha256", tmp)
        dt = time.time() - t
        tar_path = Path(tmp) / f"{Path(name).name}.tar"
        want = (Path(tmp) / f"{Path(name).name}.sha256").read_text().strip()
        if _sha256(tar_path) != want:
            raise RuntimeError(f"asset {name}: sha256 mismatch")
        size = tar_path.stat().st_size
        with tarfile.open(tar_path) as tar:
            tar.extractall(dest, filter="data")
    return {"asset": name, "bytes": size, "get_seconds": round(dt, 1), "MBps": round(size / dt / 1e6, 1)}


def get(name, dest):
    """On the host: download one stage file (for example a job's output tar) into dest."""
    s = _host_session()
    Path(dest).mkdir(parents=True, exist_ok=True)
    rows = s.file.get(f"@{STAGE}/{name}", str(dest))
    s.close()
    for r in rows:
        print(f"got {r.file} ({r.size} bytes) -> {dest}")


def ls():
    s = _host_session()
    for r in s.sql(f"LIST @{STAGE}").collect():
        print(r["name"], r["size"])
    s.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "put":
        put(sys.argv[2], sys.argv[3].strip("/"))
    elif len(sys.argv) == 4 and sys.argv[1] == "get":
        get(sys.argv[2].strip("/"), sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "ls":
        ls()
    else:
        raise SystemExit(__doc__)
