"""Download a job's packed outputs from the asset stage (payloads upload them; jobs have no other way out).

  python scripts/snowflake/results.py get <prefix>/<run> <local dir>   # e.g. e12/e1-same-node
  python scripts/snowflake/results.py ls [<prefix>]

Payloads write @CRUCIBLE_LAB.PROBE.ASSETS/results/<prefix>/<run>.tar.gz; `get` downloads it into
<local dir> and unpacks it there (data filter, no absolute paths or links outside).
"""

import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.snowflake.assets import STAGE, _host_session


def get(name, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    s = _host_session()
    with tempfile.TemporaryDirectory() as tmp:
        s.file.get(f"@{STAGE}/results/{name}.tar.gz", tmp)
        archive = Path(tmp) / f"{Path(name).name}.tar.gz"
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
        size = archive.stat().st_size
    s.close()
    print(f"got results/{name}.tar.gz ({size} bytes) -> {dest}")


def ls(prefix=""):
    s = _host_session()
    for r in s.sql(f"LIST @{STAGE}/results/{prefix}").collect():
        print(r["name"], r["size"])
    s.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "get":
        get(sys.argv[2].strip("/"), sys.argv[3])
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "ls":
        ls(sys.argv[2] if len(sys.argv) == 3 else "")
    else:
        raise SystemExit(__doc__)
