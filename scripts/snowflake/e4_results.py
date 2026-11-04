"""Download an E4 result tar that a job put on the asset stage, and unpack it.

  python scripts/snowflake/e4_results.py ls
  python scripts/snowflake/e4_results.py get <tag> <dest dir>

The payload (scripts/snowflake/payloads/e4_pd.py) puts @CRUCIBLE_LAB.PROBE.ASSETS/e4-results/<tag>.tar;
this unpacks it into <dest dir>/<tag>/<layout>/.
"""

import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.snowflake.assets import STAGE, _host_session

PREFIX = f"@{STAGE}/e4-results"


def get(tag, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    s = _host_session()
    with tempfile.TemporaryDirectory() as tmp:
        s.file.get(f"{PREFIX}/{tag}.tar", tmp)
        with tarfile.open(Path(tmp) / f"{tag}.tar") as tar:
            tar.extractall(dest, filter="data")
    s.close()
    print(f"unpacked {tag} into {dest / tag}")


def ls():
    s = _host_session()
    for r in s.sql(f"LIST {PREFIX}").collect():
        print(r["name"], r["size"])
    s.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "ls":
        ls()
    else:
        raise SystemExit(__doc__)
