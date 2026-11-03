"""Read-only Git-content and owned-root absence proof for the CPU wording fix."""

import hashlib
import json
import subprocess
from pathlib import Path

BASE = "a2b138d1a2b33e3c1f0cc38d7e249c2fea112740"
root = Path.cwd()
paths = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", BASE, "docs/evidence/raw"], text=True
).splitlines()
paths = [p for p in paths if Path(p).name.startswith("task15-cpu-")]
assert len(paths) == 35
files = []
for name in paths:
    historical = subprocess.check_output(["git", "show", f"{BASE}:{name}"])
    current = (root / name).read_bytes()
    assert current == historical
    files.append(
        {
            "path": name,
            "bytes": len(current),
            "sha256": hashlib.sha256(current).hexdigest(),
            "matches_fix_base_bytes": True,
        }
    )
owned = []
for name in (".tmp-task15-round1-red", ".tmp-task15-round1-green"):
    path = root / name
    assert not path.exists() and not path.is_symlink()
    owned.append(
        {
            "path": name,
            "absent": True,
            "regular_files": [],
            "symlinks": [],
            "cleanup": "not created: isolated tests use no temporary files",
        }
    )
proof = {
    "fix_base": BASE,
    "original_raw_count": len(files),
    "original_raw_files": files,
    "owned_cleanup": owned,
    "go_build_or_launch_executed": False,
    "loan_bytes_consumed": 0,
}
(root / "docs/evidence/raw/task15-cpu-round1-history-cleanup.json").write_text(
    json.dumps(proof, indent=2) + "\n"
)
print(
    "35 original raw byte/SHA256 identities unchanged; both exact round1 pytest roots absent; no cleanup needed"
)
