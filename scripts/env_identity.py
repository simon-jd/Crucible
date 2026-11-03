"""Print a reproducible identity for a host-native Python environment.

Containers are not used here, so there is no image digest. The identity is the sorted
`name==version` list of installed distributions (from dist-info/egg-info directory
names, normalized like PEP 503) and the SHA256 of that list, one line per entry.

    .venv/bin/python scripts/env_identity.py .venv-ml/lib/python3.12/site-packages
"""

import hashlib
import re
import sys
from pathlib import Path

DIST = re.compile(r"^(?P<name>.+?)-(?P<version>[^-]+)\.(?:dist|egg)-info$")


def identity(site_packages):
    lines = sorted(
        f"{re.sub(r'[-_.]+', '-', m['name']).lower()}=={m['version']}"
        for entry in Path(site_packages).iterdir()
        if entry.is_dir() and (m := DIST.match(entry.name))
    )
    digest = hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest()
    return lines, digest


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    lines, digest = identity(argv[0])
    print("\n".join(lines))
    print(f"# {len(lines)} distributions, sha256 {digest}")


if __name__ == "__main__":
    main()
