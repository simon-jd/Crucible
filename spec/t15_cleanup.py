"""Delete one T15 offline feature batch (the user's 2026-10-29 permission, phase 7).

Only a directory named features* directly under .cache/t15/ may be removed, and only
when nothing inside it is a symlink. Everything else raises PermissionError before any
file is touched. usage: python -m spec.t15_cleanup .cache/t15/features-full
The only other targets are phase 7's own scratch dirs in OWN_DIRS:
python -m spec.t15_cleanup --own train-full
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(".cache/t15")


def _check(target, root):
    root = Path(root).resolve()
    target = Path(target)
    if target.is_symlink():
        raise PermissionError(f"refusing symlink {target}")
    resolved = target.resolve()
    if resolved.parent != root or not resolved.name.startswith("features"):
        raise PermissionError(f"refusing {target}: only {root}/features* may be deleted")
    if not resolved.is_dir():
        raise PermissionError(f"refusing {target}: not a directory")
    files, dirs = [], []
    for dirpath, dirnames, filenames in os.walk(resolved):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                raise PermissionError(f"refusing {target}: contains symlink {path}")
        dirs.append(Path(dirpath))
        files.extend(Path(dirpath) / name for name in filenames)
    return files, dirs


def delete_feature_batch(target, root=ROOT):
    files, dirs = _check(target, root)
    freed = 0
    for path in files:
        freed += path.stat().st_size
        path.unlink()
    for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        path.rmdir()
    return {"files": len(files), "bytes": freed}


OWN_DIRS = ("smoke/train", "train-full", "sf-cache")


def delete_own_dir(name, root=ROOT):
    """Remove one of phase 7's own scratch directories under .cache/t15, named exactly:
    smoke/train (smoke checkpoints, so the full run keeps at most 2 checkpoints), train-full
    (the full run's checkpoint, once exported) or sf-cache (staged batches and SpecForge
    dataset caches). SpecForge keeps a relative {run_id}-latest link in a train dir; links
    are unlinked (never followed) and must point inside the directory."""
    if name not in OWN_DIRS:
        raise PermissionError(f"refusing {name}: only {OWN_DIRS} under {root} may be deleted")
    root = Path(root).resolve()
    target = root / name
    if target.is_symlink() or not target.is_dir():
        raise PermissionError(f"refusing {target}: not a real directory")
    files, links, dirs = [], [], []
    for dirpath, dirnames, filenames in os.walk(target):
        for entry in dirnames + filenames:
            path = Path(dirpath) / entry
            if path.is_symlink():
                if target not in path.resolve().parents:
                    raise PermissionError(f"refusing {target}: {path} points outside it")
                links.append(path)
            elif entry in filenames:
                files.append(path)
        dirs.append(Path(dirpath))
    freed = 0
    for path in links:
        path.unlink()
    for path in files:
        freed += path.stat().st_size
        path.unlink()
    for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        path.rmdir()
    return {"files": len(files), "links": len(links), "bytes": freed}


if __name__ == "__main__":
    if sys.argv[1] == "--own":
        print(json.dumps({"deleted": str(ROOT / sys.argv[2]), **delete_own_dir(sys.argv[2])}))
    else:
        print(json.dumps({"deleted": sys.argv[1], **delete_feature_batch(sys.argv[1])}))
