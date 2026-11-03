"""One proposed post-fix host metadata check; no native imports or pool mutation."""
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "docs/evidence/raw"
POOL = ROOT / ".tmp-task12-native-03-pool"
WORK = POOL / "native-cache"
RECEIPT = RAW / "task12-native-pool-fix1-host-receipt.json"
PINS = {'serving/baseline/native_pool.py': '0f02c526c85904bfaa9b01d4255a992b505ea4d7acfce0aad6dd198bd0c5496d', 'serving/baseline/experiment.py': '5d36897a9a6fedd53664cf38b258d6aa7a455a4ad9433a857e81fad1ccc3f596', 'serving/baseline/native_pool_test.py': '057b2bcf748ba22c7ae34a4b47f394e8c7e04b60880c26eae455931ddb3e0556', 'serving/baseline/README.md': '8eb49fabd7c644c32d4081d90c33ee14e5d4e4104f19215e2be911d801276cc6', 'serving/baseline/runtime_entry.py': '4270ba9208228e68841448f49a5b54f92b1e334101d90a66fa70adf225def1cd', 'serving/baseline/model_snapshot.py': 'fbff2794beda16b656fe3f7c23ae02fe15e72393aac8b9dea05b3b7ecf082524', 'serving/baseline/prepare_runtime.py': 'cf5dcdaa8bdb72d28184f9f4bad5bfb6c7b96c1222b52c4dd685d77cc8a601f1', 'scripts/resource_preflight.py': '89deb73ca9915555f2efcabd5ede65bc36347fb729c666c64f8efc8533ec1d23', 'serving/baseline/cache-audit.json': '297a0a25e62a15be1ceb3dfa7dc0e9ef03d665a77de959f31a75ddc1f8385d55', 'docs/evidence/raw/task12-native-pool-manifest.json': '39ea8ed99a4a4262e2d2d8f52f86d6cba4904015752c85d52e9a1868f1c0f1be', 'docs/evidence/raw/task12-native-pool-handoff-receipt.json': 'e324e6e51c013449e20e7b4ba5cf71e4e57817bcef33461b22eae5d8f0ab321d', 'docs/evidence/raw/task12-native-pool-host-receipt.json': '7bbacf755caeed34dbc23dbf77f3a75f0f39c15a12799c7ef0dbaa02bda08536'}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(data):
    encoded = (json.dumps(data, indent=2) + "\n").encode()
    if len(encoded) >= 65536:
        raise ValueError("host receipt exceeds64KiB; no complete evidence")
    with RECEIPT.open("xb") as target:
        target.write(encoded)


def main():
    if RECEIPT.exists() or RECEIPT.is_symlink():
        raise ValueError("fresh host receipt required")
    for name, expected in PINS.items():
        if digest(ROOT / name) != expected:
            raise ValueError("frozen source/input changed: " + name)
    old = json.loads((RAW / "task12-native-pool-host-receipt.json").read_text())
    prepared = json.loads((RAW / "task12-native-pool-handoff-receipt.json").read_text())
    for suffix in ("", "-guard.log", "-launch.json"):
        path = RAW / ("task12-baseline-full-03" + suffix)
        if path.exists() or path.is_symlink():
            raise ValueError("native output/guard/launch must remain absent")
    from serving.baseline.native_pool import NativePool
    from serving.baseline.runtime_entry import runtime_environment
    from serving.baseline.model_snapshot import canonical_bytes

    pool = NativePool(ROOT, RAW / "task12-native-pool-manifest.json", POOL / "replay",
                      RAW / "task12-baseline-full-03")
    if pool.binding != old["binding"]:
        raise ValueError("prepared client/root identity changed")
    directories = {}
    for name, expected in prepared["directory_identities"].items():
        path = WORK if name == "native-cache" else WORK / name
        value = path.lstat()
        identity = [value.st_dev, value.st_ino]
        if not stat.S_ISDIR(value.st_mode) or identity != expected:
            raise ValueError("cache directory identity changed: " + name)
        if name != "native-cache" and any(path.iterdir()):
            raise ValueError("native cache no longer empty: " + name)
        directories[name] = identity
    if {x.name for x in WORK.iterdir()} != set(directories) - {"native-cache"}:
        raise ValueError("unexpected cache directory")
    audit = json.loads((ROOT / "serving/baseline/cache-audit.json").read_text())
    env = runtime_environment(WORK)
    for key, relative in audit["environment"].items():
        env[key] = str(WORK / relative)
    env.update(audit["disable_download_environment"])
    derived = {key: env.get(key) for key in old["derived_native_environment"]}
    inherited = {key: os.environ.get(key) for key in old["inherited_identity_environment"]}
    if derived != old["derived_native_environment"] or inherited != old["inherited_identity_environment"]:
        raise ValueError("safe host environment differs from reviewed43+4 snapshot")
    if any(os.environ.get(key) for key in ("GODEBUG", "GOGC", "GOMEMLIMIT", "GOMAXPROCS",
                                          "GOTRACEBACK", "GO_ENV", "RAYON_NUM_THREADS")):
        raise ValueError("unexpected inherited tool/Go runtime override")
    sample = pool.measure()
    shared = canonical_bytes(Path.home() / ".cache/huggingface")
    space = os.statvfs(ROOT)
    free = space.f_bavail * space.f_frsize
    if shared > 40000000000 or free < 500000000 - sample["pool_bytes"]:
        raise ValueError("shared/free-space prerequisite failed")
    evidence = {"status": "host-metadata-verified", "script_sha256": digest(Path(__file__)),
                "pins": PINS, "binding": pool.binding, "directory_identities": directories,
                "resource_sample": sample, "shared_cache_bytes": shared, "free_bytes": free,
                "derived_native_environment": derived, "inherited_identity_environment": inherited,
                "environment_scope": "current host tool inherited snapshot; no login/credential attestation",
                "client_sha256": pool.binding["client_sha256"],
                "remaining_pool_bytes": 500000000 - sample["pool_bytes"],
                "manifest_sha256": pool.manifest_sha256, "loan_repaid": False,
                "native_executed": False, "native_output_absent": True}
    save(evidence)
    print(json.dumps({"status": evidence["status"], "projected_bytes": sample["projected_bytes"],
                      "native_executed": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # The primary failure stays explicit. No retry, cleanup or native fallback.
        failure = {"status": "failed", "repr": repr(error)[:1024],
                   "pool_scan_history": getattr(error, "_pool_scan_history", None),
                   "native_executed": False}
        try:
            save(failure)
        except Exception as report_error:
            print("receipt publication failed: " + repr(report_error)[:1024])
        print(json.dumps(failure)[:16384])
        raise SystemExit(1)
