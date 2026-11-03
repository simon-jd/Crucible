"""Proposed exact Task12 handoff/host metadata checks; no native execution."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / '.tmp-task12-native-03-pool'
WORK = POOL / 'native-cache'
BINARY = POOL / 'replay'
RAW = ROOT / 'docs/evidence/raw'
MANIFEST = RAW / 'task12-native-pool-manifest.json'
OUTPUT = RAW / 'task12-baseline-full-03'
CLIENT_SHA = 'ee968c696cb476a5aae97537991adea237400626518dfe34229fc346be5c855b'
DIRECTORIES = ('cuda', 'datasets', 'deep-gemm', 'flashinfer', 'hf-modules', 'inductor',
               'offline-cubin-source', 'outlines', 'tmp', 'torch', 'torch-home',
               'triton', 'tvm-ffi', 'xdg')
PINS = {'serving/baseline/native_pool.py': '2526489bd9493bdaf8c364efb297a3846d2e0a6da221aee64a7474846dab9a90',
 'serving/baseline/runtime_entry.py': '4270ba9208228e68841448f49a5b54f92b1e334101d90a66fa70adf225def1cd',
 'serving/baseline/model_snapshot.py': 'fbff2794beda16b656fe3f7c23ae02fe15e72393aac8b9dea05b3b7ecf082524',
 'serving/baseline/prepare_runtime.py': 'cf5dcdaa8bdb72d28184f9f4bad5bfb6c7b96c1222b52c4dd685d77cc8a601f1',
 'scripts/resource_preflight.py': '89deb73ca9915555f2efcabd5ede65bc36347fb729c666c64f8efc8533ec1d23',
 'serving/baseline/cache-audit.json': '297a0a25e62a15be1ceb3dfa7dc0e9ef03d665a77de959f31a75ddc1f8385d55'}
DERIVED_KEYS = ('CUDA_CACHE_MAXSIZE',
 'CUDA_CACHE_PATH',
 'CUDA_HOME',
 'CUTE_DSL_ARCH',
 'CUTE_DSL_DISABLE_FILE_CACHING',
 'CUTE_DSL_KEEP_IR',
 'CUTE_DSL_LOG_TO_FILE',
 'FLASHINFER_CUBINS_REPOSITORY',
 'FLASHINFER_DISABLE_JIT',
 'FLASHINFER_WORKSPACE_BASE',
 'HF_DATASETS_CACHE',
 'HF_DATASETS_OFFLINE',
 'HF_HOME',
 'HF_HUB_CACHE',
 'HF_HUB_DISABLE_XET',
 'HF_HUB_ENABLE_HF_TRANSFER',
 'HF_HUB_OFFLINE',
 'HF_MODULES_CACHE',
 'HUGGINGFACE_HUB_CACHE',
 'MAX_JOBS',
 'OUTLINES_CACHE_DIR',
 'PATH',
 'PYTHONDONTWRITEBYTECODE',
 'PYTHONNOUSERSITE',
 'SGLANG_DG_CACHE_DIR',
 'SGLANG_USE_MODELSCOPE',
 'TMPDIR',
 'TOKENIZERS_PARALLELISM',
 'TORCHINDUCTOR_CACHE_DIR',
 'TORCHINDUCTOR_COMPILE_THREADS',
 'TORCH_EXTENSIONS_DIR',
 'TORCH_HOME',
 'TRANSFORMERS_OFFLINE',
 'TRITON_CACHE_DIR',
 'TVM_FFI_CACHE_DIR',
 'TVM_FFI_DISABLE_TORCH_C_DLPACK',
 'UV_OFFLINE',
 'XDG_CACHE_HOME',
 'XDG_DATA_DIRS',
 'XDG_RUNTIME_DIR',
 'XDG_SESSION_CLASS',
 'XDG_SESSION_ID',
 'XDG_SESSION_TYPE')
INHERITED_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def identity(path, directory=False):
    s = path.lstat()
    if not (stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode)):
        raise ValueError('unexpected identity type: ' + str(path))
    return [s.st_dev, s.st_ino]


def save(path, value):
    data = (json.dumps(value, indent=2) + '\n').encode()
    if len(data) >= 65536:
        raise ValueError('handoff evidence limit')
    with path.open('xb') as target:
        target.write(data)


def main(mode):
    if mode not in ('prepare', 'host'):
        raise ValueError('exact prepare or host mode required')
    for name, expected in PINS.items():
        if digest(ROOT / name) != expected:
            raise ValueError('source changed: ' + name)
    if identity(POOL, True) != [66307, 55447049]:
        raise ValueError('pool replaced')
    if identity(BINARY) != [66307, 55468959]:
        raise ValueError("client inode replaced")
    if BINARY.stat().st_size != 10160231 or digest(BINARY) != CLIENT_SHA:
        raise ValueError('client changed')
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('native output must remain fresh')
    # Imports below are the pinned pure metadata/preparation paths only.
    from serving.baseline.native_pool import NativePool
    from serving.baseline.runtime_entry import runtime_environment
    audit = json.loads((ROOT / 'serving/baseline/cache-audit.json').read_text())
    env = runtime_environment(WORK)
    for key, relative in audit['environment'].items():
        env[key] = str(WORK / relative)
    env.update(audit['disable_download_environment'])
    if env.get('TORCH_HOME') != str(WORK / 'torch-home'):
        raise ValueError('TORCH_HOME must use this pool')
    derived_paths = {Path(v) for v in env.values() if isinstance(v, str) and v.startswith(str(WORK) + '/')}
    expected_paths = {WORK / name for name in DIRECTORIES}
    if derived_paths != expected_paths:
        raise ValueError('derived directory set differs from reviewed list')
    if mode == 'prepare':
        if {p.name for p in POOL.iterdir()} != {'replay'} or MANIFEST.exists() or MANIFEST.is_symlink():
            raise ValueError('handoff needs exact client-only pool and fresh manifest')
        WORK.mkdir()
        for name in DIRECTORIES:
            (WORK / name).mkdir()
        value = BINARY.stat()
        binding = {'version': 1, 'root': str(POOL), 'root_identity': identity(POOL, True),
                   'work_identity': identity(WORK, True), 'client_identity': identity(BINARY),
                   'client_bytes': value.st_size, 'client_mtime_ns': value.st_mtime_ns,
                   'client_sha256': CLIENT_SHA}
        save(MANIFEST, binding)
    else:
        previous = json.loads((RAW / 'task12-native-pool-handoff-receipt.json').read_text())
        if digest(MANIFEST) != previous['manifest_sha256']:
            raise ValueError('manifest differs from prepared identity')
        if identity(WORK, True) != previous['directory_identities']['native-cache']:
            raise ValueError('host work identity differs')
    if {p.name for p in WORK.iterdir()} != set(DIRECTORIES):
        raise ValueError('unexpected native work layout')
    dirs = {'native-cache': identity(WORK, True)}
    for name in DIRECTORIES:
        dirs[name] = identity(WORK / name, True)
        if any((WORK / name).iterdir()):
            raise ValueError('handoff cache must remain empty')
        if mode == 'host' and dirs[name] != previous['directory_identities'][name]:
            raise ValueError('host cache identity differs')
    pool = NativePool(ROOT, MANIFEST, BINARY, OUTPUT)
    sample = pool.measure(canonical=True)
    from serving.baseline.model_snapshot import canonical_bytes
    shared_bytes = canonical_bytes(Path.home() / '.cache/huggingface')
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if shared_bytes > 40_000_000_000 or free_bytes < 500_000_000-sample['pool_bytes']:
        raise ValueError('shared/free resource prerequisite')
    if any(os.environ.get(key) for key in ('GODEBUG','GOGC','GOMEMLIMIT','GOMAXPROCS','GOTRACEBACK','GO_ENV')):
        raise ValueError('unexpected inherited Go runtime override')
    evidence = {'mode': mode, 'script_sha256': digest(Path(__file__)), 'pins': PINS,
                'manifest_sha256': digest(MANIFEST), 'binding': pool.binding,
                'directory_identities': dirs, 'resource_sample': sample,
                'shared_cache_bytes': shared_bytes, 'free_bytes': free_bytes,
                'client_bytes': BINARY.stat().st_size, 'client_sha256': CLIENT_SHA,
                'remaining_pool_bytes': 500000000-BINARY.stat().st_size,
                'derived_native_environment': {key: env.get(key) for key in DERIVED_KEYS},
                'inherited_identity_environment': {key: os.environ.get(key) for key in INHERITED_KEYS},
                'environment_scope': 'only reviewed43 derived keys +4 inherited identity keys; host mode is authoritative',
                'native_output_absent': True, 'native_executed': False,
                'loan_repaid': False}
    save(RAW / ('task12-native-pool-handoff-receipt.json' if mode == 'prepare'
                else 'task12-native-pool-host-receipt.json'), evidence)
    print(json.dumps({'mode':mode,'manifest_sha256':evidence['manifest_sha256'],
                      'projected_bytes':sample['projected_bytes'],'native_executed':False}))


if __name__ == '__main__':
    try:
        main(sys.argv[1] if len(sys.argv)==2 else '')
    except Exception as error:
        print(repr(error).encode('utf-8','replace')[:1024].decode('utf-8','replace'))
        raise SystemExit(1)
