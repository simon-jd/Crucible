#!/usr/bin/env bash
# Extend the existing ABI-pinned ML environment; never sync/remove its Torch stack.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/.cache/task7/uv"
export TMPDIR="$PWD/.cache/task7/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface \
  HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  .venv-ml/bin/python - <<'PY'
from importlib import metadata
import json
from pathlib import Path
import subprocess

# Fail before mutation if the reused ABI or prerequisite packages differ from the lock.
for line in Path('requirements/rollout-abi.txt').read_text().splitlines():
    name, expected = line.split('==')
    assert metadata.version(name) == expected, (name, metadata.version(name), expected)
rows = json.loads(Path('docs/evidence/raw/task7-wheel-inventory.json').read_text())
pending = []
for row in rows:
    try:
        present = metadata.version(row['package'])
    except metadata.PackageNotFoundError:
        present = None
    if present != row['version']:
        pending.append(row)
# Reserve final files + one unpack/copy + compressed wheel + compiler caches.
extra = sum(r['uncompressed_bytes'] for r in pending)
staging = max((r['uncompressed_bytes'] + r['bytes'] for r in pending), default=0)
planned = extra + staging + 800_000_000
assert planned <= 6_200_000_000, 'exceeds approved installation peak; re-plan first'
subprocess.run(['.venv/bin/python','scripts/resource_preflight.py',
    '--planned-project-bytes',str(planned),'--download-bytes',
    str(sum(r['bytes'] for r in pending)),'--checkpoints','0',
    '--service-memory-bytes','0'],check=True)
lock = Path('requirements/rollout.lock').read_text()
# Sequential, hash-checked, no persistent wheel cache. Never install a Torch copy.
for row in sorted(pending, key=lambda r: -r['uncompressed_bytes']):
    assert row['sha256'] in lock
    assert row['package'] not in {'torch','triton','numpy'}
    assert not row['package'].startswith('nvidia-')
    subprocess.run(['uv','pip','install','--python','.venv-ml/bin/python',
        '--no-deps','--no-cache','--only-binary',':all:',
        row['url']+'#sha256='+row['sha256']],check=True)
PY
uv pip check --python .venv-ml/bin/python
.venv-ml/bin/python - <<'PY'
from importlib import metadata
import torch
for name in ['vllm','transformers','torch','triton','numpy']:
    print(f'{name}=={metadata.version(name)}')
assert torch.__version__ == '2.8.0+cu128'
assert metadata.version('vllm') == '0.10.2'
assert metadata.version('transformers') == '4.55.4'
PY
