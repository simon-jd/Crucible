"""Exact owned terminal Task15 cleanup; lstat links, never traverse them."""
import hashlib
import json
import shutil
import stat
from pathlib import Path

root = Path.cwd().resolve()
raw = root / 'docs/evidence/raw'
phases = []
for phase in ('smoke', 'red', 'green', 'vet'):
    value = json.loads((raw / f'task15-cpu-go-{phase}.json').read_text())
    assert value.get('terminal_group_proved_before_reap') is True
    assert not value.get('quarantined') and not value.get('abort') and not value.get('final_account_error')
    phases.append({'phase': phase, 'exit_code': value['exit_code'],
                   'terminal_group_proved_before_reap': True,
                   'elapsed_seconds': value['elapsed_seconds'],
                   'max_sampled_owned_bytes': max([s['owned_bytes'] for s in value['samples']] + [value['after']['owned_bytes']]),
                   'max_sampled_rss_bytes': max([s['rss_bytes'] for s in value['samples']] + [0]),
                   'max_observed_canonical_bytes': max([s.get('canonical_bytes', 0) for s in value['samples']] + [value['before']['canonical_bytes'], value['after']['canonical_bytes']]),
                   'samples': len(value['samples']), 'scan_errors': value['scan_errors']})
assert '53 passed' in (raw / 'task15-cpu-python-green.log').read_text()
assert '1 error in' in (raw / 'task15-cpu-python-red.log').read_text()
paths = [root / name for name in ('.tmp-task15-go-build', '.tmp-task15-go-monitor.py',
                                  '.tmp-task15-py-red', '.tmp-task15-py-green')]
known_link = root / '.tmp-task15-py-green/test_cli_json_errors_are_striccurrent'
known_target = root / '.tmp-task15-py-green/test_cli_json_errors_are_stric0'
entries = []
links = []
for path in paths:
    assert path.parent == root and path.name.startswith('.tmp-task15-')
    if not path.exists() and not path.is_symlink():
        entries.append({'root': path.name, 'absent_before': True, 'files': [], 'symlinks': []})
        continue
    assert not path.is_symlink()
    items = sorted(path.rglob('*')) if path.is_dir() else [path]
    files, symlinks = [], []
    for item in items:
        details = item.lstat()
        relative = str(item.relative_to(root))
        if stat.S_ISLNK(details.st_mode):
            target = item.readlink()
            assert item == known_link and target == known_target
            assert stat.S_ISDIR(known_target.lstat().st_mode)
            symlinks.append({'path': relative, 'lstat_bytes': details.st_size, 'target': str(target),
                             'target_inside_same_owned_root': True})
            links.append(item)
        elif stat.S_ISREG(details.st_mode):
            files.append({'path': relative, 'bytes': details.st_size,
                          'sha256': hashlib.sha256(item.read_bytes()).hexdigest()})
        elif not stat.S_ISDIR(details.st_mode):
            raise RuntimeError('unexpected owned file type: ' + relative)
    entries.append({'root': path.name, 'files': files, 'symlinks': symlinks,
                    'logical_file_bytes': sum(item['bytes'] for item in files)})
receipt = {'scope': 'exact owned terminal Task15 CPU roots only', 'terminal_proofs': phases,
           'pytest_terminal_logs': ['task15-cpu-python-red.log', 'task15-cpu-python-green.log'],
           'inventory': entries}
output = raw / 'task15-cpu-owned-cleanup.json'
output.write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
for link in links:
    link.unlink()  # Delete link itself, never follow its target.
for path in paths:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
receipt['absence_after'] = {p.name: not p.exists() and not p.is_symlink() for p in paths}
assert all(receipt['absence_after'].values())
output.write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
print(json.dumps({'removed_regular_bytes': sum(e.get('logical_file_bytes', 0) for e in entries),
                  'removed_symlinks': len(links), 'absence_after': receipt['absence_after']}, indent=2))
