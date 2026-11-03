"""Temporary Task15 CPU build monitor. Sampling is not an aggregate disk quota."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path.cwd().resolve()
OWNED = ROOT / '.tmp-task15-go-build'
PREFIX = ROOT / 'docs/evidence/raw' / ('task15-cpu-go-' + sys.argv[1])
COMMAND = sys.argv[2:]
LOAN = 500_000_000

def tree_bytes(path):
    total, seen = 0, set()
    for directory, dirs, files in os.walk(path, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        for name in dirs + files:
            item = Path(directory) / name
            try:
                stat = item.lstat()
            except FileNotFoundError:
                continue  # Transient compiler files may disappear during a scan.
            if item.is_symlink():
                if path == OWNED:
                    raise RuntimeError('unexpected owned symlink: ' + str(item))
                continue
            key = (stat.st_dev, stat.st_ino)
            if item.is_file() and key not in seen:
                seen.add(key)
                total += stat.st_size
    return total

def processes():
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(path.name)] = {'state': fields[0], 'group': int(fields[2]),
                                    'start': fields[19], 'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows

def bounds():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (67_108_864, 67_108_864))

def check_disk():
    owned = tree_bytes(OWNED)
    canonical = tree_bytes(ROOT)
    projected = canonical + 190_000_000 + max(0, LOAN - owned)
    if owned >= 400_000_000 or projected > 14_000_000_000:
        raise RuntimeError('disk threshold exceeded')
    if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < max(0, LOAN - owned):
        raise RuntimeError('insufficient free bytes')
    return {'canonical_bytes': canonical, 'owned_bytes': owned, 'projected_bytes': projected}

if not COMMAND or not sys.argv[1].replace('-', '').isalnum():
    raise SystemExit('phase and command required')
if OWNED.is_symlink():
    raise SystemExit('owned root must not be symlink')
for name in ('cache', 'gopath', 'mod', 'tmp'):
    (OWNED / name).mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env.pop('TRANSFORMERS_CACHE', None)
env.update(PYTHONDONTWRITEBYTECODE='1', HF_HOME='/home/postedism/.cache/huggingface',
           HF_HUB_CACHE='/home/postedism/.cache/huggingface/hub',
           HUGGINGFACE_HUB_CACHE='/home/postedism/.cache/huggingface/hub',
           GOENV='off', GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off', GOMAXPROCS='2',
           GOCACHE=str(OWNED / 'cache'), GOPATH=str(OWNED / 'gopath'),
           GOMODCACHE=str(OWNED / 'mod'), GOTMPDIR=str(OWNED / 'tmp'), TMPDIR=str(OWNED / 'tmp'))
record = {'command': COMMAND, 'before': check_disk(), 'samples': [],
          'sampling': 'owned/RSS every 0.25s; canonical every 2s, plus scan time; not a quota',
          'limits': {'owned_abort': 400_000_000, 'loan': LOAN, 'rss_abort': 2_000_000_000,
                     'deadline_seconds': 180, 'file_bytes': 67_108_864, 'core_bytes': 0},
          'environment': {k: v for k, v in env.items() if k.startswith(('GO', 'HF_', 'HUGGINGFACE_', 'TMPDIR'))}}
started = time.monotonic()
with PREFIX.with_suffix('.log').open('wb') as log:
    child = subprocess.Popen(COMMAND, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, preexec_fn=bounds)
    leader_start = processes().get(child.pid, {}).get('start')
    if leader_start is None:
        raise RuntimeError('cannot establish unreaped leader identity')
    record['leader'] = {'pid': child.pid, 'start': leader_start}
    abort, term_time, killed, last_disk = None, None, False, started
    while True:
        now = time.monotonic()
        try:
            rows = processes()
            leader = rows.get(child.pid)
            if leader is None or leader['start'] != leader_start or leader['group'] != child.pid:
                raise RuntimeError('leader ownership lost; signalling permanently forbidden')
            members = {pid: row for pid, row in rows.items() if row['group'] == child.pid}
            live = [pid for pid, row in members.items() if row['state'] != 'Z']
            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None and not live:
                break
            rss = sum(row['rss'] for row in members.values())
            owned = tree_bytes(OWNED)
            sample = {'seconds': now - started, 'owned_bytes': owned, 'rss_bytes': rss}
            if now - last_disk >= 2:
                sample.update(check_disk())
                last_disk = now
            record['samples'].append(sample)
            if abort is None and (owned >= 400_000_000 or rss >= 2_000_000_000 or now-started >= 180):
                abort = 'owned/RSS/deadline threshold exceeded'
            if abort is None and exited is not None and live:
                abort = 'leader exited with live group descendants'
        except Exception as exc:
            abort = str(exc)
        if abort is not None:
            # Keep leader unreaped until group is terminal; check its start identity BEFORE each signal.
            anchor = processes().get(child.pid)
            if anchor is None or anchor['start'] != leader_start or anchor['group'] != child.pid:
                record['ownership_lost'] = True
                break
            if term_time is None:
                os.killpg(child.pid, signal.SIGTERM)
                term_time = now
            elif now-term_time >= 2 and not killed:
                os.killpg(child.pid, signal.SIGKILL)
                killed = True
        time.sleep(0.25)
    if record.get('ownership_lost'):
        record['abort'] = abort
        PREFIX.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit('ownership lost; cleanup forbidden; root intervention required')
    # No more group signals after this reaps the ownership anchor.
    record['exit_code'] = child.wait()
    record['abort'] = abort
    record['elapsed_seconds'] = time.monotonic()-started
    record['live_group_after'] = [pid for pid, row in processes().items()
                                  if row['group'] == child.pid and row['state'] != 'Z']
record['after'] = check_disk()
record['monitor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PREFIX.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
if record['live_group_after']:
    raise SystemExit('live group remains; cleanup forbidden')
raise SystemExit(125 if abort else record['exit_code'])
