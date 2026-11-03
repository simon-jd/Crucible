"""Unexecuted round1 candidate; no future launch approval. Sampling is not a quota."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

from scripts.resource_preflight import project_bytes

ROOT = Path.cwd().resolve()
OWNED = ROOT / '.tmp-task12-sender-phase-go-loan'
PHASE = sys.argv[1]
COMMAND = sys.argv[2:]
PREFIX = ROOT / 'docs/evidence/raw' / ('task12-sender-phase-go-' + PHASE)
LOAN = 500_000_000
record = {'command': COMMAND, 'samples': [], 'scan_errors': [], 'signals': [],
          'sampling': 'requested owned/RSS interval0.25s; canonical2s, plus scan duration; not a quota',
          'limits': {'owned_abort': 400_000_000, 'loan': LOAN, 'rss_abort': 2_000_000_000,
                     'deadline_seconds': 180, 'file_bytes': 67_108_864, 'core_bytes': 0,
                     'log_abort_bytes': 1_048_576},
          'monitor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def persist():
    PREFIX.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')


def scan(label, operation):
    # Only owned/canonical/nonloan FileNotFoundError permits a complete byte rescan.
    # All process/identity and nontransient failures propagate on the first call.
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            record['scan_errors'].append({'scan': label, 'attempt': attempt, 'error': str(exc)})
            if not (isinstance(exc, FileNotFoundError)
                    and label in ('owned', 'canonical', 'nonloan') and attempt < 2):
                raise


def owned_bytes():
    if OWNED.is_symlink():
        raise RuntimeError('owned root is a symlink')
    for directory, dirs, files in os.walk(OWNED, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        for name in dirs + files:
            if (Path(directory) / name).is_symlink():
                raise RuntimeError('unexpected owned symlink')
    return project_bytes(OWNED)


def nonloan_bytes():
    # Exclude only this verified direct child, never a name prefix or inode set.
    if OWNED.parent != ROOT or OWNED.name != '.tmp-task12-sender-phase-go-loan':
        raise RuntimeError('unexpected owned root path')
    identity = OWNED.lstat()
    if not stat.S_ISDIR(identity.st_mode):
        raise RuntimeError('owned root is not a nonsymlink directory')
    expected = (identity.st_dev, identity.st_ino)

    def verify_owned():
        current = OWNED.lstat()
        if (not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != expected):
            raise RuntimeError('owned root identity changed')

    def unreadable(error):
        raise error

    size, seen, excluded = 0, set(), False
    for directory, dirs, files in os.walk(ROOT, followlinks=False, onerror=unreadable):
        if Path(directory) == ROOT and OWNED.name in dirs:
            verify_owned()
            dirs.remove(OWNED.name)
            excluded = True
        for filename in files:
            path = Path(directory) / filename
            if path.is_symlink():
                continue
            info = path.stat()
            inode = (info.st_dev, info.st_ino)
            if inode not in seen:
                size += info.st_size
                seen.add(inode)
    verify_owned()
    if not excluded:
        raise RuntimeError('owned root was not encountered')
    return size


def disk():
    owned = scan('owned', owned_bytes)
    nonloan = scan('nonloan', nonloan_bytes)
    canonical = scan('canonical', lambda: project_bytes(ROOT))
    projected = nonloan + 190_000_000 + LOAN
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    result = {'canonical_bytes': canonical, 'owned_bytes': owned,
              'nonloan_bytes': nonloan, 'projected_bytes': projected, 'free_bytes': free,
              'accounting': 'independent sequential full scans; nonloan excludes only verified owned root; '
                            'cross-boundary hardlinks may double count; non-atomic, not a continuous quota'}
    if (owned >= 400_000_000 or canonical > 14_000_000_000
            or projected > 14_000_000_000 or free < max(0, LOAN-owned)):
        record.setdefault('failed_disk_checks', []).append(result)
        raise RuntimeError('disk threshold exceeded')
    return result


def process(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'state': fields[0], 'group': int(fields[2]), 'start': fields[19],
            'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}


def group(pid):
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            row = process(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue  # An unrelated/owned transient /proc entry has already exited.
        if row['group'] == pid:
            rows[int(path.name)] = row
    return rows


def bounds():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (67_108_864, 67_108_864))


child, anchor_start, abort = None, None, None
started = time.monotonic()
try:
    if not COMMAND or not PHASE.replace('-', '').isalnum():
        raise ValueError('phase and command required')
    if OWNED.is_symlink():
        raise RuntimeError('owned root must not be symlink')
    for name in ('cache', 'gopath', 'mod', 'tmp'):
        (OWNED / name).mkdir(parents=True, exist_ok=True)
    record['before'] = disk()
    env = dict(os.environ)
    env.pop('TRANSFORMERS_CACHE', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HOME='/home/postedism/.cache/huggingface',
               HF_HUB_CACHE='/home/postedism/.cache/huggingface/hub',
               HUGGINGFACE_HUB_CACHE='/home/postedism/.cache/huggingface/hub',
               GOENV='off', GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off', GOFLAGS='', GOMAXPROCS='2',
               GOCACHE=str(OWNED / 'cache'), GOPATH=str(OWNED / 'gopath'),
               GOMODCACHE=str(OWNED / 'mod'), GOTMPDIR=str(OWNED / 'tmp'), TMPDIR=str(OWNED / 'tmp'))
    record['environment'] = {k: env[k] for k in ('PYTHONDONTWRITEBYTECODE', 'HF_HOME', 'HF_HUB_CACHE',
        'HUGGINGFACE_HUB_CACHE', 'GOENV', 'GOTOOLCHAIN', 'GOPROXY', 'GOSUMDB', 'GOFLAGS',
        'GOMAXPROCS', 'GOCACHE', 'GOPATH', 'GOMODCACHE', 'GOTMPDIR', 'TMPDIR')}
    with PREFIX.with_suffix('.log').open('wb') as log:
        child = subprocess.Popen(COMMAND, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=bounds)
        record['pid'] = child.pid
        # Popen has not polled/reaped this leader. If identity cannot be read,
        # retain the unreaped leader and quarantine the root; never guess a PGID.
        anchor = scan('initial_leader', lambda: process(child.pid))
        anchor_start = anchor['start']
        if anchor['group'] != child.pid:
            raise RuntimeError('new child is not its process-group leader')
        record['leader_start'] = anchor_start
        last_disk, previous = started, started
        while True:
            now = time.monotonic()
            anchor = scan('leader', lambda: process(child.pid))
            if anchor['start'] != anchor_start or anchor['group'] != child.pid:
                raise RuntimeError('leader identity changed; quarantine')
            members = scan('group', lambda: group(child.pid))
            live = [pid for pid, row in members.items() if row['state'] != 'Z']
            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None and not live:
                break
            sample = {'seconds': now-started, 'interval_seconds': now-previous,
                      'owned_bytes': scan('owned', owned_bytes),
                      'rss_bytes': sum(row['rss'] for row in members.values()),
                      'log_bytes': PREFIX.with_suffix('.log').stat().st_size}
            previous = now
            record['samples'].append(sample)
            if now-last_disk >= 2:
                sample.update(disk())
                last_disk = now
            if (sample['owned_bytes'] >= 400_000_000 or sample['rss_bytes'] >= 2_000_000_000
                    or sample['log_bytes'] >= 1_048_576 or now-started >= 180):
                raise RuntimeError('owned/RSS/log/deadline threshold exceeded')
            if exited is not None and live:
                raise RuntimeError('leader exited with live descendants')
            time.sleep(0.25)
except BaseException as exc:
    abort = f'{type(exc).__name__}: {exc}'
    record['abort'] = abort
finally:
    if child is not None:
        try:
            if abort is not None:
                if anchor_start is None:
                    raise RuntimeError('startup identity unavailable; cleanup forbidden')
                for sig, delay in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
                    anchor = scan('signal_anchor', lambda: process(child.pid))
                    if anchor['start'] != anchor_start or anchor['group'] != child.pid:
                        raise RuntimeError('ownership lost; signalling permanently forbidden')
                    members = scan('terminal_group', lambda: group(child.pid))
                    if not any(row['state'] != 'Z' for row in members.values()):
                        break
                    os.killpg(child.pid, sig)
                    record['signals'].append({'signal': sig.name, 'seconds': time.monotonic()-started})
                    deadline = time.monotonic()+delay
                    while time.monotonic() < deadline:
                        members = scan('terminal_group', lambda: group(child.pid))
                        if not any(row['state'] != 'Z' for row in members.values()):
                            break
                        time.sleep(0.05)
            members = scan('terminal_group', lambda: group(child.pid))
            if any(row['state'] != 'Z' for row in members.values()):
                raise RuntimeError('live owned group remains; cleanup forbidden')
            # No group signals after reap: PID/PGID reuse can no longer be ruled out.
            record['exit_code'] = child.wait(timeout=1)
            record['terminal_group_proved_before_reap'] = True
        except BaseException as exc:
            record['quarantined'] = True
            record['cleanup_error'] = f'{type(exc).__name__}: {exc}'
    record['elapsed_seconds'] = time.monotonic()-started
    try:
        record['after'] = disk()
    except BaseException as exc:
        record['final_account_error'] = f'{type(exc).__name__}: {exc}'
    persist()
if abort or record.get('quarantined') or record.get('final_account_error'):
    raise SystemExit(125)
raise SystemExit(record.get('exit_code', 125))
