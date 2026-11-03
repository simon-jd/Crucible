"""One proposed CPU phase screen; import is inert, execution requires its gate."""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat

from scripts.resource_preflight import project_bytes

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'docs/evidence/raw'
LOAN = ROOT / '.tmp-task12-wait-go-loan'
OUT = RAW / 'task12-wait-cpu-01'
FIXTURE = RAW / 'task12-sender-cpu-01-fixture.py'
MONITOR = RAW / 'task12-wait-go-monitor.py'
BINARY = '.tmp-task12-wait-go-loan/replay-wait'
PINS = {'docs/evidence/raw/task12-sender-cpu-01-fixture.py': 'ad7f88b786fd0bc8e938855f7ab867f6e96ef0e6393e377162396290f576ea24',
 'docs/evidence/raw/task12-wait-go-monitor.py': '8d06f5ca394d81e81ee419f8b4e91ac6cf4365ede7244b5773809564b3eb524c',
 '.tmp-task12-wait-go-loan/replay-wait': '33105f112395f2dda307979f169395b6ff9d3f3d56d4b20e492b0d3fa241e210',
 'serving/replay/schedule.go': '5298bdd52aa578a55775fc38d003380563caf437947059babf6feb71ea1ca7e2',
 'serving/replay/transport.go': '7f8f876dc930c75a352f5e12d465283ba38ec11d65bb954afa64d385be7cb34b',
 'serving/replay/diagnostic.go': '4c781f1a17627f968c21490c6b90bc5a9da78ccfa7ff9b7cfa61d6e0e201b02f',
 'serving/replay/wait.go': 'c92ccf294510b6f8964ab0179e14b6af1834be1f948119f23ab2ead2ca9a7ee3',
 'serving/replay/cmd/replay/main.go': 'aed7288c3e9367797598aa6f4d98cbfad3af771818ae6a6fb4bed543f3904670'}


def read_bounded(path, cap):
    if path.is_symlink():
        raise ValueError('symlink input')
    with path.open('rb') as inp:
        data = inp.read(cap + 1)
    if len(data) > cap:
        raise ValueError('input byte bound')
    return data


def accounting():
    # Reuse precisely the reviewed accounting functions, never its launch body.
    tree = ast.parse(MONITOR.read_text())
    names = {'scan', 'owned_bytes', 'nonloan_bytes', 'disk'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('accounting function identity')
    scope = dict(Path=Path, os=os, stat=stat, ROOT=ROOT, OWNED=LOAN,
                 LOAN=500_000_000, record={'scan_errors': []}, project_bytes=project_bytes)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MONITOR), 'exec'), scope)
    result = scope['disk']()
    result['scan_errors'] = scope['record']['scan_errors']
    return result


def phase_analysis(capture, raw, expected):
    if capture.get('version') != 2 or capture.get('capture_valid') is not True or capture.get('problems'):
        raise ValueError('capture reports invalid or unknown version')
    requests = capture['requests']
    records = raw['Run']['Records']
    if len(requests) != 23 or len(records) != 23:
        raise ValueError('phase/raw count')
    indices = [r['index'] for r in requests]
    if any(type(i) is not int for i in indices) or set(indices) != set(range(23)):
        raise ValueError('phase index duplicate/missing')
    ordered = sorted(expected.values(), key=lambda r: r['index'])
    result, connections = [], []
    allowed = {'post_select', 'timer_stopped', 'no_timer', 'send_entry', 'get_conn',
               'connect_start', 'connect_done', 'got_conn', 'write_entry', 'write_return'}
    for request in sorted(requests, key=lambda r: r['index']):
        i = request['index']
        events = request['events']
        if request.get('problems') or not 1 <= len(events) <= 48:
            raise ValueError('request capture problems/bound')
        intermediate = request.get('intermediate_wakes')
        if type(intermediate) is not int or not 0 <= intermediate <= 65535:
            raise ValueError('intermediate wake counter type/bound')
        phases = {}
        for position, event in enumerate(events):
            name, at = event['phase'], event['at_ns']
            if name not in allowed or type(at) is not int or at < 0 or event.get('error'):
                raise ValueError('phase type/time/error')
            phases.setdefault(name, []).append((position, event))
        def one(name):
            rows = phases.get(name, [])
            if len(rows) != 1:
                raise ValueError('missing/repeated ' + name)
            return rows[0]
        timer = ['no_timer'] if 'no_timer' in phases else ['post_select', 'timer_stopped']
        if timer == ['no_timer'] and ('post_select' in phases or 'timer_stopped' in phases):
            raise ValueError('mixed timer path')
        if timer == ['no_timer'] and intermediate != 0:
            raise ValueError('no-timer path has intermediate wakes')
        got = one('got_conn')[1]
        conn = got.get('connection')
        if type(conn) is not int or conn <= 0:
            raise ValueError('connection identity')
        connects = ['connect_start', 'connect_done'] if 'connect_start' in phases or 'connect_done' in phases else []
        if not got.get('reused', False) and not connects:
            raise ValueError('missing fresh connect')
        chain = timer + ['send_entry', 'get_conn'] + connects + ['got_conn']
        selected = [one(name) for name in chain]
        if any(a[0] >= b[0] or a[1]['at_ns'] > b[1]['at_ns'] for a, b in zip(selected, selected[1:])):
            raise ValueError('phase position/time order')
        entries, returns = phases.get('write_entry', []), phases.get('write_return', [])
        if not 1 <= len(entries) <= 16 or len(entries) != len(returns):
            raise ValueError('write pair count')
        writes = []
        last_return = got['at_ns']
        for number, ((ep, entry), (rp, returned)) in enumerate(zip(entries, returns), 1):
            length = entry.get('length')
            if (entry.get('write') != number or returned.get('write') != number
                    or type(length) is not int or length <= 0
                    or returned.get('length') != length or returned.get('n') != length
                    or entry.get('connection') != conn or returned.get('connection') != conn
                    or ep >= rp or ep <= one('got_conn')[0]
                    or entry['at_ns'] < last_return or returned['at_ns'] < entry['at_ns']):
                raise ValueError('write identity/order/partial emission')
            last_return = returned['at_ns']
            writes.append({'write': number, 'length': length, 'duration_ns': last_return-entry['at_ns']})
        sent = records[i]['SentMs'] * 1_000_000
        if not math.isfinite(sent) or abs(sent - last_return) > 1:
            raise ValueError('write finished differs from original SentMs')
        if sum(w['length'] for w in writes) < ordered[i]['bytes']:
            raise ValueError('write payload smaller than prepared body')
        intervals = {a + '_to_' + b + '_ns': one(b)[1]['at_ns'] - one(a)[1]['at_ns']
                     for a, b in zip(chain, chain[1:])}
        intervals['got_conn_to_first_write_ns'] = entries[0][1]['at_ns'] - got['at_ns']
        intervals['scheduled_to_' + timer[0] + '_ns'] = one(timer[0])[1]['at_ns'] - round(records[i]['ScheduledMs']*1_000_000)
        connections.append((got['at_ns'], conn, got.get('reused', False)))
        result.append({'index': i, 'intermediate_wakes': intermediate, 'connection': conn, 'reused': got.get('reused', False),
                       'was_idle': got.get('was_idle', False), 'idle_ns': got.get('idle_ns', 0),
                       'lag_ms': records[i]['SentMs']-records[i]['ScheduledMs'],
                       'intervals': intervals, 'writes': writes})
    seen = set()
    for _, conn, reused in sorted(connections):
        if (conn in seen) != reused:
            raise ValueError('connection reuse identity')
        seen.add(conn)
    return result


def main():
    os.chdir(ROOT)
    for name, sha in PINS.items():
        if hashlib.sha256(read_bounded(ROOT/name, 16 << 20)).hexdigest() != sha:
            raise ValueError('frozen input changed: ' + name)
    spec = importlib.util.spec_from_file_location('task12_frozen_fixture', FIXTURE)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    if any(os.environ.get(name) for name in (*fixture.GO_ENV, 'GO_ENV')):
        raise ValueError('unexpected Go runtime override')
    inputs = {}
    for name, sha in fixture.PINNED.items():
        data = read_bounded(ROOT/name, 16 << 20)
        if fixture.digest(data) != sha:
            raise ValueError('historical fixture input changed')
        inputs[name] = data
    tokens = json.loads(inputs[fixture.TOKENS])['token_ids']
    expected = fixture.prepare(inputs[fixture.TRACE], tokens)
    if OUT.exists() or OUT.is_symlink():
        raise ValueError('output must be fresh')
    before = accounting()
    # All fixture hard caps + sidecar + phase-terminal + outer log <=1272KiB (includes a retained sidecar partial).
    # Count all phase-prefixed raw (including root receipts) conservatively.
    retained = sum(p.stat().st_size for p in RAW.glob('task12-wait-*') if p.is_file())
    retained += sum(p.stat().st_size for p in (ROOT/'serving/replay').rglob('diagnostic*.go'))
    retained += sum((ROOT/p).stat().st_size for p in ('serving/replay/schedule.go',
                    'serving/replay/wait.go', 'serving/replay/wait_test.go',
                    '.superpowers/sdd/5-crucible-gpu-ml-plan/task-12-wait-report.md'))
    if retained + 1272*1024 + 270_000 + 65_536 > 2_000_000:
        raise ValueError('phase retained evidence allowance')
    command = [BINARY, '--trace', fixture.TRACE, '--format', 'burstgpt', '--url',
               'http://127.0.0.1:31013/generate', '--token-ids', fixture.TOKENS,
               '--mult', '1', '--timeout', '60s', '--inflight', '192', '--seed', '17',
               '--phase-sidecar', str(OUT/'phases.json')]
    provenance = {'driver_sha256': fixture.digest(Path(__file__).read_bytes()), 'pins': PINS,
                  'inputs': fixture.PINNED, 'go_environment': {n: os.environ.get(n) for n in (*fixture.GO_ENV, 'GO_ENV')},
                  'accounting_before': before, 'retained_before': retained, 'cleanup_inventory_holdback': 270_000,
                  'root_reporting_holdback': 65_536,
                  'runtime': 'ordinary inherited host Go runtime; no build GOMAXPROCS override',
                  'bodies': [{k:v for k,v in row.items() if k != 'body'} | {'sha256':key}
                             for key,row in expected.items()]}
    terminal = fixture.screen(command, expected, OUT, provenance=provenance)
    result = {'diagnostic_valid': False, 'native_performance_accepted': False,
              'fixture_screen_completed': terminal['screen_completed'],
              'limitation': 'same-run instrumented CPU phases only; hooks perturb timing; no historical kernel attribution; v2 last actual wake may precede due when Stop/bookkeeping crosses due; intermediate wakes counted separately'}
    try:
        if not terminal['screen_completed']:
            raise ValueError('fixture screen invalid: ' + str(terminal['failure']))
        data = read_bounded(OUT/'phases.json', 60 << 10)
        raw = json.loads(read_bounded(OUT/'stdout.json', 65536))
        expected_config = dict(URL='http://127.0.0.1:31013/generate', Multiplier=1,
                               MaxInflight=192, Timeout=60_000_000_000, TokenIDs=tokens, Seed=17)
        if raw['Config'] != expected_config:
            raise ValueError('changed original replay config')
        result['requests'] = phase_analysis(json.loads(data), raw, expected)
        result['phase_sha256'] = fixture.digest(data)
        result['sender_valid'] = raw['Run']['SenderValid']
        result['diagnostic_valid'] = True
    except Exception as exc:
        result['error'] = fixture.error_text(exc)
    try:
        result['accounting_after'] = accounting()
    except Exception as exc:
        result['diagnostic_valid'] = False
        result['accounting_error'] = fixture.error_text(exc)
    fixture.save(OUT/'phase-terminal.json', result)
    return 0 if result['diagnostic_valid'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(repr(error).encode('utf-8', 'replace')[:2048].decode('utf-8', 'replace'))
        raise SystemExit(1)
