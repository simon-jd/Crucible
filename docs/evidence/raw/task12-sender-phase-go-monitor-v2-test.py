"""Function-only accounting regressions; never execute the monitor module body."""
import ast
import os
from pathlib import Path
import stat

import pytest

from scripts.resource_preflight import project_bytes

SOURCE = Path(__file__).with_name('task12-sender-phase-go-monitor-v2.py')


@pytest.fixture
def account(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    owned = root / '.tmp-task12-sender-phase-go-loan'
    owned.mkdir()
    tree = ast.parse(SOURCE.read_text())
    names = {'scan', 'owned_bytes', 'nonloan_bytes', 'disk'}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    scope = dict(Path=Path, os=os, stat=stat, ROOT=root, OWNED=owned,
                 LOAN=500_000_000, record={'scan_errors': []}, project_bytes=project_bytes)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


def test_growing_loan_does_not_mix_canonical_minus_stale_owned(account):
    root, loan = account['ROOT'], account['OWNED']
    (root / 'outside').write_bytes(b'x' * 31)
    payload = loan / 'growing'
    for size in (1, 100, 1000):
        payload.write_bytes(b'x' * size)
        assert account['nonloan_bytes']() == 31
    # Exact historical failed scan, not a fabricated success expectation.
    old_owned, later_canonical = 8_337_419, 13_318_870_180
    assert later_canonical + 190_000_000 + 500_000_000 - old_owned > 14_000_000_000
    account['owned_bytes'] = lambda: old_owned
    account['nonloan_bytes'] = lambda: 13_297_672_660
    account['project_bytes'] = lambda root: later_canonical
    result = account['disk']()
    assert result['projected_bytes'] == 13_987_672_660
    assert result['canonical_bytes'] == later_canonical


def test_hidden_siblings_hardlinks_and_symlinks(account):
    root, loan = account['ROOT'], account['OWNED']
    sibling = root / (loan.name + '-similar')
    sibling.mkdir()
    (sibling / 'payload').write_bytes(b'x' * 11)
    (root / '.git').mkdir()
    (root / '.git' / 'hidden').write_bytes(b'x' * 13)
    first = root / 'first'
    first.write_bytes(b'x' * 17)
    os.link(first, root / 'same-inode')
    cross = loan / 'cross'
    cross.write_bytes(b'x' * 19)
    os.link(cross, root / 'cross-outside')
    (root / 'link-file').symlink_to(first)
    (root / 'link-dir').symlink_to(sibling, target_is_directory=True)
    assert account['nonloan_bytes']() == 11 + 13 + 17 + 19
    assert project_bytes(root) == 60
    # Outside is never excluded because its inode also exists inside the loan.
    assert account['nonloan_bytes']() + account['owned_bytes']() == 79


@pytest.mark.parametrize('kind', ['symlink', 'wrong-name'])
def test_only_verified_exact_owned_root_can_be_pruned(account, kind):
    if kind == 'symlink':
        account['OWNED'].rmdir()
        account['OWNED'].symlink_to(account['ROOT'], target_is_directory=True)
    else:
        account['OWNED'] = account['ROOT'] / 'another-directory'
        account['OWNED'].mkdir()
    with pytest.raises(RuntimeError):
        account['nonloan_bytes']()


def test_transient_nonloan_enoent_restarts_whole_scan(account, monkeypatch):
    (account['ROOT'] / 'payload').write_bytes(b'1234567')
    original = os.walk
    calls = []

    def walk(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            yield from original(*args, **kwargs)
            raise FileNotFoundError(2, 'injected after first full enumeration', 'gone')
        yield from original(*args, **kwargs)

    monkeypatch.setattr(os, 'walk', walk)
    assert account['scan']('nonloan', account['nonloan_bytes']) == 7
    assert calls == [account['ROOT'], account['ROOT']]
    assert len(account['record']['scan_errors']) == 1


@pytest.mark.parametrize('label,error,attempts', [
    ('nonloan', FileNotFoundError(2, 'gone'), 3),
    ('nonloan', PermissionError(13, 'denied'), 1),
    ('nonloan', OSError(5, 'I/O'), 1),
    ('leader', FileNotFoundError(2, 'gone'), 1),
])
def test_rescan_limit_and_other_errors_propagate(account, label, error, attempts):
    calls = []

    def fail():
        calls.append(1)
        raise error

    with pytest.raises(type(error)) as caught:
        account['scan'](label, fail)
    assert caught.value is error
    assert len(calls) == attempts
    assert len(account['record']['scan_errors']) == attempts


def test_full_canonical_cap_is_independent(account):
    account['owned_bytes'] = lambda: 1
    account['nonloan_bytes'] = lambda: 1
    account['project_bytes'] = lambda root: 14_000_000_001
    with pytest.raises(RuntimeError, match='disk threshold exceeded'):
        account['disk']()
    assert account['record']['failed_disk_checks'][0]['canonical_bytes'] == 14_000_000_001
