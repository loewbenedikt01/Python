"""
Waiting for a database locked by another process. No network.
"""

import duckdb
import pytest

from pipeline import common


def test_lock_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(common.time, 'sleep', lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise duckdb.IOException('IO Error: Cannot open file: The process cannot access the file '
                                     'because it is being used by another process.')
        return 'con'
    assert common.with_lock_retry(fn, 'prices.duckdb', wait=30, max_wait=600) == 'con' and len(calls) == 3


def test_lock_gives_up_after_max_wait(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(common.time, 'sleep', lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(common.time, 'monotonic', lambda: clock[0])

    def fn():
        raise duckdb.IOException('Could not set lock on file')
    with pytest.raises(duckdb.IOException):
        common.with_lock_retry(fn, 'x', wait=30, max_wait=600)
    assert clock[0] == 600                       # 20 waits of 30 s = the full 10 minutes, then it gives up


def test_other_io_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(common.time, 'sleep', lambda s: pytest.fail('should not wait'))

    def fn():
        raise duckdb.IOException('No space left on device')
    with pytest.raises(duckdb.IOException):
        common.with_lock_retry(fn, 'x')
