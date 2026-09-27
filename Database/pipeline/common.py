"""
Shared helpers: API keys, logging, rate limits, HTTP, DuckDB connections,
upserts, the load_log and reading the _tickers files.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import duckdb
import pandas as pd
import requests

import config


# ----
# API KEYS
# ----

_KEYS: dict[str, str] | None = None


def load_keys() -> dict[str, str]:
    """
    KEY=value lines from api_keys.txt. Values are never printed or logged.
    """
    global _KEYS
    if _KEYS is None:
        keys = {}
        if config.API_KEYS_FILE.exists():
            for line in config.API_KEYS_FILE.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    keys[k.strip()] = v.strip()
        _KEYS = keys
    return _KEYS


def get_key(name: str) -> str:
    value = load_keys().get(name)
    if not value:
        raise KeyError(f'{name} is missing in {config.API_KEYS_FILE.name}')
    return value


def sec_user_agent() -> str:
    return f"{get_key('NAME')} {get_key('EMAIL')}"


# ----
# LOGGING
# ----

def redact(text: str) -> str:
    """
    Replace every API key value in `text` with ***.
    """
    for name, value in load_keys().items():
        if name != 'NAME' and value and len(value) >= 6 and value in text:
            text = text.replace(value, '***')
    return text


class _RedactKeys(logging.Filter):
    """
    Replace any API key value that ends up in a log message with ***.
    """
    def filter(self, record):
        record.msg, record.args = redact(record.getMessage()), None
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


def get_logger(name: str = 'pipeline') -> logging.Logger:
    logger = logging.getLogger(name)
    root = logging.getLogger('pipeline')
    if not root.handlers:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s %(levelname)-7s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        file = logging.FileHandler(config.LOG_DIR / 'pipeline.log', encoding='utf-8')
        file.setFormatter(fmt)
        for h in (console, file):
            h.addFilter(_RedactKeys())
            root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


# ----
# RATE LIMITS / HTTP
# ----

class RateLimiter:
    """
    Thread-safe sliding window: at most `calls` per `period` seconds.
    """
    def __init__(self, calls: int, period: float):
        self.calls, self.period = calls, period
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.period:
                    self._times.popleft()
                if len(self._times) < self.calls:
                    self._times.append(now)
                    return
                sleep = self.period - (now - self._times[0])
            time.sleep(max(sleep, 0.01))


_LIMITERS: dict[str, RateLimiter] = {}
_SESSION = requests.Session()


def limiter(service: str) -> RateLimiter:
    if service not in _LIMITERS:
        calls, period = config.RATE_LIMITS[service]
        _LIMITERS[service] = RateLimiter(calls, period)
    return _LIMITERS[service]


def http_get(url: str, service: str, params: dict | None = None,
             headers: dict | None = None, stream: bool = False) -> requests.Response:
    """
    GET with the service's rate limit and retries (backoff on 429 / 5xx / network errors).
    Raises after the last retry; 4xx other than 429 raise immediately.
    """
    headers = dict(headers or {})
    if service == 'sec':
        headers.setdefault('User-Agent', sec_user_agent())
    delay = config.HTTP_BACKOFF
    for attempt in range(config.HTTP_RETRIES + 1):
        limiter(service).wait()
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT, stream=stream)
            if r.status_code >= 400:
                # own message without the query string: it may contain an API token
                raise requests.HTTPError(f'HTTP {r.status_code} from {service} ({url})', response=r)
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt == config.HTTP_RETRIES:
                raise
            wait = delay
            retry_after = getattr(getattr(e, 'response', None), 'headers', {}).get('Retry-After')
            if retry_after and str(retry_after).isdigit():
                wait = max(wait, int(retry_after))
            time.sleep(wait)
            delay *= 2
    raise RuntimeError('unreachable')


# ----
# DUCKDB
# ----

DATABASES = {
    'prices':    config.PRICES_DB,
    'companies': config.COMPANIES_DB,
    'raw':       config.RAW_DB,
}


def is_lock_error(e: Exception) -> bool:
    """
    The file is held by another process (Streamlit app, notebook, a second run).
    """
    msg = str(e).lower()
    return isinstance(e, duckdb.IOException) and ('lock' in msg or 'used by another process' in msg)


def with_lock_retry(fn, what: str, wait: float | None = None, max_wait: float | None = None):
    """
    Call fn(); while the database is locked by another process, log it and retry every
    DB_LOCK_RETRY_SECONDS for up to DB_LOCK_MAX_WAIT_SECONDS, then raise.
    """
    wait = config.DB_LOCK_RETRY_SECONDS if wait is None else wait
    max_wait = config.DB_LOCK_MAX_WAIT_SECONDS if max_wait is None else max_wait
    start = time.monotonic()
    while True:
        try:
            return fn()
        except duckdb.IOException as e:
            waited = time.monotonic() - start
            if not is_lock_error(e) or waited + wait > max_wait:
                raise
            get_logger('pipeline.common').warning(f'{what} is locked by another process, retrying in {wait:.0f} s '
                         f'(waited {waited:.0f} of {max_wait:.0f} s)')
            time.sleep(wait)


def connect(db: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Open one of the pipeline databases ('prices', 'companies', 'raw'),
    creating its tables first when opened for writing. Waits while another
    process holds the file (with_lock_retry).
    """
    from pipeline import schema
    path = DATABASES[db]
    path.parent.mkdir(parents=True, exist_ok=True)
    con = with_lock_retry(lambda: duckdb.connect(str(path), read_only=read_only), path.name)
    if not read_only:
        schema.create(con, db)
    return con


def attach(con, db: str, alias: str) -> None:
    """
    ATTACH another pipeline database read-only as `alias`, waiting while it is locked.
    """
    path = DATABASES[db].as_posix()
    with_lock_retry(lambda: con.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)"), DATABASES[db].name)


def primary_key(con, table: str) -> list[str]:
    rows = con.execute("""
        SELECT constraint_column_names FROM duckdb_constraints()
        WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'""", [table]).fetchall()
    return list(rows[0][0]) if rows else []


def upsert(con, table: str, df: pd.DataFrame) -> int:
    """
    Insert new rows; for rows whose primary key exists, update only the
    columns present in `df` (other columns keep their values, so steps can
    fill different columns of the same row). Duplicate keys inside `df` keep
    the last row. Returns the number of rows written.
    """
    if df is None or df.empty:
        return 0
    key = primary_key(con, table)
    if key:
        df = df.drop_duplicates(subset=key, keep='last')
    con.register('_incoming', df)
    try:
        cols = ', '.join(f'"{c}"' for c in df.columns)
        sql = f'INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming'
        if key:
            updates = [c for c in df.columns if c not in key]
            conflict = ', '.join(f'"{c}"' for c in key)
            if updates:
                sets = ', '.join(f'"{c}" = EXCLUDED."{c}"' for c in updates)
                sql += f' ON CONFLICT ({conflict}) DO UPDATE SET {sets}'
            else:
                sql += f' ON CONFLICT ({conflict}) DO NOTHING'
        con.execute(sql)
    finally:
        con.unregister('_incoming')
    return len(df)


# ----
# RUN CONTEXT / LOAD LOG
# ----

@dataclass
class StepResult:
    """
    What a step did: counts per item (ticker / company / series) plus rows written.
    """
    rows: int = 0
    updated: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    message: str = ''

    def fail(self, item: str, error: Exception | str, log: logging.Logger | None = None) -> None:
        self.failed.append(item)
        if log:
            log.warning(f'{item}: {error}')


@dataclass
class RunContext:
    tickers: list[str] | None = None        # --tickers filter, None = all
    nct_ids: list[str] | None = None        # --nct: extra trials for the 'trials' step
    run_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d-%H%M%S'))

    def selected(self, tickers) -> list[str]:
        tickers = list(tickers)
        if self.tickers is None:
            return tickers
        wanted = {t.upper() for t in self.tickers}
        return [t for t in tickers if t.upper() in wanted]


def write_load_log(con, run_id: str, step: str, started_at: datetime, result: StepResult | None,
                   status: str, error: str | None = None) -> None:
    con.execute("""
        INSERT INTO load_log (run_id, step, started_at, finished_at, rows_written,
                              n_updated, n_skipped, n_failed, failed_items, status, message)
        VALUES (?, ?, ?, current_timestamp, ?, ?, ?, ?, ?, ?, ?)""", [
        run_id, step, started_at,
        result.rows if result else 0,
        result.updated if result else 0,
        result.skipped if result else 0,
        len(result.failed) if result else 0,
        result.failed[:500] if result else [],
        status,
        error or (result.message if result else None),
    ])


def format_error(e: BaseException) -> str:
    return redact(''.join(traceback.format_exception_only(type(e), e)).strip())


# ----
# TICKER FILES
# ----

def load_ticker_files() -> pd.DataFrame:
    """
    Every dict in _tickers/*.py as rows: ticker, name, asset_class (file name),
    group_name (dict name). The files are only read, never changed.
    """
    rows = []
    for path in sorted(config.TICKERS_DIR.glob('*.py')):
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location(f'_tickers_{path.stem}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for group, values in vars(module).items():
            if group.startswith('_') or not isinstance(values, dict):
                continue
            for ticker, name in values.items():
                rows.append({'ticker': ticker, 'name': name, 'asset_class': path.stem, 'group_name': group})
    return pd.DataFrame(rows, columns=['ticker', 'name', 'asset_class', 'group_name'])
