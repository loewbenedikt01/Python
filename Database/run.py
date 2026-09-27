"""
Entry script for the data pipeline.

    python run.py                          # all steps in order
    python run.py prices intraday          # only the named steps
    python run.py --tickers AAPL MSFT      # limit to some tickers (testing)
    python run.py status                   # row counts, latest dates, last runs
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb
import pandas as pd

import config
from pipeline import common, schema


log = common.get_logger('pipeline.run')


# ----
# STEPS
# ----

def run_steps(steps: list[str], ctx: common.RunContext) -> None:
    summary = []
    for step in steps:
        name = f'pipeline.{config.STEPS[step]}'
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
            module = None
        if not hasattr(module, 'run'):
            log.info(f'[{step}] not implemented yet, skipped')
            summary.append((step, 'not built', 0, 0, 0, 0))
            continue

        log.info(f'[{step}] start')
        started = datetime.now()
        result, status, error = None, 'ok', None
        try:
            result = module.run(ctx)
            if result.failed:
                status = 'partial'
        except Exception as e:                      # one broken step must not stop the run
            status, error = 'failed', common.format_error(e)
            log.exception(f'[{step}] failed')

        con = common.connect(schema.STEP_DB[step])
        try:
            common.write_load_log(con, ctx.run_id, step, started, result, status, error)
        finally:
            con.close()

        r = result or common.StepResult()
        log.info(f'[{step}] {status}: {r.rows} rows, {r.updated} updated, {r.skipped} skipped, {len(r.failed)} failed')
        summary.append((step, status, r.rows, r.updated, r.skipped, len(r.failed)))

    print('\n=== Summary ===')
    print(pd.DataFrame(summary, columns=['step', 'status', 'rows', 'updated', 'skipped', 'failed']).to_string(index=False))


# ----
# STATUS
# ----

def print_status() -> None:
    rows = []
    for db, tables in schema.TABLE_INFO.items():
        path = common.DATABASES[db]
        if not path.exists():
            rows.append((path.name, '(not created yet)', None, None, None, None))
            continue
        con = duckdb.connect(str(path), read_only=True)
        try:
            for table, (date_col, step) in tables.items():
                n, latest = con.execute(f'SELECT count(*), max({date_col}) FROM {table}').fetchone()
                # the step's run is logged in its own database (executives writes to companies and raw)
                log_db = schema.STEP_DB[step]
                lcon = con if log_db == db else duckdb.connect(str(common.DATABASES[log_db]), read_only=True)
                try:
                    last_ok = lcon.execute("""
                        SELECT max(finished_at) FROM load_log WHERE step = ? AND status IN ('ok', 'partial')""",
                        [step]).fetchone()[0]
                    last_failed = lcon.execute("""
                        SELECT n_failed FROM load_log WHERE step = ? ORDER BY finished_at DESC LIMIT 1""",
                        [step]).fetchone()
                finally:
                    if lcon is not con:
                        lcon.close()
                rows.append((path.name, table, n, latest, last_ok, last_failed[0] if last_failed else None))
        finally:
            con.close()

    for folder in [config.INTRADAY_BASE_INTERVAL, *config.INTRADAY_DERIVED]:
        files = list((config.INTRADAY_DIR / folder).glob('ticker=*/*.parquet'))
        if not files:
            rows.append(('intraday', folder, 0, None, None, None))
            continue
        pattern = str(config.INTRADAY_DIR / folder / 'ticker=*' / '*.parquet').replace('\\', '/')
        n, latest = duckdb.sql(f"SELECT count(*), max(ts) FROM read_parquet('{pattern}', hive_partitioning = true)").fetchone()
        rows.append(('intraday', folder, n, latest, None, None))

    df = pd.DataFrame(rows, columns=['database', 'table', 'rows', 'latest', 'last_successful_run', 'failed_last_run'])
    with pd.option_context('display.width', 200, 'display.max_columns', 10):
        print(df.to_string(index=False))

    # tickers without data: 'no_data' (skipped) and those on their way there
    if config.PRICES_DB.exists():
        con = duckdb.connect(str(config.PRICES_DB), read_only=True)
        try:
            bad = con.execute("""
                SELECT asset_class, ticker, status, consecutive_failures, last_checked
                FROM instruments WHERE status = 'no_data' OR consecutive_failures > 0
                ORDER BY status DESC, asset_class, ticker""").df()
            bad_series = con.execute("""
                SELECT category, series_id, status, consecutive_failures, last_checked
                FROM macro_series WHERE coalesce(status, 'active') <> 'removed'
                  AND (status = 'no_data' OR consecutive_failures > 0)
                ORDER BY status DESC, series_id""").df()
        finally:
            con.close()
        if not bad_series.empty:
            n = int((bad_series['status'] == 'no_data').sum())
            print(f'\nFRED series without data ({n} no_data = skipped, re-checked every '
                  f'{config.NO_DATA_RECHECK_DAYS} days; others: failed runs so far):')
            print(bad_series.to_string(index=False))
        if not bad.empty:
            n = int((bad['status'] == 'no_data').sum())
            print(f'\nTickers without data ({n} no_data = skipped, re-checked every '
                  f'{config.NO_DATA_RECHECK_DAYS} days; others: failed runs so far):')
            print(bad.to_string(index=False))


# ----
# MAIN
# ----

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Market-data pipeline')
    parser.add_argument('steps', nargs='*', choices=[*config.STEPS, 'status'],
                        help='steps to run (default: all, in order)')
    parser.add_argument('--tickers', nargs='+', help='limit to these tickers')
    parser.add_argument('--nct', nargs='+', help="extra NCT IDs for the 'trials' step (added to the stored trials)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):         # company names like 'Moët' on a Windows console
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    for d in (config.DATA_DIR, config.CACHE_DIR, config.INTRADAY_DIR, config.REVIEW_DIR, config.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.steps == ['status']:
        print_status()
        return

    steps = [s for s in config.STEPS if s in args.steps] if args.steps else list(config.STEPS)
    run_steps(steps, common.RunContext(tickers=args.tickers, nct_ids=args.nct))
    if 'status' in args.steps:
        print_status()


if __name__ == '__main__':
    main()
