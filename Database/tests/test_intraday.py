"""
Intraday resampling and the regular-hours filter. No network.
"""

import pandas as pd
import pytest

from pipeline.intraday import flag_deviations, regular_hours, resample


def minutes(day, times, price=100.0, volume=10.0):
    """
    1-minute bars at New York times 'HH:MM' on `day`.
    """
    ts = pd.to_datetime([f'{day} {t}' for t in times]).tz_localize('America/New_York').tz_convert('UTC')
    n = len(ts)
    return pd.DataFrame({'ts': ts, 'open': price, 'high': price + 1, 'low': price - 1, 'close': price,
                         'volume': volume, 'vwap': price, 'trade_count': 2})


def ny(ts):
    return ts.tz_convert('America/New_York').strftime('%H:%M')


def test_one_hour_bars_anchored_at_open():
    day = [f'{h:02d}:{m:02d}' for h in range(9, 16) for m in range(60) if (h, m) >= (9, 30)]
    h = resample(minutes('2026-09-24', day), 60)
    assert list(map(ny, h['ts'])) == ['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30']
    assert list(h['volume']) == [600] * 6 + [300]              # last bar 15:30-16:00 = 30 minutes


def test_no_empty_bars_and_sums():
    df = minutes('2026-09-24', ['09:30', '09:31', '09:52'])      # nothing traded 09:32-09:51
    b = resample(df, 5)
    assert list(map(ny, b['ts'])) == ['09:30', '09:50']
    assert list(b['trade_count']) == [4, 2] and list(b['volume']) == [20, 10]


def test_vwap_volume_weighted_and_ohlc():
    df = pd.concat([minutes('2026-09-24', ['09:30'], price=100, volume=100),
                    minutes('2026-09-24', ['09:31'], price=110, volume=300)])
    b = resample(df, 10).iloc[0]
    assert b['vwap'] == pytest.approx((100 * 100 + 110 * 300) / 400)
    assert (b['open'], b['close'], b['high'], b['low']) == (100, 110, 111, 99)


def test_early_close_is_just_a_shorter_day():
    day = [f'{h:02d}:{m:02d}' for h in range(9, 13) for m in range(60) if (h, m) >= (9, 30)] + ['13:00']
    h = resample(minutes('2026-11-27', day), 60)                  # day after Thanksgiving, close 13:00
    assert list(map(ny, h['ts'])) == ['09:30', '10:30', '11:30', '12:30']
    assert h['volume'].iloc[-1] == 310                              # 12:30-12:59 + the 13:00 closing auction


def test_closing_auction_minute_folded_into_last_bar():
    df = pd.concat([minutes('2026-09-24', ['15:58', '15:59'], price=100),
                    minutes('2026-09-24', ['16:00'], price=105, volume=1000)])
    for m, first in ((5, '15:55'), (60, '15:30')):
        b = resample(df, m)
        assert list(map(ny, b['ts'])) == [first]
        assert b['close'].iloc[0] == 105 and b['volume'].iloc[0] == 1020


def test_regular_hours_filter():
    df = minutes('2026-09-24', ['04:00', '09:29', '09:30', '15:59', '16:00', '16:01', '19:59'])
    assert list(map(ny, regular_hours(df)['ts'])) == ['09:30', '15:59', '16:00']      # 16:00 = closing auction
    half = minutes('2026-12-24', ['12:59', '13:00', '13:01', '15:00'])
    assert list(map(ny, regular_hours(half)['ts'])) == ['12:59', '13:00']


def test_volume_check_against_own_median_ratio():
    # ticker A always ~80 % of daily volume in the session (heavy extended hours): not flagged;
    # one day at 60 % is 25 % below its median: flagged
    ratios = {'A': [0.80, 0.81, 0.79, 0.80, 0.60], 'B': [0.97, 0.98, 0.97, 0.96, 0.97]}
    rows = [{'ticker': t, 'date': i, 'i_high': 10, 'i_low': 9, 'd_high': 10, 'd_low': 9,
             'i_volume': r * 1000, 'd_volume': 1000} for t, rs in ratios.items() for i, r in enumerate(rs)]
    df = flag_deviations(pd.DataFrame(rows))
    assert df.loc[df['flagged'], ['ticker', 'date']].values.tolist() == [['A', 4]]
