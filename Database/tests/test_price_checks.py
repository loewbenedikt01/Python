"""
Price glitches: GBp <-> GBP unit switches and one-day spikes. No network.
"""

import pandas as pd
import pytest

from pipeline.price_checks import spike_days, unit_factors


def test_stretch_in_pounds_is_scaled_to_pence():
    close = pd.Series([1500, 1510, 15.2, 15.3, 1540, 1550.0])      # days 2-3 in pounds
    assert list(unit_factors(close)) == [1, 1, 100, 100, 1, 1]


def test_history_in_pounds_before_a_permanent_switch():
    close = pd.Series([14.9, 15.0, 1505, 1510.0])                    # Yahoo switched to pence on day 2
    assert list(unit_factors(close)) == [100, 100, 1, 1]


def test_day_100x_too_high():
    close = pd.Series([1500, 151000, 1505.0])
    assert list(unit_factors(close)) == pytest.approx([1, 0.01, 1])


def test_real_moves_are_not_unit_switches():
    close = pd.Series([100, 60, 110, 3000.0])                         # -40 %, +83 %, x27
    assert list(unit_factors(close)) == [1, 1, 1, 1]


def test_one_day_spikes_up_and_down():
    close = pd.Series([100, 5100, 99, 100, 101, 10.5, 102, 103.0])    # ULVR.L-like x51, TATASTEEL-like /10
    assert list(spike_days(close)) == [False, True, False, False, False, True, False, False]


def test_spike_over_three_days_reverses():
    close = pd.Series([100, 200, 205, 198, 101, 102.0])               # NOVO-B.CO-like x2 for 3 days
    assert list(spike_days(close)) == [False, True, True, True, False, False]


def test_real_moves_are_not_spikes():
    crash = pd.Series([100, 65, 70, 72, 75, 74.0])                    # -35 %, no reversal
    partial = pd.Series([100, 60, 80, 85, 90.0])                      # rebounds only partly
    slow = pd.Series([100, 140, 145, 150, 155, 100.0])                # back after 4 days: not a spike
    for s in (crash, partial, slow):
        assert not spike_days(s).any()
