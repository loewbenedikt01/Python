
"""
Definition of Crisis Periods

Used for Metrics calculation and timewindow definition
of different crisis periods.
6 Main-Crises
3 Sub-Crises of GFC
"""


main_crises = [
    {'label': 'crisis_1',   'peak': '2000-03-23',      'trough': '2002-10-09',     'even': '2007-05-31'},        # Dotcom Crisis
    {'label': 'crisis_2',   'peak': '2007-10-09',      'trough': '2009-03-09',     'even': '2013-03-28'},        # Global Financial Crisis
    {'label': 'crisis_3',   'peak': '2018-09-21',      'trough': '2018-12-24',     'even': '2019-04-23'},        # Monetary Policy Shock
    {'label': 'crisis_4',   'peak': '2020-02-19',      'trough': '2020-03-23',     'even': '2020-08-12'},        # Covid-19 Crisis
    {'label': 'crisis_5',   'peak': '2022-01-03',      'trough': '2022-10-12',     'even': '2024-01-19'},        # Inflation and Rate Hike Cycle
    {'label': 'crisis_6',   'peak': '2025-02-19',      'trough': '2025-04-08',     'even': '2025-06-26'},        # Trade Policy Shock
]

sub_crises  = [
    {'label': 'sub_crisis_1',   'peak': '2007-10-09',      'trough': '2008-09-15',     'even': '2008-09-15'},        # Early Credit Crunch
    {'label': 'sub_crisis_2',   'peak': '2008-09-15',      'trough': '2009-03-09',     'even': '2010-04-23'},        # Acute GFC Crash
    {'label': 'sub_crisis_3',   'peak': '2010-01-23',      'trough': '2011-10-03',     'even': '2012-03-23'},        # EU Debt + US Debt Ceiling
]

crisis_metrics_ptt  = [
    {'label': 'ptt_sharpe',     'format': '{:.3f}'},        # Sharpe Ratio
    {'label': 'ptt_sortino',    'format': '{:.3f}'},        # Sortino Ratio
    {'label': 'ptt_calmar',     'format': '{:.3f}'},        # Calmar Ratio
    {'label': 'ptt_ulcer',      'format': '{:.3f}'},        # Ulcer Index
    {'label': 'ptt_mdd',        'format': '{:.2%}'},        # Maximum Drawdown
    {'label': 'ptt_return',     'format': '{:.2%}'},        # Cumulative Return
]

crisis_metrics_ttp  = [
    {'label': 'ttp_sharpe',     'format': '{:.3f}'},        # Sharpe Ratio
    {'label': 'ttp_sortino',    'format': '{:.3f}'},        # Sortino Ratio
    {'label': 'ttp_calmar',     'format': '{:.3f}'},        # Calmar Ratio
    {'label': 'ttp_ulcer',      'format': '{:.3f}'},        # Ulcer Index
    {'label': 'ttp_mdd',        'format': '{:.2%}'},        # Maximum Drawdown
    {'label': 'ttp_return',     'format': '{:.2%}'},        # Cumulative Return
]

crisis_metrics_full  = [
    {'label': 'full_sharpe',     'format': '{:.3f}'},        # Sharpe Ratio
    {'label': 'full_sortino',    'format': '{:.3f}'},        # Sortino Ratio
    {'label': 'full_calmar',     'format': '{:.3f}'},        # Calmar Ratio
    {'label': 'full_ulcer',      'format': '{:.3f}'},        # Ulcer Index
    {'label': 'full_mdd',        'format': '{:.2%}'},        # Maximum Drawdown
    {'label': 'full_return',     'format': '{:.2%}'},        # Cumulative Return
]

