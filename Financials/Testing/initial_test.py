


import pandas as pd
import numpy as np
from pathlib import Path

import risk_metrics
import portfolio
import plotting

current_dir = Path.cwd()


universe = [
    'US',
    'North America',
    'Europe',
    'Asia',
    'Developed Markets',
    'Emerging Markets',
]

mcap = {
    'large_1b', 'min': 1e9, 'max':None,
    'small_cap', 'min': 1e8, 'max':1e9,
}

regime = {
    # Simple Regimes
    "bond":             ,
    "commodity":        ,
    "crypto":           ,
    "equity":           ,
    "forex":            ,

    # Complex Multi-Layer Regimes
    "growth":           ,
    "inflation":        ,
    "liquidity":        ,
    "risk_appetite":    ,
    "hidden_markov":    ,
}




def _make_dir():
    os.



def main(): 




if __name__ == "__main__":
    main()