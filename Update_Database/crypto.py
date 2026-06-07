import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

ticker_crypto = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'USDT-USD': 'Tether'
}