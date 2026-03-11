"""
bootstrap.py
Runs once on Streamlit Cloud (or any cold start) to download
BTC, ETH, SOL OHLCV data from Binance if the CSV files are missing.
Uses only the public Binance REST API — no API key required.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime

RAW_DIR   = "data/raw"
SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["1d", "1h"]
START     = "2017-01-01"
BASE_URL  = "https://api.binance.com/api/v3/klines"


def _files_exist():
    """Return True only if all 6 expected CSV files are present."""
    for sym in SYMBOLS:
        for iv in INTERVALS:
            if not os.path.exists(f"{RAW_DIR}/{sym}_{iv}.csv"):
                return False
    return True


def _fetch(symbol, interval, start_str):
    os.makedirs(RAW_DIR, exist_ok=True)
    start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    all_rows = []

    while True:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "limit":     1000,
        }
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 1
        time.sleep(0.15)          # stay well under Binance rate limit

    cols = ["timestamp","open","high","low","close","volume",
            "close_time","quote_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])

    path = f"{RAW_DIR}/{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    return len(df)


def ensure_data(status_callback=None):
    """
    Call this at app startup.
    If all 6 CSVs already exist → returns immediately (no network call).
    If any are missing → downloads all of them with a progress callback.

    status_callback(message: str) is called with progress updates
    so the Streamlit UI can display them.
    """
    if _files_exist():
        return   # nothing to do

    if status_callback:
        status_callback("📡 First launch detected — downloading market data from Binance...")

    total = len(SYMBOLS) * len(INTERVALS)
    done  = 0
    for sym in SYMBOLS:
        for iv in INTERVALS:
            label = f"{sym} {'Daily' if iv == '1d' else 'Hourly'}"
            if status_callback:
                status_callback(f"⬇️  Downloading {label}  ({done+1}/{total})")
            rows = _fetch(sym, iv, START)
            done += 1
            if status_callback:
                status_callback(f"✅  {label} — {rows:,} candles saved  ({done}/{total})")

    if status_callback:
        status_callback("🎉  All data ready — loading dashboard...")
