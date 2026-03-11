"""
precompute_results.py
---------------------
Run this script ONCE from the terminal before launching the dashboard.
It computes all 6 strategies × 3 assets × daily interval and saves
everything to data/processed/precomputed_results.json

Runtime: ~3-8 minutes depending on your machine.
After this, the dashboard Final Results view loads in under 1 second.

Usage:
    python precompute_results.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.engine import BacktestEngine
from backtesting.metrics import calculate_metrics
from strategies.traditional.macd import MACDStrategy
from strategies.traditional.rsi import RSIStrategy
from strategies.traditional.bollinger import BollingerStrategy
from strategies.cycle_detection.fft_strategy import FFTStrategy
from strategies.cycle_detection.mesa_strategy import MESAStrategy
from strategies.cycle_detection.hilbert_strategy import HilbertStrategy

# ── Config ────────────────────────────────────────────────────────────────────
STRATEGY_MAP = {
    'MACD':            (MACDStrategy,     {},                                    'traditional'),
    'RSI':             (RSIStrategy,      {},                                    'traditional'),
    'Bollinger Bands': (BollingerStrategy,{},                                    'traditional'),
    'FFT':             (FFTStrategy,      {'window':64,'min_period':10,
                                           'max_period':40},                     'dsp'),
    'MESA':            (MESAStrategy,     {'window':32,'order':12,'min_period':10,
                                           'max_period':40,'min_hold_bars':5},   'dsp'),
    'Hilbert':         (HilbertStrategy,  {'window':64,'smooth_period':7,
                                           'adaptive_lookback':50,
                                           'threshold_pct':25},                  'dsp'),
}

ASSETS    = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
INTERVAL  = '1d'
OUT_PATH  = 'data/processed/precomputed_results.json'

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(symbol, interval):
    path = f"data/raw/{symbol}_{interval}.csv"
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def safe_serialize(obj):
    """Convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return 0.0
    return obj

def serialize_dict(d):
    return {k: safe_serialize(v) for k, v in d.items()}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs('data/processed', exist_ok=True)
    all_results = {}
    total = len(ASSETS) * len(STRATEGY_MAP)
    done  = 0

    print(f"\n{'='*60}")
    print(f"  Precomputing {total} backtests ({len(ASSETS)} assets × {len(STRATEGY_MAP)} strategies)")
    print(f"  Interval : {INTERVAL}")
    print(f"  Output   : {OUT_PATH}")
    print(f"{'='*60}\n")

    for symbol in ASSETS:
        df = load_data(symbol, INTERVAL)
        if df is None:
            print(f"  Skipping {symbol} — data file missing.\n")
            continue

        bh_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        date_from = df['timestamp'].iloc[0].strftime('%d %b %Y')
        date_to   = df['timestamp'].iloc[-1].strftime('%d %b %Y')

        print(f"  Asset : {symbol}  |  {date_from} → {date_to}  |  {len(df):,} bars")
        print(f"  Buy & Hold : {bh_return:+.2f}%\n")

        all_results[symbol] = {
            '_meta': {
                'symbol':    symbol,
                'interval':  INTERVAL,
                'date_from': date_from,
                'date_to':   date_to,
                'bars':      len(df),
                'bh_return': round(bh_return, 2),
            }
        }

        for name, (cls, kwargs, stype) in STRATEGY_MAP.items():
            done += 1
            print(f"  [{done:02d}/{total}] {symbol} — {name} ...", end=' ', flush=True)
            t0 = time.time()

            try:
                strategy = cls(**kwargs)
                signals  = strategy.generate_signals(df)
                engine   = BacktestEngine(initial_capital=10000)
                eq_df, tr_df = engine.run(df, signals)
                metrics  = calculate_metrics(eq_df, tr_df, interval=INTERVAL)

                # Add equity curve (downsample to 500 points max for fast chart loading)
                step = max(1, len(eq_df) // 500)
                eq_sampled = eq_df.iloc[::step][['timestamp','equity']].copy()
                eq_sampled['timestamp'] = eq_sampled['timestamp'].astype(str)

                metrics['_type']       = stype
                metrics['_bh_return']  = round(bh_return, 2)
                metrics['_equity']     = eq_sampled.to_dict(orient='list')
                metrics['_timestamps'] = df['timestamp'].astype(str).iloc[::step].tolist()
                metrics['_closes']     = df['close'].iloc[::step].tolist()

                all_results[symbol][name] = serialize_dict(metrics)

                elapsed = time.time() - t0
                ret     = metrics.get('Total Return (%)', 0)
                sharpe  = metrics.get('Sharpe Ratio', 0)
                trades  = metrics.get('Total Trades', 0)
                print(f"done in {elapsed:.1f}s  |  Return: {ret:+.1f}%  Sharpe: {sharpe:.3f}  Trades: {trades}")

            except Exception as e:
                print(f"ERROR — {e}")
                all_results[symbol][name] = {'_error': str(e), '_type': stype}

        print()

    # Save
    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=safe_serialize)

    print(f"\n{'='*60}")
    print(f"  Saved to {OUT_PATH}")
    print(f"  File size: {os.path.getsize(OUT_PATH) / 1024:.1f} KB")
    print(f"\n  Dashboard Final Results will now load in < 1 second.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()