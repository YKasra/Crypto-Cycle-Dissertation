"""
run_backtest.py
---------------
Runs all traditional indicator strategies (MACD, RSI, Bollinger Bands)
against real BTC, ETH, and SOL data and prints a performance summary table.
This produces the Phase 1 benchmark metrics for the dissertation.
"""

import pandas as pd
import sys
import os

# Make sure imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.engine import BacktestEngine
from backtesting.metrics import calculate_metrics
from strategies.traditional.macd import MACDStrategy
from strategies.traditional.rsi import RSIStrategy
from strategies.traditional.bollinger import BollingerStrategy
from strategies.cycle_detection.fft_strategy import FFTStrategy
from strategies.cycle_detection.mesa_strategy import MESAStrategy
from strategies.cycle_detection.hilbert_strategy import HilbertStrategy
# ── Configuration ────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10000
INTERVAL        = '1d'       # '1d' or '1h'
DATA_DIR        = 'data/raw'

SYMBOLS = {
    'BTC': 'BTCUSDT_1d.csv',
    'ETH': 'ETHUSDT_1d.csv',
    'SOL': 'SOLUSDT_1d.csv',
}

STRATEGIES = {
    'MACD':           MACDStrategy(),
    'RSI':            RSIStrategy(),
    'Bollinger Bands': BollingerStrategy(),
    'FFT':             FFTStrategy(window=64, min_period=10, max_period=40),
    'MESA':            MESAStrategy(window=32, min_period=10, max_period=40),
    'Hilbert': HilbertStrategy(window=64, smooth_period=7, adaptive_lookback=50, threshold_pct=25),
    }
# ─────────────────────────────────────────────────────────────────────────────


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def run_all():
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    all_results = []

    for asset, filename in SYMBOLS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath} — skipping.")
            continue

        data = load_data(filepath)
        print(f"\n{'='*60}")
        print(f"  Asset: {asset}  |  Rows: {len(data)}  |  Interval: {INTERVAL}")
        print(f"  From: {data['timestamp'].iloc[0].date()}  "
              f"To: {data['timestamp'].iloc[-1].date()}")
        print(f"{'='*60}")

        for strategy_name, strategy in STRATEGIES.items():
            signals = strategy.generate_signals(data)
            equity_df, trades_df = engine.run(data, signals)
            metrics = calculate_metrics(
                equity_df, trades_df,
                initial_capital=INITIAL_CAPITAL,
                interval=INTERVAL
            )

            # Buy & Hold benchmark
            bh_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100

            print(f"\n  Strategy : {strategy_name}")
            print(f"  ─────────────────────────────────────")
            print(f"  Total Return         : {metrics.get('Total Return (%)', 0):>8.2f}%")
            print(f"  Buy & Hold Return    : {bh_return:>8.2f}%")
            print(f"  Annualized Volatility: {metrics.get('Annualized Volatility (%)', 0):>8.2f}%")
            print(f"  Sharpe Ratio         : {metrics.get('Sharpe Ratio', 0):>8.4f}")
            print(f"  Max Drawdown         : {metrics.get('Max Drawdown (%)', 0):>8.2f}%")
            print(f"  Win Rate             : {metrics.get('Win Rate (%)', 0):>8.2f}%")
            print(f"  Total Trades         : {metrics.get('Total Trades', 0):>8}")
            print(f"  Total Fees Paid      : ${metrics.get('Total Fees Paid ($)', 0):>7.2f}")

            all_results.append({
                'Asset': asset,
                'Strategy': strategy_name,
                **metrics,
                'Buy & Hold Return (%)': round(bh_return, 2)
            })

    # Save summary to CSV for use in dissertation tables
    results_df = pd.DataFrame(all_results)
    output_path = 'data/processed/baseline_results.csv'
    os.makedirs('data/processed', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\n✅ All results saved to {output_path}")
    print("\nFull Summary Table:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_all()