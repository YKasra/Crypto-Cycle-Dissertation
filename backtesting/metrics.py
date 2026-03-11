import numpy as np
import pandas as pd


def calculate_metrics(equity_df, trades_df, initial_capital=10000, interval='1d'):
    """
    Comprehensive performance metrics for dissertation backtesting.
    Covers all Group 1 and Group 2 metrics.
    """
    if equity_df.empty:
        return {}

    equity = equity_df['equity'].values
    timestamps = equity_df['timestamp']

    # ── Basic Returns ────────────────────────────────────────────────────────
    total_return = (equity[-1] / initial_capital) - 1

    # Annualisation factor
    periods_per_year = 8760 if interval == '1h' else 365
    period_returns = pd.Series(equity).pct_change().dropna()

    # CAGR
    n_years = len(equity) / periods_per_year
    cagr = (equity[-1] / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0

    # ── Volatility & Risk ────────────────────────────────────────────────────
    ann_volatility = period_returns.std() * np.sqrt(periods_per_year) * 100

    # VaR 95% (daily loss not exceeded 95% of the time)
    var_95 = float(np.percentile(period_returns, 5) * 100)

    # ── Drawdown Analysis ────────────────────────────────────────────────────
    equity_series = pd.Series(equity)
    rolling_max   = equity_series.cummax()
    drawdown      = (equity_series - rolling_max) / rolling_max
    max_drawdown  = float(drawdown.min() * 100)
    avg_drawdown  = float(drawdown[drawdown < 0].mean() * 100) if (drawdown < 0).any() else 0

    # Drawdown periods for recovery calculation
    in_drawdown       = drawdown < 0
    drawdown_start    = (~in_drawdown).shift(1) & in_drawdown
    recovery_factor   = abs(total_return * 100 / max_drawdown) if max_drawdown != 0 else 0

    # ── Sharpe Ratio ─────────────────────────────────────────────────────────
    sharpe = float(
        np.sqrt(periods_per_year) * period_returns.mean() / period_returns.std()
        if len(period_returns) > 1 and period_returns.std() != 0 else 0
    )

    # ── Sortino Ratio ────────────────────────────────────────────────────────
    downside = period_returns[period_returns < 0]
    downside_std = downside.std()
    sortino = float(
        np.sqrt(periods_per_year) * period_returns.mean() / downside_std
        if downside_std != 0 else 0
    )

    # ── Calmar Ratio ─────────────────────────────────────────────────────────
    calmar = float(cagr * 100 / abs(max_drawdown)) if max_drawdown != 0 else 0

    # ── Trade-level Metrics ──────────────────────────────────────────────────
    win_rate        = 0.0
    profit_factor   = 0.0
    payoff_ratio    = 0.0
    expectancy      = 0.0
    avg_trade_dur   = 0.0
    best_trade      = 0.0
    worst_trade     = 0.0
    win_streak      = 0
    loss_streak     = 0
    gross_profit    = 0.0
    gross_loss      = 0.0
    num_trades      = 0
    total_fees      = 0.0

    if not trades_df.empty and len(trades_df) >= 2:
        buys  = trades_df[trades_df['side'] == 'buy'].reset_index(drop=True)
        sells = trades_df[trades_df['side'] == 'sell'].reset_index(drop=True)
        num_trades = min(len(buys), len(sells))
        total_fees = trades_df['fee'].sum()

        pnls      = []
        durations = []

        for i in range(num_trades):
            pnl = sells.iloc[i]['value'] - buys.iloc[i]['value']
            pnls.append(pnl)

            # Trade duration
            try:
                buy_time  = pd.to_datetime(buys.iloc[i]['timestamp'])
                sell_time = pd.to_datetime(sells.iloc[i]['timestamp'])
                dur_hours = (sell_time - buy_time).total_seconds() / 3600
                durations.append(dur_hours)
            except Exception:
                pass

        pnls_arr  = np.array(pnls)
        wins_arr  = pnls_arr[pnls_arr > 0]
        losses_arr= pnls_arr[pnls_arr < 0]

        win_rate      = len(wins_arr) / num_trades * 100 if num_trades > 0 else 0
        gross_profit  = float(wins_arr.sum()) if len(wins_arr) > 0 else 0
        gross_loss    = float(abs(losses_arr.sum())) if len(losses_arr) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win  = wins_arr.mean()   if len(wins_arr)   > 0 else 0
        avg_loss = abs(losses_arr.mean()) if len(losses_arr) > 0 else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        expectancy    = float(pnls_arr.mean()) if len(pnls_arr) > 0 else 0
        best_trade    = float(pnls_arr.max())  if len(pnls_arr) > 0 else 0
        worst_trade   = float(pnls_arr.min())  if len(pnls_arr) > 0 else 0
        avg_trade_dur = float(np.mean(durations)) if durations else 0

        # Win/Loss streaks
        cur_win = cur_loss = 0
        for p in pnls:
            if p > 0:
                cur_win += 1; cur_loss = 0
                win_streak = max(win_streak, cur_win)
            else:
                cur_loss += 1; cur_win = 0
                loss_streak = max(loss_streak, cur_loss)

    # ── Best / Worst Day ─────────────────────────────────────────────────────
    best_day  = float(period_returns.max() * 100)
    worst_day = float(period_returns.min() * 100)

    return {
        # Core
        'Total Return (%)':          round(total_return * 100, 2),
        'CAGR (%)':                  round(cagr * 100, 2),
        'Annualized Volatility (%)': round(ann_volatility, 2),

        # Risk-adjusted
        'Sharpe Ratio':              round(sharpe, 4),
        'Sortino Ratio':             round(sortino, 4),
        'Calmar Ratio':              round(calmar, 4),
        'Recovery Factor':           round(recovery_factor, 2),
        'VaR 95% (%)':               round(var_95, 2),

        # Drawdown
        'Max Drawdown (%)':          round(max_drawdown, 2),
        'Avg Drawdown (%)':          round(avg_drawdown, 2),

        # Trade quality
        'Profit Factor':             round(profit_factor, 2),
        'Payoff Ratio':              round(payoff_ratio, 2),
        'Expectancy ($)':            round(expectancy, 2),
        'Win Rate (%)':              round(win_rate, 2),
        'Total Trades':              num_trades,
        'Best Trade ($)':            round(best_trade, 2),
        'Worst Trade ($)':           round(worst_trade, 2),
        'Win Streak':                win_streak,
        'Loss Streak':               loss_streak,
        'Gross Profit ($)':          round(gross_profit, 2),
        'Gross Loss ($)':            round(gross_loss, 2),
        'Avg Trade Duration (hrs)':  round(avg_trade_dur, 1),

        # Period
        'Best Day (%)':              round(best_day, 2),
        'Worst Day (%)':             round(worst_day, 2),
        'Total Fees Paid ($)':       round(total_fees, 2),
    }


if __name__ == "__main__":
    print("=== Comprehensive Metrics Self-Test ===\n")
    equity_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
        'equity':    10000 * (1 + np.random.randn(100).cumsum() * 0.01)
    })
    trades_data = pd.DataFrame([
        {'side': 'buy',  'timestamp': '2023-01-05', 'value': 9990.0,  'fee': 10.0},
        {'side': 'sell', 'timestamp': '2023-01-20', 'value': 11000.0, 'fee': 11.0},
        {'side': 'buy',  'timestamp': '2023-02-01', 'value': 10900.0, 'fee': 11.0},
        {'side': 'sell', 'timestamp': '2023-02-15', 'value': 10500.0, 'fee': 10.5},
        {'side': 'buy',  'timestamp': '2023-03-01', 'value': 10400.0, 'fee': 10.4},
        {'side': 'sell', 'timestamp': '2023-03-20', 'value': 12000.0, 'fee': 12.0},
    ])
    m = calculate_metrics(equity_data, trades_data, interval='1d')
    for k, v in m.items():
        print(f"  {k:<30}: {v}")