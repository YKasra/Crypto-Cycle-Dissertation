import pandas as pd
import numpy as np


class BacktestEngine:
    def __init__(self, initial_capital=10000, fee_rate=0.001, slippage=0.0005):
        """
        :param initial_capital: Starting balance in USDT
        :param fee_rate: 0.001 = 0.1% Binance standard fee
        :param slippage: 0.0005 = 0.05% price impact per trade
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0       # Quantity of asset held
        self.trades = []        # Full trade ledger
        self.equity_curve = []
        self.is_long = False

    def run(self, data, signals):
        """
        Run backtest on OHLCV data using a signal series.

        :param data:    DataFrame with columns: timestamp, open, high, low, close
        :param signals: Series or array with values 1 (Buy), -1 (Sell/Exit), 0 (Hold)
        :return:        (equity_df, trades_df)
        """
        self.reset()

        df = data.copy().reset_index(drop=True)
        df['signal'] = pd.Series(signals).values

        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']
            signal = row['signal']

            if signal == 1 and not self.is_long:
                self._execute_buy(row)
            elif signal == -1 and self.is_long:
                self._execute_sell(row)

            total_equity = self.cash + (self.position * price)
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': total_equity,
                'cash': self.cash,
                'position_value': self.position * price
            })

        return pd.DataFrame(self.equity_curve), pd.DataFrame(self.trades)

    def _execute_buy(self, row):
        price = row['close']
        timestamp = row['timestamp']

        # FIX 1: Apply slippage first, then calculate fee on full cash amount
        # This mirrors how Binance actually charges: fee is taken from quantity received
        execution_price = price * (1 + self.slippage)
        fee = self.cash * self.fee_rate
        investable_cash = self.cash - fee                  # cash after fee
        self.position = investable_cash / execution_price   # quantity of asset bought
        self.cash = 0
        self.is_long = True

        self.trades.append({
            'timestamp': timestamp,
            'side': 'buy',
            'price': execution_price,
            'quantity': self.position,
            'fee': fee,
            'value': investable_cash   # net cash spent (after fee)
        })

    def _execute_sell(self, row):
        price = row['close']
        timestamp = row['timestamp']

        execution_price = price * (1 - self.slippage)
        gross_value = self.position * execution_price
        fee = gross_value * self.fee_rate
        net_value = gross_value - fee                      # cash received after fee
        self.cash = net_value
        self.position = 0
        self.is_long = False

        self.trades.append({
            'timestamp': timestamp,
            'side': 'sell',
            'price': execution_price,
            'quantity': 0,
            'fee': fee,
            'value': net_value   # net cash received (after fee)
        })


if __name__ == "__main__":
    print("=== BacktestEngine Self-Test ===\n")

    # Deterministic test: rising price should produce a profit
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.linspace(16000, 30000, 100)  # price rises linearly

    data = pd.DataFrame({'timestamp': dates, 'close': prices})

    # Buy at start, sell at end
    signals = np.zeros(100)
    signals[0] = 1    # Buy on day 1
    signals[-1] = -1  # Sell on day 100

    engine = BacktestEngine(initial_capital=10000)
    equity_df, trades_df = engine.run(data, signals)

    print(f"Initial Capital : $10,000.00")
    print(f"Final Equity    : ${equity_df.iloc[-1]['equity']:,.2f}")
    print(f"Total Trades    : {len(trades_df)}")
    print(f"\nTrade Ledger:")
    print(trades_df.to_string(index=False))