import pandas as pd


class MACDStrategy:
    """
    MACD (Moving Average Convergence Divergence) trend-following strategy.
    - Buy  when MACD line crosses ABOVE the signal line (bullish crossover)
    - Sell when MACD line crosses BELOW the signal line (bearish crossover)

    FIX: Detects the crossover EVENT, not just the state of being above/below.
    This prevents holding perpetual signals and generates clean entry/exit points.
    """

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data):
        """
        :param data: DataFrame with 'close' column
        :return:     Series of signals: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        df = data.copy()

        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()

        # FIX: Detect crossover by comparing current vs previous bar
        # prev_diff < 0 and curr_diff > 0 means MACD just crossed ABOVE signal line
        diff = macd - signal_line
        prev_diff = diff.shift(1)

        signals = pd.Series(0, index=df.index)
        in_position = False

        for i in range(1, len(df)):
            curr = diff.iloc[i]
            prev = prev_diff.iloc[i]

            if pd.isna(curr) or pd.isna(prev):
                continue

            # Bullish crossover: MACD crosses above signal line
            if prev < 0 and curr > 0 and not in_position:
                signals.iloc[i] = 1
                in_position = True

            # Bearish crossover: MACD crosses below signal line
            elif prev > 0 and curr < 0 and in_position:
                signals.iloc[i] = -1
                in_position = False

        return signals


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    prices = 30000 + np.cumsum(np.random.randn(300) * 500)
    data = pd.DataFrame({'close': prices}, index=range(300))
    data['timestamp'] = dates

    strategy = MACDStrategy()
    signals = strategy.generate_signals(data)

    buys = (signals == 1).sum()
    sells = (signals == -1).sum()
    print(f"MACD Self-Test")
    print(f"  Buy signals : {buys}")
    print(f"  Sell signals: {sells}")
    print(f"  Balanced pairs: {buys - sells <= 1}")