import pandas as pd
import numpy as np


class RSIStrategy:
    """
    RSI (Relative Strength Index) momentum oscillator strategy.
    - Buy  when RSI crosses below the oversold threshold (default 30)
    - Sell when RSI crosses above the overbought threshold (default 70)

    FIX 1: Uses Wilder's Smoothing Method (alpha = 1/period) which is the
            academically and industrially correct RSI calculation, not SMA.
    FIX 2: State tracking prevents duplicate signals in prolonged trends.
    """

    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_rsi(self, close):
        """
        Wilder's RSI — the correct standard implementation.
        Uses exponential smoothing with alpha = 1/period.
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # FIX: Wilder's smoothing = EMA with alpha=1/period (com=period-1)
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data):
        """
        :param data: DataFrame with 'close' column
        :return:     Series of signals: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        df = data.copy()
        rsi = self._compute_rsi(df['close'])

        signals = pd.Series(0, index=df.index)
        in_position = False

        for i in range(self.period, len(df)):
            rsi_val = rsi.iloc[i]

            if pd.isna(rsi_val):
                continue

            # Only buy when oversold AND not already holding
            if rsi_val < self.oversold and not in_position:
                signals.iloc[i] = 1
                in_position = True

            # Only sell when overbought AND currently holding
            elif rsi_val > self.overbought and in_position:
                signals.iloc[i] = -1
                in_position = False

        return signals


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    prices = 30000 + np.cumsum(np.random.randn(300) * 500)
    data = pd.DataFrame({'close': prices}, index=range(300))
    data['timestamp'] = dates

    strategy = RSIStrategy()
    signals = strategy.generate_signals(data)

    buys = (signals == 1).sum()
    sells = (signals == -1).sum()
    print(f"RSI Self-Test (Wilder's Smoothing)")
    print(f"  Buy signals : {buys}")
    print(f"  Sell signals: {sells}")
    print(f"  Balanced pairs: {buys - sells <= 1}")