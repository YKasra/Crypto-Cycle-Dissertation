import pandas as pd


class BollingerStrategy:
    """
    Bollinger Bands mean-reversion strategy.
    - Buy  when price touches/crosses below the lower band (oversold)
    - Sell when price touches/crosses above the upper band (overbought)

    Uses state tracking to ensure only one open position at a time,
    preventing whipsaw signals in sideways markets.
    """

    def __init__(self, period=20, std_dev=2):
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data):
        """
        :param data: DataFrame with 'close' column
        :return:     Series of signals: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        df = data.copy()

        sma = df['close'].rolling(window=self.period).mean()
        std = df['close'].rolling(window=self.period).std()
        df['upper_band'] = sma + (std * self.std_dev)
        df['lower_band'] = sma - (std * self.std_dev)

        signals = pd.Series(0, index=df.index)
        in_position = False

        for i in range(self.period, len(df)):
            price = df['close'].iloc[i]
            lower = df['lower_band'].iloc[i]
            upper = df['upper_band'].iloc[i]

            # FIX: Only buy if NOT already in a position
            if price <= lower and not in_position:
                signals.iloc[i] = 1
                in_position = True

            # Only sell if we ARE in a position
            elif price >= upper and in_position:
                signals.iloc[i] = -1
                in_position = False

        return signals


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    prices = 30000 + np.cumsum(np.random.randn(200) * 500)
    data = pd.DataFrame({'close': prices}, index=range(200))
    data['timestamp'] = dates

    strategy = BollingerStrategy()
    signals = strategy.generate_signals(data)

    buys = (signals == 1).sum()
    sells = (signals == -1).sum()
    print(f"Bollinger Bands Self-Test")
    print(f"  Buy signals : {buys}")
    print(f"  Sell signals: {sells}")
    print(f"  Signals never exceed positions: {buys - sells <= 1}")