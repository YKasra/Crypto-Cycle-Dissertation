import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq


class FFTStrategy:
    """
    Fast Fourier Transform (FFT) Cycle Detection Strategy.

    Instead of using a fixed look-back window like traditional indicators,
    this strategy dynamically detects the dominant market cycle at each point
    in time using a sliding-window FFT.

    How it works:
    1. At each bar, take the last `window` closing prices
    2. Apply FFT to decompose prices into constituent sine waves
    3. Find the dominant frequency (the sine wave with highest amplitude)
    4. Reconstruct ONLY the dominant cycle (filtering out noise)
    5. Generate buy signals at cycle troughs, sell signals at cycle peaks

    Academic Reference:
    Ehlers, J.F. (2001). Rocket Science for Traders. Wiley.
    Cooley, J.W. & Tukey, J.W. (1965). An Algorithm for Machine Calculation
    of Complex Fourier Series. Mathematics of Computation, 19(90), 297-301.
    """

    def __init__(self, window=64, min_period=10, max_period=40):
        """
        :param window:     Number of bars used in each FFT window (power of 2 is faster)
        :param min_period: Minimum cycle length to consider (bars)
        :param max_period: Maximum cycle length to consider (bars)
        """
        self.window     = window
        self.min_period = min_period
        self.max_period = max_period

    def _detect_dominant_cycle(self, prices):
        """
        Apply FFT to a price window and return the dominant cycle length in bars.

        :param prices: numpy array of closing prices (length = self.window)
        :return:       dominant cycle period in bars (int)
        """
        # Detrend prices by subtracting a linear trend
        # This prevents the DC component (trend) from dominating the FFT
        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))

        # Apply Hann window to reduce spectral leakage at the edges
        hann = np.hanning(len(detrended))
        windowed = detrended * hann

        # Compute FFT
        N = len(windowed)
        yf = fft(windowed)
        xf = fftfreq(N)  # frequencies in cycles per bar

        # Take only positive frequencies (second half is mirror)
        positive_freqs = xf[:N // 2]
        amplitudes     = np.abs(yf[:N // 2])

        # Convert frequencies to periods (bars per cycle)
        # Avoid division by zero for the DC component (freq=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            periods = np.where(positive_freqs > 0, 1.0 / positive_freqs, 0)

        # Filter to only consider cycles within our min/max range
        mask = (periods >= self.min_period) & (periods <= self.max_period)

        if not mask.any():
            return self.window // 2  # fallback to half-window

        # The dominant cycle = period with highest amplitude in valid range
        dominant_period = int(round(periods[mask][np.argmax(amplitudes[mask])]))
        return dominant_period

    def _reconstruct_cycle(self, prices, dominant_period):
        """
        Reconstruct the dominant cycle using inverse FFT,
        keeping only the frequency components near the dominant period.

        :param prices:           numpy array of closing prices
        :param dominant_period:  dominant cycle length in bars
        :return:                 reconstructed cycle as numpy array
        """
        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
        hann      = np.hanning(len(detrended))
        windowed  = detrended * hann

        N  = len(windowed)
        yf = fft(windowed)
        xf = fftfreq(N)

        # Zero out all frequencies except those near the dominant cycle
        # Allow ±20% bandwidth around the dominant frequency
        dominant_freq = 1.0 / dominant_period
        bandwidth     = dominant_freq * 0.20

        filtered_yf = np.zeros_like(yf)
        for k in range(N):
            if abs(abs(xf[k]) - dominant_freq) <= bandwidth:
                filtered_yf[k] = yf[k]

        # Reconstruct signal via inverse FFT
        reconstructed = np.real(np.fft.ifft(filtered_yf))
        return reconstructed

    def generate_signals(self, data):
        """
        Generate trading signals using sliding-window FFT cycle detection.

        :param data: DataFrame with 'close' and 'timestamp' columns
        :return:     Series of signals: 1 (Buy at trough), -1 (Sell at peak), 0 (Hold)
        """
        df      = data.copy().reset_index(drop=True)
        closes  = df['close'].values
        signals = pd.Series(0, index=df.index)

        in_position      = False
        dominant_periods = []  # store for analysis/dissertation

        for i in range(self.window, len(df)):
            # Extract the current window of prices
            price_window = closes[i - self.window: i]

            # Step 1: Detect dominant cycle
            dominant_period = self._detect_dominant_cycle(price_window)
            dominant_periods.append(dominant_period)

            # Step 2: Reconstruct the dominant cycle
            cycle = self._reconstruct_cycle(price_window, dominant_period)

            # Step 3: Detect turning points using the last 3 bars of the cycle
            # Trough: cycle value is rising (prev bar was lower than bar before it)
            # Peak:   cycle value is falling (prev bar was higher than bar before it)
            if len(cycle) >= 3:
                c_prev2 = cycle[-3]
                c_prev1 = cycle[-2]
                c_curr  = cycle[-1]

                # Trough detection: cycle was falling, now rising
                is_trough = (c_prev1 < c_prev2) and (c_curr > c_prev1)
                # Peak detection: cycle was rising, now falling
                is_peak   = (c_prev1 > c_prev2) and (c_curr < c_prev1)

                if is_trough and not in_position:
                    signals.iloc[i] = 1
                    in_position = True

                elif is_peak and in_position:
                    signals.iloc[i] = -1
                    in_position = False

        # Store dominant periods as attribute for dissertation analysis
        self.dominant_periods_ = dominant_periods
        return signals


if __name__ == "__main__":
    print("=== FFT Strategy Self-Test ===\n")

    # Create synthetic data with a known 20-bar cycle + noise
    np.random.seed(42)
    N      = 400
    t      = np.arange(N)
    cycle  = 5000 * np.sin(2 * np.pi * t / 20)   # 20-bar dominant cycle
    trend  = 0.5 * t                               # slight uptrend
    noise  = np.random.randn(N) * 300              # market noise
    prices = 30000 + trend + cycle + noise

    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=N, freq='D'),
        'close':     prices
    })

    strategy = FFTStrategy(window=64, min_period=10, max_period=40)
    signals  = strategy.generate_signals(data)

    buys  = (signals == 1).sum()
    sells = (signals == -1).sum()

    print(f"Synthetic data: 20-bar cycle + noise, {N} bars")
    print(f"Buy signals  : {buys}")
    print(f"Sell signals : {sells}")
    print(f"Balanced     : {abs(buys - sells) <= 1}")

    if hasattr(strategy, 'dominant_periods_'):
        avg_period = np.mean(strategy.dominant_periods_)
        print(f"Avg detected cycle: {avg_period:.1f} bars (true cycle = 20 bars)")