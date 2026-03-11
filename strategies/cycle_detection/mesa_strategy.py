import pandas as pd
import numpy as np


class MESAStrategy:
    """
    Maximum Entropy Spectral Analysis (MESA) Cycle Detection Strategy.
    Polished version with minimum holding period to reduce overtrading.

    Key improvement over v1:
    - Added min_hold_bars parameter: once a position is entered, it must
      be held for at least N bars before a sell signal is accepted.
    - This prevents the AR prediction error from triggering rapid exits
      on short-term noise, reducing trade count and fee drag significantly.

    Academic References:
    Burg, J.P. (1967). Maximum Entropy Spectral Analysis.
    Ehlers, J.F. (2001). Rocket Science for Traders. Wiley & Sons.
    """

    def __init__(self, window=32, order=12, min_period=10, max_period=40, min_hold_bars=5):
        """
        :param window:         Bars per MESA window
        :param order:          AR model order
        :param min_period:     Minimum cycle length in bars
        :param max_period:     Maximum cycle length in bars
        :param min_hold_bars:  Minimum bars to hold before allowing a sell
        """
        self.window        = window
        self.order         = order
        self.min_period    = min_period
        self.max_period    = max_period
        self.min_hold_bars = min_hold_bars

    def _burg_method(self, x):
        N  = len(x)
        p  = self.order
        f  = x.copy().astype(float)
        b  = x.copy().astype(float)
        ar_coeffs = np.zeros(p)
        ef = f[1:]
        eb = b[:-1]
        for k in range(p):
            num = -2.0 * np.dot(eb, ef)
            den = np.dot(ef, ef) + np.dot(eb, eb)
            if den == 0:
                break
            kk           = num / den
            ar_coeffs[k] = kk
            ef_new = ef[1:]  + kk * eb[1:]
            eb_new = eb[:-1] + kk * ef[:-1]
            ef = ef_new
            eb = eb_new
        return ar_coeffs

    def _estimate_dominant_period(self, detrended):
        if detrended.std() == 0:
            return self.window // 2
        normalised = detrended / detrended.std()
        ar_coeffs  = self._burg_method(normalised)
        n_freqs    = 2048
        freqs      = np.linspace(1e-6, 0.5, n_freqs)
        psd        = np.zeros(n_freqs)
        for i, freq in enumerate(freqs):
            z      = np.exp(-1j * 2 * np.pi * freq * np.arange(1, self.order + 1))
            H      = 1.0 / (1.0 + np.dot(ar_coeffs, z))
            psd[i] = np.abs(H) ** 2
        periods = 1.0 / freqs
        mask    = (periods >= self.min_period) & (periods <= self.max_period)
        if not mask.any():
            return self.window // 2
        return int(round(periods[mask][np.argmax(psd[mask])]))

    def _extract_cycle_via_ar(self, detrended, ar_coeffs):
        p     = len(ar_coeffs)
        N     = len(detrended)
        cycle = np.zeros(N)
        for i in range(p, N):
            history = np.array([detrended[i-1-j] for j in range(p) if i-1-j >= 0])
            if len(history) < p:
                history = np.pad(history, (0, p - len(history)))
            predicted = -np.dot(ar_coeffs, history)
            cycle[i]  = detrended[i] - predicted
        return cycle

    def generate_signals(self, data):
        df      = data.copy().reset_index(drop=True)
        closes  = df['close'].values
        signals = pd.Series(0, index=df.index)

        in_position      = False
        bars_in_position = 0
        dominant_periods = []
        cycle_values     = []

        for i in range(self.window, len(df)):
            price_window = closes[i - self.window: i]
            detrended    = price_window - np.linspace(
                price_window[0], price_window[-1], len(price_window)
            )
            detrended -= detrended.mean()

            dominant_period = self._estimate_dominant_period(detrended)
            dominant_periods.append(dominant_period)

            if detrended.std() == 0:
                cycle_values.append(0)
                if in_position:
                    bars_in_position += 1
                continue

            norm_detrended = detrended / detrended.std()
            ar_coeffs      = self._burg_method(norm_detrended)
            cycle          = self._extract_cycle_via_ar(norm_detrended, ar_coeffs)
            cycle_values.append(cycle[-1])

            # Track how long we've been in position
            if in_position:
                bars_in_position += 1

            if len(cycle_values) >= 3:
                c2 = cycle_values[-3]
                c1 = cycle_values[-2]
                c0 = cycle_values[-1]

                is_trough = (c1 < c2) and (c0 > c1) and (c1 < 0)
                is_peak   = (c1 > c2) and (c0 < c1) and (c1 > 0)

                if is_trough and not in_position:
                    signals.iloc[i] = 1
                    in_position      = True
                    bars_in_position = 0

                # POLISH: only sell after minimum holding period
                elif is_peak and in_position and bars_in_position >= self.min_hold_bars:
                    signals.iloc[i] = -1
                    in_position      = False
                    bars_in_position = 0

        self.dominant_periods_ = dominant_periods
        self.cycle_values_     = cycle_values
        return signals


if __name__ == "__main__":
    print("=== MESA Strategy Self-Test (Polished) ===\n")
    np.random.seed(42)
    N      = 400
    t      = np.arange(N)
    cycle  = 5000 * np.sin(2 * np.pi * t / 20)
    trend  = 0.5 * t
    noise  = np.random.randn(N) * 300
    prices = 30000 + trend + cycle + noise
    data   = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=N, freq='D'),
        'close':     prices
    })
    strategy = MESAStrategy(window=32, order=12, min_period=10, max_period=40, min_hold_bars=5)
    signals  = strategy.generate_signals(data)
    buys  = (signals == 1).sum()
    sells = (signals == -1).sum()
    print(f"Buy signals  : {buys}")
    print(f"Sell signals : {sells}")
    print(f"Balanced     : {abs(buys - sells) <= 1}")
    print(f"Target       : significantly fewer than the previous 40 trades")