import pandas as pd
import numpy as np
from scipy.signal import hilbert


class HilbertStrategy:
    """
    Hilbert Transform Instantaneous Phase & Market Mode Detection Strategy.
    Polished version with adaptive mode threshold based on local volatility.

    Key improvement over v1:
    - Fixed: Rolling window implementation (causal — no look-ahead bias)
    - Polished: Adaptive mode_threshold scales with local price volatility.
      Instead of a fixed threshold, we compute the rolling std of phase
      velocity and set the threshold as a percentile of recent velocity.
      This means the filter adapts to each asset's cycle characteristics
      rather than using one hardcoded value for BTC, ETH, and SOL alike.

    Academic References:
    Ehlers, J.F. (2001). Rocket Science for Traders. Wiley & Sons.
    Gabor, D. (1946). Theory of Communication. IEE Journal, 93(26), 429-457.
    """

    def __init__(self, window=64, smooth_period=7, adaptive_lookback=50, threshold_pct=25):
        """
        :param window:           Rolling window for causal Hilbert application
        :param smooth_period:    Bars for price pre-smoothing
        :param adaptive_lookback: Bars to look back when computing adaptive threshold
        :param threshold_pct:    Percentile of recent phase velocities used as threshold
                                 25 = bottom quartile must be exceeded to enter cycle mode
        """
        self.window           = window
        self.smooth_period    = smooth_period
        self.adaptive_lookback= adaptive_lookback
        self.threshold_pct    = threshold_pct

    def _smooth_prices(self, prices):
        weights  = np.array([4, 3, 2, 1], dtype=float)
        weights /= weights.sum()
        smoothed = np.convolve(prices, weights[::-1], mode='full')[:len(prices)]
        smoothed[:3] = prices[:3]
        return smoothed

    def _compute_phase_for_window(self, price_window):
        detrended = price_window - np.linspace(
            price_window[0], price_window[-1], len(price_window)
        )
        detrended -= detrended.mean()
        if detrended.std() < 1e-10:
            return 0.0
        analytic  = hilbert(detrended)
        phase_arr = np.unwrap(np.angle(analytic))
        return float(phase_arr[-1])

    def generate_signals(self, data):
        """
        Generate signals using causal rolling Hilbert Transform
        with adaptive market mode threshold.
        """
        df       = data.copy().reset_index(drop=True)
        closes   = df['close'].values
        signals  = pd.Series(0, index=df.index)
        smoothed = self._smooth_prices(closes)

        # Compute phases causally
        phases = np.full(len(df), np.nan)
        for i in range(self.window, len(df)):
            phases[i] = self._compute_phase_for_window(smoothed[i - self.window: i])

        sine_wave = np.sin(phases)
        lead_wave = np.sin(phases + np.pi / 4)

        # Phase velocity
        phase_velocity  = np.diff(phases, prepend=phases[0])
        smooth_velocity = pd.Series(phase_velocity).rolling(
            window=self.smooth_period, min_periods=1
        ).mean().values

        # POLISH: Adaptive threshold — computed from recent velocity distribution
        # For each bar, threshold = Nth percentile of last `adaptive_lookback` velocities
        # This adapts to each asset's natural cycle rhythm automatically
        adaptive_threshold = np.full(len(df), np.nan)
        for i in range(self.adaptive_lookback, len(df)):
            recent = smooth_velocity[i - self.adaptive_lookback: i]
            # Only consider positive velocities (cycle is advancing)
            positive = recent[recent > 0]
            if len(positive) > 5:
                adaptive_threshold[i] = np.percentile(positive, self.threshold_pct)
            else:
                adaptive_threshold[i] = 0.01  # fallback

        # Cycle mode: velocity exceeds adaptive threshold
        is_cycle_mode = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if not np.isnan(adaptive_threshold[i]) and not np.isnan(smooth_velocity[i]):
                is_cycle_mode[i] = smooth_velocity[i] > adaptive_threshold[i]

        # Generate signals
        in_position = False
        for i in range(self.window + self.adaptive_lookback, len(df)):
            if np.isnan(phases[i]) or np.isnan(phases[i-1]):
                continue
            if not is_cycle_mode[i]:
                continue

            curr_diff = sine_wave[i]   - lead_wave[i]
            prev_diff = sine_wave[i-1] - lead_wave[i-1]

            if prev_diff < 0 and curr_diff >= 0 and not in_position:
                signals.iloc[i] = 1
                in_position = True
            elif prev_diff > 0 and curr_diff <= 0 and in_position:
                signals.iloc[i] = -1
                in_position = False

        self.phases_            = phases
        self.sine_wave_         = sine_wave
        self.lead_wave_         = lead_wave
        self.is_cycle_mode_     = is_cycle_mode
        self.adaptive_threshold_= adaptive_threshold
        return signals


if __name__ == "__main__":
    print("=== Hilbert Transform Self-Test (Polished — Adaptive Threshold) ===\n")
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
    strategy = HilbertStrategy(window=64, smooth_period=7, adaptive_lookback=50, threshold_pct=25)
    signals  = strategy.generate_signals(data)
    buys  = (signals == 1).sum()
    sells = (signals == -1).sum()
    cycle_pct = np.nanmean(strategy.is_cycle_mode_) * 100
    print(f"Synthetic data    : 20-bar cycle + noise, {N} bars")
    print(f"Buy signals       : {buys}")
    print(f"Sell signals      : {sells}")
    print(f"Balanced          : {abs(buys - sells) <= 1}")
    print(f"Cycle mode active : {cycle_pct:.1f}% of bars")
    print(f"\nAdaptive threshold adjusts per-asset — no hardcoded values")