# Crypto Cycle Analysis — MSc Dissertation

**Comparing Traditional Technical Indicators vs. Digital Signal Processing Cycle Detection for Cryptocurrency Trading**

> MSc Computer Science for Business Management · Romanian-American University · 2017–2026  
> **Kasra Yaraei**

---

## Live Dashboard

**[▶ Open the Streamlit App](https://your-app-name.streamlit.app)**  
*(replace this link after deploying to Streamlit Community Cloud)*

---

## What This Project Does

Most retail algorithmic trading relies on traditional indicators like MACD and RSI, which are inherently lagging — they use fixed look-back windows and react to price movements after they happen.

This dissertation investigates whether **Digital Signal Processing (DSP)** methods can generate earlier, more accurate trading signals by dynamically detecting the dominant market cycle at any given moment — rather than relying on fixed windows.

Six strategies are implemented, backtested on **BTC, ETH, and SOL** from **2017 to 2026**, and compared across 10+ performance metrics.

---

## Strategies Compared

| Category | Strategy | Mechanism |
|---|---|---|
| Traditional | **MACD** | EMA crossover momentum |
| Traditional | **RSI** | Overbought/oversold oscillator |
| Traditional | **Bollinger Bands** | Volatility mean-reversion |
| DSP Cycle Detection | **FFT** | Sliding-window Fast Fourier Transform |
| DSP Cycle Detection | **MESA** | Maximum Entropy Spectral Analysis |
| DSP Cycle Detection | **Hilbert Transform** | Instantaneous phase & market mode detection |

---

## Backtesting Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Initial Capital | $10,000 | Standard retail portfolio |
| Fee Rate | 0.1% per side | Standard Binance spot fee |
| Slippage | 0.05% per side | Conservative market impact |
| Total Round-Trip Cost | ~0.3% | Academically defensible retail assumption |
| Data Range | Jan 2017 – Mar 2026 | Covers full crypto market cycles |
| Assets | BTC, ETH, SOL | Large, mid, and high-volatility crypto |

---

## Key Findings

- **FFT achieved the highest average Sharpe Ratio** across all three assets, outperforming all traditional methods on a risk-adjusted basis
- Traditional indicators performed well on trending assets (BTC) but struggled with sideways/cyclical periods
- DSP methods showed lower trade frequency, resulting in significantly lower total fee costs
- Hilbert Transform on SOL produced an outlier raw return due to SOL's exceptional 2020–2024 volatility; Sharpe Ratio and Max Drawdown provide the more meaningful comparison for this case

---

## Project Structure

```
crypto-cycle-dissertation/
├── app.py                          # Streamlit dashboard (main entry point)
├── bootstrap.py                    # Auto-downloads data on first launch
├── data_loader.py                  # Binance REST API fetcher
├── precompute_results.py           # Pre-runs all 18 backtests → JSON cache
├── requirements.txt
│
├── backtesting/
│   ├── engine.py                   # Core backtesting logic (fees, slippage, equity)
│   └── metrics.py                  # 15+ performance metrics
│
├── strategies/
│   ├── traditional/
│   │   ├── macd.py
│   │   ├── rsi.py
│   │   └── bollinger.py
│   └── cycle_detection/
│       ├── fft_strategy.py
│       ├── mesa_strategy.py
│       └── hilbert_strategy.py
│
└── data/
    ├── raw/                        # CSV files (auto-downloaded, not in repo)
    └── processed/
        └── precomputed_results.json  # Cached backtest results for instant load
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/ykasra/crypto-cycle-dissertation.git
cd crypto-cycle-dissertation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download historical data (runs once, ~2-3 min)
python data_loader.py

# 4. Pre-compute all 18 backtest results (runs once, ~5-8 min)
python precompute_results.py

# 5. Launch the dashboard
streamlit run app.py
```

> **Note:** The Streamlit Cloud deployment handles steps 3 and 4 automatically on first launch.

---

## Dashboard Features

- **Backtest View** — Select any asset + strategy, run a live backtest, and explore:
  - Interactive candlestick chart with Buy/Sell signal markers
  - Performance ticker (Return, CAGR, Sharpe, Sortino, Drawdown, Win Rate)
  - Equity curve vs Buy & Hold with drawdown subplot
  - Risk analytics panels (Calmar, Recovery Factor, Profit Factor, VaR)
  - Paired trade log with Entry Price, Exit Price, P&L, and Margin per trade

- **Final Results View** — Full cross-asset comparison:
  - Bar chart comparing any metric across all 6 strategies
  - Sharpe Ratio heatmap (3 assets × 6 strategies)
  - Detailed results tables with best-in-class highlighting
  - Average Sharpe summary cards

---

## Technical Implementation Notes

- **Look-ahead bias prevention:** The Hilbert Transform uses a causal rolling window implementation — `scipy.signal.hilbert()` is never applied to the full series at once
- **Fee model:** 0.1% per side matches the standard Binance retail spot fee (no BNB discount, no VIP tier)
- **Slippage model:** 0.05% per side applied as price impact on execution (buy higher, sell lower)
- **Sharpe annualization:** √365 for daily data, √8760 for hourly data

---

## Academic References

- Ehlers, J.F. (2001). *Rocket Science for Traders*. Wiley & Sons.
- Gabor, D. (1946). Theory of Communication. *IEE Journal*, 93(26), 429–457.
- Burg, J.P. (1967). Maximum Entropy Spectral Analysis. *37th Annual SEG Meeting*.

---

## Author

**Kasra Yaraei**  
MSc Computer Science for Business Management  
Romanian-American University, Bucharest

[LinkedIn](https://linkedin.com/in/kasrayaraei) · [GitHub](https://github.com/ykasra)
