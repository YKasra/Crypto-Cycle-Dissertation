# Crypto Cycle Analysis
### DSP Cycle Detection vs Traditional Indicators — Algorithmic Crypto Trading

[![Streamlit](https://img.shields.io/badge/Live_Demo-Streamlit-ff6600?style=flat-square&logo=streamlit&logoColor=white)](https://dspcycles-vs-classicindicators.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> MSc Dissertation — Computer Science for Business Management  
> Romanian-American University, Bucharest · 2025–2026  
> **Kasra Yaraei**

---

## What This Is

A backtesting system that investigates whether **Digital Signal Processing (DSP) methods** can outperform traditional technical indicators in cryptocurrency trading, measured by risk-adjusted returns.

**Research question:** Do FFT, MESA, and Hilbert Transform generate better Sharpe Ratios than MACD, RSI, and Bollinger Bands on BTC, ETH, and SOL?

---

## Methods

| Group | Algorithms |
|---|---|
| Traditional (Control) | MACD · RSI · Bollinger Bands |
| DSP Cycle Detection (Experimental) | FFT · MESA · Hilbert Transform |

**Assets:** BTC, ETH, SOL · **Period:** 2017–2026 · **Timeframe:** Daily & Hourly  
**Capital:** $10,000 · **Fees:** 0.1%/side · **Slippage:** 0.05%/side

---

## Live Demo

🔗 **[dspcycles-vs-classicindicators.streamlit.app](https://dspcycles-vs-classicindicators.streamlit.app)**

- **Backtest tab** — run any strategy on any asset, see signals, equity curve, and 20+ metrics
- **Final Results tab** — compare all 18 backtests side-by-side with Sharpe heatmap

---

## Project Structure

```
crypto_cycle_dissertation/
├── app.py                        # Streamlit dashboard
├── precompute_results.py         # Pre-run all 18 backtests → JSON cache
├── data_loader.py                # Binance historical data downloader
├── backtesting/
│   ├── engine.py                 # Backtesting execution engine
│   └── metrics.py                # 20+ performance metrics
├── strategies/
│   ├── traditional/              # MACD, RSI, Bollinger Bands
│   └── cycle_detection/          # FFT, MESA, Hilbert Transform
└── data/
    ├── raw/                      # Historical OHLCV CSVs
    └── processed/
        └── precomputed_results.json
```

---

## Run Locally

```bash
git clone https://github.com/YKasra/Crypto-Cycle-Dissertation.git
cd Crypto-Cycle-Dissertation
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

`Python` `Streamlit` `Plotly` `Pandas` `NumPy` `SciPy`