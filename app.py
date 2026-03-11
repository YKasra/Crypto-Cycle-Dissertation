import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.engine import BacktestEngine
from backtesting.metrics import calculate_metrics
from strategies.traditional.macd import MACDStrategy
from strategies.traditional.rsi import RSIStrategy
from strategies.traditional.bollinger import BollingerStrategy
from strategies.cycle_detection.fft_strategy import FFTStrategy
from strategies.cycle_detection.mesa_strategy import MESAStrategy
from strategies.cycle_detection.hilbert_strategy import HilbertStrategy

st.set_page_config(page_title="Crypto Cycle Analysis", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:     #0a0a0a;
    --bg1:    #111111;
    --bg2:    #181818;
    --bg3:    #222222;
    --border: #2a2a2a;
    --bord2:  #333333;
    --text:   #e8e8e8;
    --text2:  #999999;
    --text3:  #555555;
    --orange: #ff6600;
    --blue:   #4a9eff;
    --green:  #00c853;
    --red:    #ff3b3b;
    --yellow: #ffb300;
    --mono:   'IBM Plex Mono', monospace;
    --sans:   'IBM Plex Sans', sans-serif;
}

/* ── Hide sidebar collapse button and icon text artifact ── */
button[data-testid="baseButton-headerNoPadding"],
button[data-testid="baseButton-header"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
.st-emotion-cache-pbsa90,
.st-emotion-cache-1puwf6r,
.st-emotion-cache-czk5ss,
.st-emotion-cache-1f391cv,
.st-emotion-cache-1rtdyuf,
.st-emotion-cache-h5rgaw { display: none !important; visibility: hidden !important; }

/* Nuclear option: hide ALL buttons in sidebar (we have our own run button) */
section[data-testid="stSidebar"] > div > div > div > button { display: none !important; }

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
.main { background: var(--bg) !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text2) !important;
    font-family: var(--mono) !important;
}
section[data-testid="stSidebar"] label {
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: var(--text3) !important;
}

/* ── Button — orange bg, BLACK bold text for full visibility ── */
.stButton > button {
    background: var(--orange) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 10px 0 !important;
    width: 100% !important;
}
.stButton > button:hover { background: #e55a00 !important; }
.stButton > button p { color: #000000 !important; font-weight: 700 !important; }

/* ── Top bar ── */
.topbar {
    background: var(--bg1);
    border-bottom: 1px solid var(--border);
    padding: 14px 0 12px 0;
    margin: 0 0 20px 0;
    display: flex;
    align-items: baseline;
    gap: 20px;
    flex-wrap: wrap;
}
.topbar-title {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.topbar-title span { color: var(--orange); }
.topbar-sep  { color: var(--bord2); font-family: var(--mono); }
.topbar-meta { font-family: var(--mono); font-size: 0.65rem; color: var(--text3); letter-spacing: 1px; text-transform: uppercase; }
.topbar-tag  { font-family: var(--mono); font-size: 0.6rem; font-weight: 600; padding: 2px 8px; border-radius: 2px; text-transform: uppercase; letter-spacing: 1px; }
.tag-t { background: #1a2a1a; color: #4caf50; border: 1px solid #2d5c2d; }
.tag-d { background: #1a1a2e; color: var(--blue); border: 1px solid #1e3a5f; }

/* ── Info strip ── */
.istrip { display: flex; gap: 0; border: 1px solid var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 20px; }
.istrip-item { flex: 1; padding: 12px 16px; border-right: 1px solid var(--border); background: var(--bg1); }
.istrip-item:last-child { border-right: none; }
.istrip-label { font-family: var(--mono); font-size: 0.56rem; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.istrip-val   { font-family: var(--mono); font-size: 0.95rem; font-weight: 600; color: var(--text); }

/* ── Section header ── */
.sh { font-family: var(--mono); font-size: 0.6rem; font-weight: 600; color: var(--text3);
    text-transform: uppercase; letter-spacing: 2px; padding: 0 0 6px 0;
    border-bottom: 1px solid var(--border); margin: 20px 0 12px 0; }
.sh span { color: var(--orange); margin-right: 8px; }

/* ── Ticker strip ── */
.ticker-strip { display: flex; gap: 32px; padding: 10px 0 14px 0;
    border-bottom: 1px solid var(--border); margin-bottom: 20px; overflow-x: auto; }
.ticker-item  { display: flex; flex-direction: column; gap: 2px; min-width: 110px; }
.ticker-label { font-family: var(--mono); font-size: 0.58rem; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; }
.ticker-val   { font-family: var(--mono); font-size: 1.1rem; font-weight: 600; color: var(--text); }
.ticker-val.up   { color: var(--green); }
.ticker-val.down { color: var(--red); }
.ticker-val.neu  { color: var(--blue); }
.ticker-val.warn { color: var(--yellow); }

/* ── Data panel ── */
.dpanel { background: var(--bg1); border: 1px solid var(--border); border-top: 2px solid var(--bord2); padding: 16px 18px; }
.dpanel.accent-o { border-top-color: var(--orange); }
.dpanel.accent-b { border-top-color: var(--blue); }
.dpanel.accent-g { border-top-color: var(--green); }
.dpanel.accent-r { border-top-color: var(--red); }
.dpanel-title { font-family: var(--mono); font-size: 0.58rem; color: var(--text3); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; }

/* ── Stat row ── */
.srow { display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid var(--border); }
.srow:last-child { border-bottom: none; }
.srow-label { font-family: var(--mono); color: var(--text2); font-size: 0.68rem; }
.srow-val   { font-family: var(--mono); font-weight: 600; color: var(--text); font-size: 0.78rem; }
.srow-val.g { color: var(--green); }
.srow-val.r { color: var(--red); }

/* ── Ratio bar ── */
.rb-wrap { margin-bottom: 12px; }
.rb-top  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.rb-name { font-family: var(--mono); font-size: 0.62rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.8px; }
.rb-right{ display: flex; align-items: center; gap: 8px; }
.rb-num  { font-family: var(--mono); font-size: 0.82rem; font-weight: 600; }
.rb-pill { font-family: var(--mono); font-size: 0.5rem; font-weight: 600; padding: 1px 5px; border-radius: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
.rb-track{ background: var(--bg3); height: 3px; border-radius: 1px; }
.rb-fill { height: 3px; border-radius: 1px; }

/* ── Win/loss bar ── */
.wl-bar { display: flex; height: 16px; border-radius: 2px; overflow: hidden; margin: 8px 0 14px 0; }
.wl-w { background: #0d2b0d; border-right: 1px solid var(--bg); display: flex; align-items: center;
    justify-content: flex-end; padding-right: 6px;
    font-family: var(--mono); font-size: 0.58rem; font-weight: 600; color: var(--green); }
.wl-l { background: #2b0d0d; display: flex; align-items: center; justify-content: flex-start; padding-left: 6px;
    font-family: var(--mono); font-size: 0.58rem; font-weight: 600; color: var(--red); }

/* ── Final Results comparison table ── */
.fr-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.72rem; }
.fr-table th { background: var(--bg2); color: var(--text3); font-size: 0.58rem; text-transform: uppercase;
    letter-spacing: 1px; padding: 10px 12px; text-align: right; border-bottom: 2px solid var(--orange); font-weight: 600; }
.fr-table th:first-child { text-align: left; }
.fr-table td { padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); color: var(--text2); }
.fr-table td:first-child { text-align: left; color: var(--text); font-weight: 600; }
.fr-table tr:hover td { background: var(--bg2); }
.fr-table tr.section-dsp td { background: #0d0d1a; }
.fr-table tr.section-dsp td:first-child { color: var(--blue); }
.fr-table tr.section-trad td { background: #0d1a0d; }
.fr-table tr.section-trad td:first-child { color: #4caf50; }
.fr-table .best  { color: var(--orange) !important; font-weight: 700 !important; }
.fr-table .pos   { color: var(--green); }
.fr-table .neg   { color: var(--red); }
.fr-table .neu   { color: var(--blue); }

.asset-header { font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
    color: var(--orange); text-transform: uppercase; letter-spacing: 2px;
    padding: 14px 0 6px 0; border-bottom: 1px solid var(--border); margin: 24px 0 0 0; }

.div { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

/* ── Trade log table ── */
.tl-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.72rem; }
.tl-table th { background: var(--bg2); color: var(--text3); font-size: 0.58rem; text-transform: uppercase;
    letter-spacing: 1px; padding: 10px 12px; text-align: right; border-bottom: 2px solid var(--orange); font-weight: 600; }
.tl-table th:first-child { text-align: left; }
.tl-table td { padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); color: var(--text2); }
.tl-table td:first-child { text-align: left; }
.tl-table tr:hover td { background: var(--bg2); }
.tl-table .pos { color: var(--green); font-weight: 600; }
.tl-table .neg { color: var(--red);   font-weight: 600; }
.tl-open { font-family: var(--mono); font-size: 0.6rem; color: var(--yellow);
    padding: 8px 0; letter-spacing: 0.5px; }
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STRATEGY_MAP = {
    'MACD':            (MACDStrategy,     {},                                                   'traditional'),
    'RSI':             (RSIStrategy,      {},                                                   'traditional'),
    'Bollinger Bands': (BollingerStrategy,{},                                                   'traditional'),
    'FFT':             (FFTStrategy,      {'window':64,'min_period':10,'max_period':40},         'dsp'),
    'MESA':            (MESAStrategy,     {'window':32,'order':12,'min_period':10,
                                           'max_period':40,'min_hold_bars':5},                  'dsp'),
    'Hilbert':         (HilbertStrategy,  {'window':64,'smooth_period':7,
                                           'adaptive_lookback':50,'threshold_pct':25},          'dsp'),
}
ASSET_LABELS = {'BTCUSDT':'Bitcoin  BTC','ETHUSDT':'Ethereum  ETH','SOLUSDT':'Solana  SOL'}
ASSET_SHORT  = {'BTCUSDT':'BTC','ETHUSDT':'ETH','SOLUSDT':'SOL'}
ALL_STRATEGIES = ['MACD','RSI','Bollinger Bands','FFT','MESA','Hilbert']

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(symbol, interval):
    path = f"data/raw/{symbol}_{interval}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

PRECOMPUTED_PATH = 'data/processed/precomputed_results.json'

@st.cache_data
def load_precomputed():
    if not os.path.exists(PRECOMPUTED_PATH):
        return None
    with open(PRECOMPUTED_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def run_all_strategies(symbol, interval):
    precomputed = load_precomputed()
    if precomputed and symbol in precomputed:
        return precomputed[symbol]
    df = load_data(symbol, interval)
    if df is None: return {}
    results = {}
    for name in ALL_STRATEGIES:
        try:
            cls, kwargs, stype = STRATEGY_MAP[name]
            signals = cls(**kwargs).generate_signals(df)
            engine  = BacktestEngine(initial_capital=10000)
            eq, tr  = engine.run(df, signals)
            met     = calculate_metrics(eq, tr, interval=interval)
            met['_type'] = stype
            met['_bh']   = (df['close'].iloc[-1]/df['close'].iloc[0]-1)*100
            results[name] = met
        except Exception as e:
            results[name] = {'_error': str(e)}
    return results

def run_strategy(df, name, interval):
    cls, kwargs, stype = STRATEGY_MAP[name]
    strat   = cls(**kwargs)
    signals = strat.generate_signals(df)
    engine  = BacktestEngine(initial_capital=10000)
    eq, tr  = engine.run(df, signals)
    met     = calculate_metrics(eq, tr, interval=interval)

    # ── Extract cycle waveform for DSP strategies ─────────────────────────
    cycle_data = None
    if stype == 'dsp':
        n = len(df)
        if name == 'FFT' and hasattr(strat, 'dominant_periods_'):
            # Re-run reconstruction on the last window to get full-length cycle
            closes = df['close'].values
            w      = strat.window
            cycle_arr = np.full(n, np.nan)
            for i in range(w, n):
                dp = strat._detect_dominant_cycle(closes[i - w: i])
                rc = strat._reconstruct_cycle(closes[i - w: i], dp)
                cycle_arr[i] = rc[-1]
            cycle_data = {
                'type':   'fft',
                'cycle':  cycle_arr,
                'label':  'Reconstructed Dominant Cycle',
                'color':  '#4a9eff',
            }

        elif name == 'MESA' and hasattr(strat, 'cycle_values_'):
            cycle_arr = np.full(n, np.nan)
            start     = strat.window
            vals      = strat.cycle_values_
            for j, val in enumerate(vals):
                idx = start + j
                if idx < n:
                    cycle_arr[idx] = val
            cycle_data = {
                'type':  'mesa',
                'cycle': cycle_arr,
                'label': 'MESA AR Cycle (normalised)',
                'color': '#00c853',
            }

        elif name == 'Hilbert' and hasattr(strat, 'sine_wave_'):
            cycle_data = {
                'type':      'hilbert',
                'sine':      strat.sine_wave_,
                'lead':      strat.lead_wave_,
                'mode':      strat.is_cycle_mode_.astype(float),
                'label':     'Sine Wave Indicator',
                'color_s':   '#4a9eff',
                'color_l':   '#ff6600',
            }

    return signals, eq, tr, met, cycle_data

def rating(v, good, ok):
    if v >= good: return 'var(--green)','#0d2b0d','GOOD'
    if v >= ok:   return 'var(--yellow)','#2b2200','OK'
    return 'var(--red)','#2b0d0d','POOR'

def rb(label, value, max_val, fg, bg, pill):
    pct = min(max(value/max_val*100,0),100) if max_val else 0
    return f"""<div class="rb-wrap">
  <div class="rb-top">
    <span class="rb-name">{label}</span>
    <div class="rb-right">
      <span class="rb-num" style="color:{fg}">{value:.2f}</span>
      <span class="rb-pill" style="background:{bg};color:{fg}">{pill}</span>
    </div>
  </div>
  <div class="rb-track"><div class="rb-fill" style="width:{pct}%;background:{fg}"></div></div>
</div>"""

def sr(label, val, cls=''):
    return f'<div class="srow"><span class="srow-label">{label}</span><span class="srow-val {cls}">{val}</span></div>'

def ticker(label, val, cls=''):
    return f'<div class="ticker-item"><div class="ticker-label">{label}</div><div class="ticker-val {cls}">{val}</div></div>'

def build_price_chart(df, signals, cycle_data=None):
    has_cycle = cycle_data is not None
    if has_cycle:
        row_heights    = [0.58, 0.16, 0.26]
        rows           = 3
        subplot_titles = ('', '', cycle_data['label'])
    else:
        row_heights    = [0.78, 0.22]
        rows           = 2
        subplot_titles = ('', '')

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.02,
        subplot_titles=subplot_titles
    )

    # ── Row 1: Candlestick + signals ──────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_line_color='#00c853', decreasing_line_color='#ff3b3b',
        increasing_fillcolor='#00c853', decreasing_fillcolor='#ff3b3b'),
        row=1, col=1)

    buys  = df[signals == 1]
    sells = df[signals == -1]
    if len(buys):
        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['low']*0.983, mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', size=9, color='#00c853',
                        line=dict(color='#0a0a0a', width=0.5))), row=1, col=1)
    if len(sells):
        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['high']*1.017, mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', size=9, color='#ff3b3b',
                        line=dict(color='#0a0a0a', width=0.5))), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────────────────────────────
    colors = ['#00c853' if c >= o else '#ff3b3b'
              for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['volume'],
        marker_color=colors, name='Volume', opacity=0.4), row=2, col=1)

    # ── Row 3: DSP Cycle Waveform ─────────────────────────────────────────
    if has_cycle:
        t = df['timestamp']
        ctype = cycle_data['type']

        if ctype == 'fft':
            fig.add_trace(go.Scatter(
                x=t, y=cycle_data['cycle'], mode='lines',
                name=cycle_data['label'],
                line=dict(color=cycle_data['color'], width=1.5)),
                row=3, col=1)
            # Zero line
            fig.add_hline(y=0, line_color='#333333', line_width=1, row=3, col=1)

        elif ctype == 'mesa':
            fig.add_trace(go.Scatter(
                x=t, y=cycle_data['cycle'], mode='lines',
                name=cycle_data['label'],
                line=dict(color=cycle_data['color'], width=1.5)),
                row=3, col=1)
            fig.add_hline(y=0, line_color='#333333', line_width=1, row=3, col=1)

        elif ctype == 'hilbert':
            # Shade cycle mode regions in background
            mode   = cycle_data['mode']
            in_seg = False
            x0_seg = None
            for i, m in enumerate(mode):
                if m > 0 and not in_seg:
                    in_seg = True
                    x0_seg = t.iloc[i]
                elif m == 0 and in_seg:
                    in_seg = False
                    fig.add_vrect(
                        x0=x0_seg, x1=t.iloc[i],
                        fillcolor='rgba(74,158,255,0.06)',
                        line_width=0, row=3, col=1)
            if in_seg:
                fig.add_vrect(
                    x0=x0_seg, x1=t.iloc[-1],
                    fillcolor='rgba(74,158,255,0.06)',
                    line_width=0, row=3, col=1)

            fig.add_trace(go.Scatter(
                x=t, y=cycle_data['sine'], mode='lines',
                name='Sine Wave',
                line=dict(color=cycle_data['color_s'], width=1.5)),
                row=3, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=cycle_data['lead'], mode='lines',
                name='Lead Wave',
                line=dict(color=cycle_data['color_l'], width=1.5, dash='dot')),
                row=3, col=1)
            fig.add_hline(y=0, line_color='#333333', line_width=1, row=3, col=1)

    fig.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(family='IBM Plex Mono', color='#555555', size=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=9, color='#999999')),
        margin=dict(l=0, r=0, t=4, b=0),
        height=560 if has_cycle else 450)
    fig.update_xaxes(gridcolor='#1a1a1a', zeroline=False)
    fig.update_yaxes(gridcolor='#1a1a1a', zeroline=False)

    # Style the cycle subplot title
    if has_cycle and fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.font.color  = '#555555'
            ann.font.size   = 9
            ann.font.family = 'IBM Plex Mono'

    return fig

def build_equity_chart(eq_df, df):
    bh  = 10000*df['close']/df['close'].iloc[0]
    eq_s= pd.Series(eq_df['equity'].values)
    dd  = (eq_s-eq_s.cummax())/eq_s.cummax()*100
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.67,0.33],vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=df['timestamp'],y=bh,mode='lines',name='Buy & Hold',
        line=dict(color='#333333',width=1.5,dash='dot')),row=1,col=1)
    fig.add_trace(go.Scatter(x=eq_df['timestamp'],y=eq_df['equity'],mode='lines',name='Strategy',
        line=dict(color='#ff6600',width=2),fill='tozeroy',fillcolor='rgba(255,102,0,0.05)'),row=1,col=1)
    fig.add_trace(go.Scatter(x=eq_df['timestamp'],y=dd,mode='lines',name='Drawdown %',
        line=dict(color='#ff3b3b',width=1),fill='tozeroy',fillcolor='rgba(255,59,59,0.1)'),row=2,col=1)
    fig.update_layout(paper_bgcolor='#0a0a0a',plot_bgcolor='#111111',
        font=dict(family='IBM Plex Mono',color='#555555',size=10),
        legend=dict(orientation='h',yanchor='bottom',y=1.03,bgcolor='rgba(0,0,0,0)',font=dict(size=9,color='#999999')),
        margin=dict(l=0,r=0,t=4,b=0),height=320)
    fig.update_xaxes(gridcolor='#1a1a1a',zeroline=False)
    fig.update_yaxes(gridcolor='#1a1a1a',zeroline=False)
    return fig

def build_comparison_chart(all_results, assets, metric, metric_label):
    strategies = ALL_STRATEGIES
    colors_t = '#4caf50'
    colors_d = '#4a9eff'
    fig = go.Figure()
    for asset in assets:
        vals = []
        bar_colors = []
        for s in strategies:
            r = all_results.get(asset, {}).get(s, {})
            vals.append(r.get(metric, 0))
            bar_colors.append(colors_d if STRATEGY_MAP[s][2]=='dsp' else colors_t)
        fig.add_trace(go.Bar(
            name=ASSET_SHORT[asset], x=strategies, y=vals,
            marker_color=bar_colors if len(assets)==1 else None,
            text=[f"{v:.2f}" for v in vals],
            textposition='outside', textfont=dict(size=9, color='#999999')))
    fig.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(family='IBM Plex Mono', color='#999999', size=10),
        barmode='group',
        title=dict(text=metric_label, font=dict(color='#ff6600', size=11), x=0),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#999999')),
        margin=dict(l=0,r=0,t=40,b=0), height=300,
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a'))
    return fig

def render_comparison_table(all_results, asset):
    df_res = load_data(asset, '1d')
    bh = (df_res['close'].iloc[-1]/df_res['close'].iloc[0]-1)*100 if df_res is not None else 0

    metrics_cols = [
        ('Total Return (%)',         'Return %',    lambda v: f"{v:+.1f}%",   True),
        ('CAGR (%)',                  'CAGR %',      lambda v: f"{v:+.1f}%",   True),
        ('Sharpe Ratio',              'Sharpe',      lambda v: f"{v:.3f}",     True),
        ('Sortino Ratio',             'Sortino',     lambda v: f"{v:.3f}",     True),
        ('Calmar Ratio',              'Calmar',      lambda v: f"{v:.3f}",     True),
        ('Max Drawdown (%)',          'Max DD',      lambda v: f"{v:.1f}%",    False),
        ('Win Rate (%)',              'Win %',       lambda v: f"{v:.1f}%",    True),
        ('Profit Factor',             'Prof.F',      lambda v: f"{v:.2f}",     True),
        ('Total Trades',              'Trades',      lambda v: str(int(v)),    None),
        ('Total Fees Paid ($)',       'Fees $',      lambda v: f"${v:,.0f}",   False),
    ]

    best = {}
    for key, _, _, higher_better in metrics_cols:
        if higher_better is None: continue
        vals = []
        for name in ALL_STRATEGIES:
            r = all_results.get(asset, {}).get(name, {})
            vals.append(r.get(key, None))
        valid = [v for v in vals if v is not None]
        if valid:
            best[key] = max(valid) if higher_better else min(valid)

    header_cells = '<th>Strategy</th>' + ''.join(f'<th>{col}</th>' for _, col, _, _ in metrics_cols)
    rows_html = f'<tr>{header_cells}</tr>'

    for name in ALL_STRATEGIES:
        r = all_results.get(asset, {}).get(name, {})
        if '_error' in r:
            continue
        stype   = STRATEGY_MAP[name][2]
        row_cls = 'section-dsp' if stype=='dsp' else 'section-trad'
        cells   = f'<td>{name}</td>'
        for key, _, fmt, higher_better in metrics_cols:
            v = r.get(key, None)
            if v is None:
                cells += '<td>—</td>'
                continue
            formatted = fmt(v)
            is_best   = (higher_better is not None) and (best.get(key) == v)
            if is_best:
                cls = 'best'
            elif higher_better is True  and isinstance(v, float) and v > 0:
                cls = 'pos'
            elif higher_better is False and isinstance(v, float) and v < 0:
                cls = 'neg'
            else:
                cls = ''
            cells += f'<td class="{cls}">{formatted}</td>'
        rows_html += f'<tr class="{row_cls}">{cells}</tr>'

    bh_cells = f'<td style="color:#ff6600">Buy & Hold</td>'
    bh_cells += f'<td class="neu">{bh:+.1f}%</td>'
    bh_cells += '<td>—</td>'*( len(metrics_cols)-1)
    rows_html += f'<tr>{bh_cells}</tr>'

    return f'<table class="fr-table">{rows_html}</table>'


def build_trade_log(tr_df):
    buys  = tr_df[tr_df['side'] == 'buy'].reset_index(drop=True)
    sells = tr_df[tr_df['side'] == 'sell'].reset_index(drop=True)
    n     = min(len(buys), len(sells))

    if n == 0:
        return '<div class="tl-open">No completed trades.</div>', False, None

    rows_html = """
    <tr>
      <th>#</th>
      <th style="text-align:left">Open Date</th>
      <th style="text-align:left">Close Date</th>
      <th>Entry Price</th>
      <th>Exit Price</th>
      <th>Fees</th>
      <th>P&amp;L ($)</th>
      <th>Margin (%)</th>
    </tr>"""

    for i in range(n):
        entry_price = buys['price'].iloc[i]
        exit_price  = sells['price'].iloc[i]
        fees        = buys['fee'].iloc[i] + sells['fee'].iloc[i]
        pnl         = sells['value'].iloc[i] - buys['value'].iloc[i]
        margin_pct  = (exit_price - entry_price) / entry_price * 100
        open_date   = pd.to_datetime(buys['timestamp'].iloc[i]).strftime('%d %b %Y')
        close_date  = pd.to_datetime(sells['timestamp'].iloc[i]).strftime('%d %b %Y')

        pnl_cls    = 'pos' if pnl    >= 0 else 'neg'
        margin_cls = 'pos' if margin_pct >= 0 else 'neg'
        pnl_str    = f"+${pnl:,.2f}"    if pnl    >= 0 else f"-${abs(pnl):,.2f}"
        margin_str = f"+{margin_pct:.2f}%" if margin_pct >= 0 else f"{margin_pct:.2f}%"

        rows_html += f"""
        <tr>
          <td style="text-align:left;color:#555555">{i+1}</td>
          <td style="text-align:left;color:#999999">{open_date}</td>
          <td style="text-align:left;color:#999999">{close_date}</td>
          <td>${entry_price:,.2f}</td>
          <td>${exit_price:,.2f}</td>
          <td style="color:#555555">${fees:,.2f}</td>
          <td class="{pnl_cls}">{pnl_str}</td>
          <td class="{margin_cls}">{margin_str}</td>
        </tr>"""

    has_open   = len(buys) > len(sells)
    open_price = buys['price'].iloc[-1] if has_open else None
    table_html = f'<table class="tl-table">{rows_html}</table>'
    return table_html, has_open, open_price


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**CONFIGURATION**")
    st.markdown('<hr style="border-top:1px solid #2a2a2a;margin:8px 0">', unsafe_allow_html=True)

    view_mode = st.radio("View", ["Backtest", "Final Results"])

    st.markdown('<hr style="border-top:1px solid #2a2a2a;margin:8px 0">', unsafe_allow_html=True)

    if view_mode == "Backtest":
        symbol   = st.selectbox("Asset", list(ASSET_LABELS.keys()), format_func=lambda x: ASSET_LABELS[x])
        interval = st.selectbox("Timeframe", ["1d","1h"], format_func=lambda x:"Daily" if x=="1d" else "Hourly")
        st.markdown('<hr style="border-top:1px solid #2a2a2a;margin:8px 0">', unsafe_allow_html=True)
        cat = st.radio("Category", ["Traditional Indicators","DSP Cycle Detection"])
        strategy_name = st.selectbox("Strategy",
            ["MACD","RSI","Bollinger Bands"] if cat=="Traditional Indicators" else ["FFT","MESA","Hilbert"])
        st.markdown('<hr style="border-top:1px solid #2a2a2a;margin:8px 0">', unsafe_allow_html=True)
        run_btn = st.button("Run Backtest")
    else:
        symbol        = 'BTCUSDT'
        interval      = '1d'
        strategy_name = 'FFT'
        run_btn       = False

    st.markdown('<hr style="border-top:1px solid #2a2a2a;margin:8px 0">', unsafe_allow_html=True)
    st.markdown("""<div style='font-family:IBM Plex Mono,monospace;font-size:0.58rem;
        color:#333333;line-height:2.2;text-transform:uppercase;letter-spacing:1px'>
        MSc Computer Science<br>Romanian-American University<br>
        DSP vs Traditional Indicators<br>2017 — 2026<br>
        <span style='color:#ff6600'>Kasra Yaraei</span>
        </div>""", unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────────────────────────────
stype   = STRATEGY_MAP[strategy_name][2]
tag_cls = 'tag-d' if stype=='dsp' else 'tag-t'
tag_txt = 'DSP' if stype=='dsp' else 'TRADITIONAL'

if view_mode == "Backtest":
    tag_display = f'<span class="topbar-tag {tag_cls}">{tag_txt} — {strategy_name}</span>'
else:
    tag_display = '<span class="topbar-tag tag-d">FINAL RESULTS — ALL STRATEGIES</span>'

st.markdown(f"""
<div class="topbar">
  <span class="topbar-title">CRYPTO <span>CYCLE</span> ANALYSIS</span>
  <span class="topbar-sep">|</span>
  <span class="topbar-meta">MSc Dissertation &nbsp;·&nbsp; Romanian-American University &nbsp;·&nbsp; 2017–2026</span>
  <span class="topbar-sep">|</span>
  {tag_display}
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL RESULTS VIEW
# ═══════════════════════════════════════════════════════════════════════════════
if view_mode == "Final Results":

    st.markdown('<div class="sh"><span>01</span>COMPARATIVE ANALYSIS — ALL STRATEGIES × ALL ASSETS</div>',
                unsafe_allow_html=True)

    asset_options = {
        'All Assets': list(ASSET_LABELS.keys()),
        'BTC vs ETH':  ['BTCUSDT','ETHUSDT'],
        'BTC vs SOL':  ['BTCUSDT','SOLUSDT'],
        'ETH vs SOL':  ['ETHUSDT','SOLUSDT'],
        'BTC Only':    ['BTCUSDT'],
        'ETH Only':    ['ETHUSDT'],
        'SOL Only':    ['SOLUSDT'],
    }

    c1, c2, _ = st.columns([2, 2, 4])
    with c1:
        asset_filter = st.selectbox("Asset Filter", list(asset_options.keys()))
    with c2:
        chart_metric = st.selectbox("Chart Metric", [
            "Sharpe Ratio", "Total Return (%)", "CAGR (%)",
            "Sortino Ratio", "Max Drawdown (%)", "Win Rate (%)"])

    selected_assets = asset_options[asset_filter]

    show_results = st.button("Show Results")

    if not show_results:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;
              color:#333333;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px'>
            Configure filters above — then click Show Results
          </div>
          <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#222222'>
            All 18 backtests precomputed — results load instantly
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    precomputed = load_precomputed()

    if precomputed is None:
        st.markdown("""
        <div style='background:#1a0d00;border:1px solid #ff6600;border-radius:4px;
            padding:16px 20px;font-family:IBM Plex Mono,monospace;'>
          <div style='color:#ff6600;font-size:0.7rem;font-weight:600;
              text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
            Precomputed results not found
          </div>
          <div style='color:#999999;font-size:0.68rem;line-height:1.8'>
            Run this command once in your terminal:<br>
            <code style='color:#ff6600'>python precompute_results.py</code><br><br>
            Takes 3-8 minutes once. After that Final Results loads instantly.
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    all_results = {}
    for asset in list(ASSET_LABELS.keys()):
        all_results[asset] = {
            k: v for k, v in precomputed.get(asset, {}).items()
            if not k.startswith('_meta')
        }

    st.markdown('''<div style="font-family:IBM Plex Mono,monospace;font-size:0.55rem;
        color:#333333;margin-bottom:16px;letter-spacing:1px">
        LOADED FROM PRECOMPUTED CACHE &nbsp;·&nbsp; INSTANT
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="sh"><span>02</span>METRIC COMPARISON CHART</div>', unsafe_allow_html=True)
    fig_comp = build_comparison_chart(all_results, selected_assets, chart_metric, chart_metric)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#444444;
        display:flex;gap:20px;margin-bottom:16px'>
      <span style='color:#4caf50'>— TRADITIONAL</span>
      <span style='color:#4a9eff'>— DSP CYCLE DETECTION</span>
      <span style='color:#ff6600'>— BEST IN CLASS</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh"><span>03</span>SHARPE RATIO HEATMAP</div>', unsafe_allow_html=True)

    sharpe_data = []
    for asset in ['BTCUSDT','ETHUSDT','SOLUSDT']:
        row = {'Asset': ASSET_SHORT[asset]}
        for name in ALL_STRATEGIES:
            r = all_results.get(asset, {}).get(name, {})
            row[name] = r.get('Sharpe Ratio', 0) if isinstance(r, dict) else 0
        sharpe_data.append(row)

    sharpe_df = pd.DataFrame(sharpe_data).set_index('Asset')
    z = sharpe_df.values.tolist()
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=ALL_STRATEGIES, y=['BTC','ETH','SOL'],
        colorscale=[[0,'#2b0d0d'],[0.4,'#2b2200'],[0.7,'#0d2b0d'],[1,'#00c853']],
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=11, family='IBM Plex Mono'),
        showscale=True, zmin=-0.5, zmax=1.5))
    fig_heat.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#111111',
        font=dict(family='IBM Plex Mono', color='#999999', size=11),
        margin=dict(l=0,r=0,t=10,b=0), height=200)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="sh"><span>04</span>DETAILED RESULTS TABLES</div>', unsafe_allow_html=True)

    for asset in selected_assets:
        st.markdown(f'<div class="asset-header">{ASSET_LABELS[asset]}</div>', unsafe_allow_html=True)
        table_html = render_comparison_table(all_results, asset)
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom:8px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sh"><span>05</span>AVERAGE SHARPE — CROSS-ASSET SUMMARY</div>',
                unsafe_allow_html=True)

    avg_sharpe = {}
    for name in ALL_STRATEGIES:
        vals = []
        for asset in ['BTCUSDT','ETHUSDT','SOLUSDT']:
            r = all_results.get(asset, {}).get(name, {})
            v = r.get('Sharpe Ratio', None) if isinstance(r, dict) else None
            if v is not None:
                vals.append(v)
        avg_sharpe[name] = np.mean(vals) if vals else 0

    sorted_strats = sorted(avg_sharpe.items(), key=lambda x: x[1], reverse=True)
    best_avg = max(avg_sharpe.values())
    cols = st.columns(6)
    for col, (name, avg) in zip(cols, sorted_strats):
        stype = STRATEGY_MAP[name][2]
        color = '#4a9eff' if stype=='dsp' else '#4caf50'
        accent = '#ff6600' if avg == best_avg else color
        with col:
            st.markdown(f"""
            <div style='background:#111111;border:1px solid #2a2a2a;
                border-top:2px solid {accent};padding:14px 12px;text-align:center'>
              <div style='font-family:IBM Plex Mono,monospace;font-size:0.55rem;
                  color:#555555;text-transform:uppercase;letter-spacing:1px;
                  margin-bottom:6px'>{name}</div>
              <div style='font-family:IBM Plex Mono,monospace;font-size:1.2rem;
                  font-weight:600;color:{accent}'>{avg:.3f}</div>
              <div style='font-family:IBM Plex Mono,monospace;font-size:0.5rem;
                  color:#333333;text-transform:uppercase;margin-top:4px'>avg sharpe</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST VIEW
# ═══════════════════════════════════════════════════════════════════════════════
else:
    df = load_data(symbol, interval)
    if df is None:
        st.error("Data not found.")
        st.stop()

    st.markdown(f"""
    <div class="istrip">
      <div class="istrip-item"><div class="istrip-label">Asset</div><div class="istrip-val">{ASSET_SHORT[symbol]}</div></div>
      <div class="istrip-item"><div class="istrip-label">Interval</div><div class="istrip-val">{"Daily" if interval=="1d" else "Hourly"}</div></div>
      <div class="istrip-item"><div class="istrip-label">From</div><div class="istrip-val">{df['timestamp'].iloc[0].strftime('%d %b %Y')}</div></div>
      <div class="istrip-item"><div class="istrip-label">To</div><div class="istrip-val">{df['timestamp'].iloc[-1].strftime('%d %b %Y')}</div></div>
      <div class="istrip-item"><div class="istrip-label">Bars</div><div class="istrip-val">{len(df):,}</div></div>
      <div class="istrip-item"><div class="istrip-label">Strategy</div><div class="istrip-val">{strategy_name}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh"><span>01</span>PRICE CHART & SIGNALS</div>', unsafe_allow_html=True)

    if run_btn:
        with st.spinner(""):
            try:
                signals, eq_df, tr_df, met, cycle_data = run_strategy(df, strategy_name, interval)
                st.plotly_chart(build_price_chart(df, signals, cycle_data), use_container_width=True)

                ret   = met.get('Total Return (%)',0)
                cagr  = met.get('CAGR (%)',0)
                sh    = met.get('Sharpe Ratio',0)
                so    = met.get('Sortino Ratio',0)
                dd    = met.get('Max Drawdown (%)',0)
                wr    = met.get('Win Rate (%)',0)
                trades= met.get('Total Trades',0)
                bh    = (df['close'].iloc[-1]/df['close'].iloc[0]-1)*100

                st.markdown(f"""
                <div class="sh"><span>02</span>PERFORMANCE SUMMARY</div>
                <div class="ticker-strip">
                  {ticker("Total Return",  f"{ret:+.1f}%",  'up' if ret>0 else 'down')}
                  {ticker("CAGR",          f"{cagr:+.1f}%", 'up' if cagr>0 else 'down')}
                  {ticker("Sharpe Ratio",  f"{sh:.3f}",     'up' if sh>0.5 else ('warn' if sh>0 else 'down'))}
                  {ticker("Sortino Ratio", f"{so:.3f}",     'up' if so>0.5 else ('warn' if so>0 else 'down'))}
                  {ticker("Max Drawdown",  f"{dd:.1f}%",    'down')}
                  {ticker("Win Rate",      f"{wr:.1f}%",    'up' if wr>=50 else 'down')}
                  {ticker("Trades",        str(trades),     'neu')}
                  {ticker("B&H Return",    f"{bh:+.1f}%",   'neu')}
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="sh"><span>03</span>EQUITY CURVE & DRAWDOWN</div>', unsafe_allow_html=True)
                st.plotly_chart(build_equity_chart(eq_df, df), use_container_width=True)

                st.markdown('<div class="sh"><span>04</span>ANALYTICS</div>', unsafe_allow_html=True)
                col_l, col_r = st.columns(2)

                with col_l:
                    ca=met.get('Calmar Ratio',0); rf=met.get('Recovery Factor',0)
                    pf=met.get('Profit Factor',0); pr=met.get('Payoff Ratio',0)
                    so_fg,so_bg,so_r=rating(so,1.0,0.5)
                    ca_fg,ca_bg,ca_r=rating(ca,0.5,0.2)
                    rf_fg,rf_bg,rf_r=rating(rf,2.0,1.0)
                    pf_fg,pf_bg,pf_r=rating(pf,1.5,1.0)
                    pr_fg,pr_bg,pr_r=rating(pr,1.5,1.0)
                    st.markdown(
                        '<div class="dpanel accent-o"><div class="dpanel-title">Risk-Adjusted Ratios</div>'+
                        rb("Sortino Ratio",so,3.0,so_fg,so_bg,so_r)+
                        rb("Calmar Ratio",ca,2.0,ca_fg,ca_bg,ca_r)+
                        rb("Recovery Factor",rf,5.0,rf_fg,rf_bg,rf_r)+
                        rb("Profit Factor",pf,3.0,pf_fg,pf_bg,pf_r)+
                        rb("Payoff Ratio",pr,3.0,pr_fg,pr_bg,pr_r)+
                        '</div>', unsafe_allow_html=True)
                    vol=met.get('Annualized Volatility (%)',0); var=met.get('VaR 95% (%)',0)
                    avgd=met.get('Avg Drawdown (%)',0); bd=met.get('Best Day (%)',0); wd=met.get('Worst Day (%)',0)
                    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="dpanel accent-b"><div class="dpanel-title">Risk Metrics</div>'+
                        sr("Annualised Volatility",f"{vol:.2f}%")+
                        sr("Value at Risk (95%)",f"{var:.2f}%",'r')+
                        sr("Average Drawdown",f"{avgd:.2f}%",'r')+
                        sr("Best Day",f"+{bd:.2f}%",'g')+
                        sr("Worst Day",f"{wd:.2f}%",'r')+
                        '</div>', unsafe_allow_html=True)

                with col_r:
                    ex=met.get('Expectancy ($)',0); dur=met.get('Avg Trade Duration (hrs)',0)
                    fees=met.get('Total Fees Paid ($)',0); gp=met.get('Gross Profit ($)',0)
                    gl=met.get('Gross Loss ($)',0); bt=met.get('Best Trade ($)',0)
                    wt=met.get('Worst Trade ($)',0); ws=met.get('Win Streak',0); ls=met.get('Loss Streak',0)
                    wins_n=int(round(trades*wr/100)); losses_n=trades-wins_n
                    wp=max(wr,1); lp=max(100-wr,1)
                    st.markdown(
                        '<div class="dpanel accent-g"><div class="dpanel-title">Win / Loss Distribution</div>'+
                        f'<div class="wl-bar"><div class="wl-w" style="width:{wp}%">{wins_n}W</div>'
                        f'<div class="wl-l" style="width:{lp}%">{losses_n}L</div></div>'+
                        sr("Gross Profit",f"${gp:,.2f}",'g')+sr("Gross Loss",f"${gl:,.2f}",'r')+
                        sr("Best Trade",f"${bt:,.2f}",'g')+sr("Worst Trade",f"${wt:,.2f}",'r')+
                        sr("Win Streak",str(ws),'g')+sr("Loss Streak",str(ls),'r')+
                        '</div>', unsafe_allow_html=True)
                    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="dpanel accent-r"><div class="dpanel-title">Trade Statistics</div>'+
                        sr("Expectancy per Trade",f"${ex:+.2f}",'g' if ex>0 else 'r')+
                        sr("Avg Trade Duration",f"{dur:.0f} hrs")+
                        sr("Total Fees Paid",f"${fees:,.2f}",'r')+
                        sr("Total Trades",str(trades))+
                        '</div>', unsafe_allow_html=True)

                # ── Trade Log ────────────────────────────────────────────────
                if not tr_df.empty:
                    st.markdown('<div class="sh"><span>05</span>TRADE LOG</div>', unsafe_allow_html=True)
                    table_html, has_open, open_price = build_trade_log(tr_df)
                    st.markdown(table_html, unsafe_allow_html=True)
                    if has_open and open_price is not None:
                        st.markdown(
                            f'<div class="tl-open">⚠ 1 open position — entry at ${open_price:,.2f} (not yet closed)</div>',
                            unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    else:
        fig=go.Figure(go.Scatter(x=df['timestamp'],y=df['close'],mode='lines',
            line=dict(color='#ff6600',width=1.5),fill='tozeroy',fillcolor='rgba(255,102,0,0.04)'))
        fig.update_layout(paper_bgcolor='#0a0a0a',plot_bgcolor='#111111',
            font=dict(family='IBM Plex Mono',color='#555555',size=10),
            height=420,margin=dict(l=0,r=0,t=4,b=0))
        fig.update_xaxes(gridcolor='#1a1a1a',zeroline=False)
        fig.update_yaxes(gridcolor='#1a1a1a',zeroline=False)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#333333;'
                    'text-align:center;padding:20px 0">Select asset and strategy — then click Run Backtest</div>',
                    unsafe_allow_html=True)