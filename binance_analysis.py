import os
import io
import base64
import logging
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template

# --- Configuration ---
SYMBOL = "BTCUSDT"
LOOKBACK_DAYS = 365
# Strategy Constants from your file
CAP_SPLIT = 0.333
PLANNER_PARAMS = {"S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, "S2_SMA": 400}
TUMBLER_PARAMS = {"SMA1": 32, "SMA2": 114, "STOP": 0.043, "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]}
GAINER_PARAMS = {"WEIGHTS": [0.8, 0.4], "MACD_1H": {'params': [(97, 366, 47)]}, "MACD_1D": {'params': [(52, 64, 61)]}}
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester")

# --- Data Fetching ---
def fetch_binance_klines(symbol, interval, days):
    """Fetches historical OHLC data from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    logger.info(f"Fetching {interval} data for {symbol}...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                break
            
            all_data.extend(data)
            last_time = data[-1][0]
            current_start = last_time + 1
            
            if len(data) < 1000 or current_start >= end_time:
                break
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df[['close']]

# --- Strategy Logic (Re-implementation) ---
def get_sma(series, window):
    return series.rolling(window=window).mean()

def calc_planner(df_1d, current_date, price, state, capital=10000):
    # Slice data up to current_date to avoid lookahead, but 1D data implies yesterday's close usually
    # For backtest efficiency, we pre-calculate indicators on the full DF and lookup by index
    # Here we assume df_1d has indicators pre-calculated
    
    # State recovery
    entry_date = state.get("entry_date")
    stopped = state.get("stopped", False)
    peak_equity = state.get("peak_equity", capital)
    
    sma120 = df_1d.at[current_date, 'SMA_120']
    sma400 = df_1d.at[current_date, 'SMA_400']
    
    if pd.isna(sma120) or pd.isna(sma400):
        return 0.0, state
        
    if peak_equity < capital: peak_equity = capital
    
    s1_lev = 0.0
    if price > sma120:
        if not stopped:
            if not entry_date: 
                entry_date = current_date
            
            # Calculate days since entry
            days = (current_date - entry_date).days
            s1_lev = 1.0 * max(0.0, 1.0 - (days / PLANNER_PARAMS["S1_DECAY"])**2)
    else:
        stopped = False
        peak_equity = capital
        entry_date = None # Reset
        s1_lev = -1.0
        
    s2_lev = 1.0 if price > sma400 else 0.0
    lev = max(-2.0, min(2.0, s1_lev + s2_lev))
    
    state["entry_date"] = entry_date
    state["stopped"] = stopped
    state["peak_equity"] = peak_equity
    
    return lev, state

def calc_tumbler(df_1d, current_date, price, state):
    flat_regime = state.get("flat_regime", False)
    
    # Lookup pre-calc values
    iii = df_1d.at[current_date, 'III']
    sma1 = df_1d.at[current_date, 'T_SMA1']
    sma2 = df_1d.at[current_date, 'T_SMA2']
    
    if pd.isna(iii) or pd.isna(sma1): return 0.0, state
    
    lev = TUMBLER_PARAMS["LEVS"][2]
    if iii < TUMBLER_PARAMS["III_TH"][0]: lev = TUMBLER_PARAMS["LEVS"][0]
    elif iii < TUMBLER_PARAMS["III_TH"][1]: lev = TUMBLER_PARAMS["LEVS"][1]
    
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]: flat_regime = True
    
    if flat_regime:
        if abs(price - sma1) <= sma1 * TUMBLER_PARAMS["BAND"]:
            flat_regime = False
            
    if flat_regime:
        state["flat_regime"] = True
        return 0.0, state
        
    state["flat_regime"] = False
    
    if price > sma1 and price > sma2:
        return lev, state
    elif price < sma1 and price < sma2:
        return -lev, state
    
    return 0.0, state

def calc_gainer(df_1h, df_1d, idx_1h, idx_1d):
    # MACD logic
    # Pre-calc MACD signals in dataframe preparation to speed up
    sig_1h = df_1h.at[idx_1h, 'MACD_SIG']
    sig_1d = df_1d.at[idx_1d, 'MACD_SIG']
    
    if pd.isna(sig_1h) or pd.isna(sig_1d): return 0.0
    
    w = GAINER_PARAMS["WEIGHTS"]
    return (sig_1h * w[0] + sig_1d * w[1]) / sum(w)

def prepare_data():
    df_1h = fetch_binance_klines(SYMBOL, "1h", LOOKBACK_DAYS + 60) # Buffer for MA
    df_1d = fetch_binance_klines(SYMBOL, "1d", LOOKBACK_DAYS + 60)
    
    # --- Pre-calculate Indicators for 1D (Planner & Tumbler) ---
    df_1d['SMA_120'] = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    df_1d['SMA_400'] = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])
    
    # Tumbler Indicators
    df_1d['T_SMA1'] = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
    df_1d['T_SMA2'] = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    
    # Tumbler III
    w = TUMBLER_PARAMS["III_WIN"]
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    df_1d['III'] = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0)
    
    # Gainer MACD 1D
    f, s, sig = GAINER_PARAMS["MACD_1D"]['params'][0]
    fast = df_1d['close'].ewm(span=f).mean()
    slow = df_1d['close'].ewm(span=s).mean()
    macd = fast - slow
    signal = macd.ewm(span=sig).mean()
    df_1d['MACD_SIG'] = np.where(macd > signal, 1.0, -1.0)
    
    # --- Pre-calculate Indicators for 1H (Gainer) ---
    f, s, sig = GAINER_PARAMS["MACD_1H"]['params'][0]
    fast = df_1h['close'].ewm(span=f).mean()
    slow = df_1h['close'].ewm(span=s).mean()
    macd = fast - slow
    signal = macd.ewm(span=sig).mean()
    df_1h['MACD_SIG'] = np.where(macd > signal, 1.0, -1.0)
    
    # Trim to analysis period (last 365 days)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
    df_1h = df_1h[df_1h.index > cutoff]
    
    # Resample 1D to match 1H index for easy lookup (ffill)
    # This aligns the daily indicators to the hourly timestamps
    df_1d_aligned = df_1d.reindex(df_1h.index, method='ffill')
    
    return df_1h, df_1d_aligned

def run_backtest():
    df_1h, df_1d = prepare_data()
    
    planner_state = {"entry_date": None, "stopped": False, "peak_equity": 0.0}
    tumbler_state = {"flat_regime": False}
    
    results = []
    
    # Initial Equity
    equity = 10000.0
    
    for i in range(len(df_1h)):
        idx = df_1h.index[i]
        price = df_1h['close'].iloc[i]
        
        # Run Strategies
        # Note: We pass the *aligned* daily data row corresponding to this hour
        lev_p, planner_state = calc_planner(df_1d, idx, price, planner_state)
        lev_t, tumbler_state = calc_tumbler(df_1d, idx, price, tumbler_state)
        lev_g = calc_gainer(df_1h, df_1d, idx, idx)
        
        # Normalize
        n_p = lev_p
        n_t = lev_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        n_g = lev_g * TARGET_STRAT_LEV
        
        total_lev = n_p + n_t + n_g
        
        # Simple Return Calculation (Next Hour Return)
        if i < len(df_1h) - 1:
            next_price = df_1h['close'].iloc[i+1]
            ret = (next_price - price) / price
            # Apply Leverage
            pnl = equity * total_lev * ret
            equity += pnl
        
        results.append({
            "timestamp": idx,
            "equity": equity,
            "price": price,
            "lev_total": total_lev,
            "lev_p": n_p,
            "lev_t": n_t,
            "lev_g": n_g
        })
        
    return pd.DataFrame(results).set_index("timestamp")

def generate_plots(df):
    # Equity Curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Benchmark (Buy and Hold)
    initial_price = df['price'].iloc[0]
    initial_equity = df['equity'].iloc[0]
    df['benchmark'] = (df['price'] / initial_price) * initial_equity
    
    ax1.plot(df.index, df['equity'], label='Strategy Equity', color='#00ff00', linewidth=1.5)
    ax1.plot(df.index, df['benchmark'], label='Buy & Hold (BTC)', color='gray', alpha=0.5, linestyle='--')
    
    ax1.set_title(f"Strategy Performance ({LOOKBACK_DAYS} Days)", fontsize=14, color='white')
    ax1.set_ylabel("Equity ($)", color='white')
    ax1.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#121212')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    
    legend = ax1.legend(facecolor='#333333', edgecolor='none')
    plt.setp(legend.get_texts(), color='white')
    
    # Format Date
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.grid(True, color='#333333', linestyle=':')
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_equity = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Drawdown Plot
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(df.index, df['drawdown'], color='red', linewidth=1)
    ax2.set_title("Drawdown", fontsize=12, color='white')
    ax2.set_facecolor('#1e1e1e')
    fig2.patch.set_facecolor('#121212')
    ax2.tick_params(colors='white')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.grid(True, color='#333333', linestyle=':')
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    plot_drawdown = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()

    return plot_equity, plot_drawdown

# --- Routes ---
@app.route('/')
def dashboard():
    # Cache checking could go here, running fresh for now
    try:
        results = run_backtest()
        
        # Metrics
        initial = results['equity'].iloc[0]
        final = results['equity'].iloc[-1]
        total_return = (final - initial) / initial * 100
        
        # Annualized Volatility (Hourly to Yearly)
        hourly_returns = results['equity'].pct_change().dropna()
        volatility = hourly_returns.std() * np.sqrt(365 * 24) * 100
        
        # Sharpe (assuming 0 risk free)
        sharpe = (hourly_returns.mean() * 24 * 365) / (hourly_returns.std() * np.sqrt(24 * 365))
        
        # Max Drawdown
        max_dd = results['drawdown'].min() * 100
        
        plot_eq, plot_dd = generate_plots(results)
        
        stats = {
            "return": f"{total_return:.2f}%",
            "equity": f"${final:,.2f}",
            "sharpe": f"{sharpe:.2f}",
            "volatility": f"{volatility:.2f}%",
            "max_dd": f"{max_dd:.2f}%"
        }
        
        return render_template('index.html', stats=stats, plot_equity=plot_eq, plot_drawdown=plot_dd)
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        return f"Error generating analytics: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
