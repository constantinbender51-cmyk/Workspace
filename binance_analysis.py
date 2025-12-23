import os
import io
import base64
import logging
import requests
import json
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template

# --- Configuration ---
SYMBOL = "BTCUSDT"
LOOKBACK_DAYS = 365
CAP_SPLIT = 0.333
PLANNER_PARAMS = {"S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, "S2_SMA": 400}
TUMBLER_PARAMS = {"SMA1": 32, "SMA2": 114, "STOP": 0.043, "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]}
GAINER_PARAMS = {"WEIGHTS": [0.8, 0.4], "MACD_1H": {'params': [(97, 366, 47)]}, "MACD_1D": {'params': [(52, 64, 61)]}}
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester")

# --- Global Cache ---
# Stores the latest analysis results so we don't re-run on every page load
GLOBAL_CACHE = {
    "stats": None,
    "plots": None,
    "last_updated": None,
    "is_updating": False
}

# --- Data Fetching ---
def fetch_binance_klines(symbol, interval, days):
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
            if not isinstance(data, list) or len(data) == 0: break
            
            all_data.extend(data)
            last_time = data[-1][0]
            current_start = last_time + 1
            if len(data) < 1000 or current_start >= end_time: break
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df[['close']]

# --- Strategy Helpers ---
def get_sma(series, window):
    return series.rolling(window=window).mean()

def calc_planner(df_1d, current_date, price, state, capital=10000):
    entry_date = state.get("entry_date")
    stopped = state.get("stopped", False)
    peak_equity = state.get("peak_equity", capital)
    
    sma120 = df_1d.at[current_date, 'SMA_120']
    sma400 = df_1d.at[current_date, 'SMA_400']
    
    if pd.isna(sma120) or pd.isna(sma400): return 0.0, state
    if peak_equity < capital: peak_equity = capital
    
    s1_lev = 0.0
    if price > sma120:
        if not stopped:
            if not entry_date: entry_date = current_date
            days = (current_date - entry_date).days
            s1_lev = 1.0 * max(0.0, 1.0 - (days / PLANNER_PARAMS["S1_DECAY"])**2)
    else:
        stopped, peak_equity, entry_date = False, capital, None
        s1_lev = -1.0
        
    s2_lev = 1.0 if price > sma400 else 0.0
    lev = max(-2.0, min(2.0, s1_lev + s2_lev))
    state.update({"entry_date": entry_date, "stopped": stopped, "peak_equity": peak_equity})
    return lev, state

def calc_tumbler(df_1d, current_date, price, state):
    flat_regime = state.get("flat_regime", False)
    iii = df_1d.at[current_date, 'III']
    sma1 = df_1d.at[current_date, 'T_SMA1']
    sma2 = df_1d.at[current_date, 'T_SMA2']
    
    if pd.isna(iii) or pd.isna(sma1): return 0.0, state
    
    lev = TUMBLER_PARAMS["LEVS"][2]
    if iii < TUMBLER_PARAMS["III_TH"][0]: lev = TUMBLER_PARAMS["LEVS"][0]
    elif iii < TUMBLER_PARAMS["III_TH"][1]: lev = TUMBLER_PARAMS["LEVS"][1]
    
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]: flat_regime = True
    if flat_regime and abs(price - sma1) <= sma1 * TUMBLER_PARAMS["BAND"]: flat_regime = False
            
    if flat_regime:
        state["flat_regime"] = True
        return 0.0, state
        
    state["flat_regime"] = False
    return (lev if price > sma1 and price > sma2 else (-lev if price < sma1 and price < sma2 else 0.0)), state

def calc_gainer(df_1h, df_1d, idx_1h, idx_1d):
    sig_1h = df_1h.at[idx_1h, 'MACD_SIG']
    sig_1d = df_1d.at[idx_1d, 'MACD_SIG']
    if pd.isna(sig_1h) or pd.isna(sig_1d): return 0.0
    w = GAINER_PARAMS["WEIGHTS"]
    return (sig_1h * w[0] + sig_1d * w[1]) / sum(w)

def prepare_data():
    df_1h = fetch_binance_klines(SYMBOL, "1h", LOOKBACK_DAYS + 60)
    df_1d = fetch_binance_klines(SYMBOL, "1d", LOOKBACK_DAYS + 60)
    
    # 1D Indicators
    df_1d['SMA_120'] = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    df_1d['SMA_400'] = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])
    df_1d['T_SMA1'] = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
    df_1d['T_SMA2'] = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    
    w = TUMBLER_PARAMS["III_WIN"]
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    df_1d['III'] = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0)
    
    # MACD 1D
    f, s, sig = GAINER_PARAMS["MACD_1D"]['params'][0]
    fast = df_1d['close'].ewm(span=f).mean()
    slow = df_1d['close'].ewm(span=s).mean()
    df_1d['MACD_SIG'] = np.where((fast - slow) > (fast - slow).ewm(span=sig).mean(), 1.0, -1.0)
    
    # MACD 1H
    f, s, sig = GAINER_PARAMS["MACD_1H"]['params'][0]
    fast = df_1h['close'].ewm(span=f).mean()
    slow = df_1h['close'].ewm(span=s).mean()
    df_1h['MACD_SIG'] = np.where((fast - slow) > (fast - slow).ewm(span=sig).mean(), 1.0, -1.0)
    
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
    df_1h = df_1h[df_1h.index > cutoff]
    df_1d_aligned = df_1d.reindex(df_1h.index, method='ffill')
    
    return df_1h, df_1d_aligned

def run_backtest_pipeline():
    logger.info("Starting Backtest Pipeline...")
    df_1h, df_1d = prepare_data()
    
    planner_state = {"entry_date": None, "stopped": False, "peak_equity": 0.0}
    tumbler_state = {"flat_regime": False}
    equity = 10000.0
    results = []
    
    for i in range(len(df_1h)):
        idx = df_1h.index[i]
        price = df_1h['close'].iloc[i]
        
        lev_p, planner_state = calc_planner(df_1d, idx, price, planner_state)
        lev_t, tumbler_state = calc_tumbler(df_1d, idx, price, tumbler_state)
        lev_g = calc_gainer(df_1h, df_1d, idx, idx)
        
        total_lev = lev_p + (lev_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)) + (lev_g * TARGET_STRAT_LEV)
        
        if i < len(df_1h) - 1:
            equity += equity * total_lev * ((df_1h['close'].iloc[i+1] - price) / price)
        
        results.append({"timestamp": idx, "equity": equity, "price": price})
        
    df = pd.DataFrame(results).set_index("timestamp")
    
    # --- Fix: Calculate Drawdown Immediately ---
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    return df

def generate_analytics():
    """Main worker function to update GLOBAL_CACHE"""
    global GLOBAL_CACHE
    if GLOBAL_CACHE['is_updating']: return
    GLOBAL_CACHE['is_updating'] = True
    
    try:
        df = run_backtest_pipeline()
        
        # 1. Metrics
        initial = df['equity'].iloc[0]
        final = df['equity'].iloc[-1]
        hourly_returns = df['equity'].pct_change().dropna()
        
        stats = {
            "return": f"{(final - initial) / initial * 100:.2f}%",
            "equity": f"${final:,.2f}",
            "sharpe": f"{(hourly_returns.mean() * 8760) / (hourly_returns.std() * np.sqrt(8760)):.2f}",
            "volatility": f"{hourly_returns.std() * np.sqrt(8760) * 100:.2f}%",
            "max_dd": f"{df['drawdown'].min() * 100:.2f}%"
        }
        
        # 2. Plots
        # Equity Curve
        fig, ax1 = plt.subplots(figsize=(10, 6))
        initial_price = df['price'].iloc[0]
        df['benchmark'] = (df['price'] / initial_price) * df['equity'].iloc[0]
        
        ax1.plot(df.index, df['equity'], label='Strategy', color='#00ff00', linewidth=1.5)
        ax1.plot(df.index, df['benchmark'], label='BTC Hold', color='gray', alpha=0.5, linestyle='--')
        
        ax1.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#121212')
        ax1.tick_params(colors='white')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.grid(True, color='#333333', linestyle=':')
        plt.legend(facecolor='#333333', labelcolor='white')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_eq = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Drawdown
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        ax2.plot(df.index, df['drawdown'], color='red', linewidth=1)
        
        ax2.set_facecolor('#1e1e1e')
        fig2.patch.set_facecolor('#121212')
        ax2.tick_params(colors='white')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.grid(True, color='#333333', linestyle=':')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        plot_dd = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)
        
        GLOBAL_CACHE['stats'] = stats
        GLOBAL_CACHE['plots'] = {'equity': plot_eq, 'drawdown': plot_dd}
        GLOBAL_CACHE['last_updated'] = datetime.now()
        logger.info("Analytics updated successfully.")
        
    except Exception as e:
        logger.error(f"Failed update: {e}")
    finally:
        GLOBAL_CACHE['is_updating'] = False

def background_updater():
    """Updates cache every 6 hours"""
    while True:
        time.sleep(21600) # 6 hours
        logger.info("Running periodic update...")
        generate_analytics()

# --- Routes ---
@app.route('/')
def dashboard():
    if not GLOBAL_CACHE['stats']:
        return """
        <div style="font-family:sans-serif; color:#fff; background:#121212; height:100vh; display:flex; align-items:center; justify-content:center;">
            <h1>Initializing Analytics... Please refresh in 10 seconds.</h1>
        </div>
        """
    return render_template('index.html', stats=GLOBAL_CACHE['stats'], 
                           plot_equity=GLOBAL_CACHE['plots']['equity'], 
                           plot_drawdown=GLOBAL_CACHE['plots']['drawdown'])

# --- Startup ---
# Run once synchronously on start so data is ready immediately
logger.info("Server starting - Pre-loading analytics...")
generate_analytics()

# Start background thread for future updates
t = threading.Thread(target=background_updater, daemon=True)
t.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
